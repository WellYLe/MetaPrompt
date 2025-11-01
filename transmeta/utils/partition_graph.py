# partition_graph.py
import math
import numpy as np
import random
from sklearn.cluster import KMeans
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph, to_undirected, degree
import torch.nn.functional as F
from typing import List, Optional, Union


def _validate_inputs(data: Data, num_parts: int, sim_metric: str):
    if num_parts <= 0:
        raise ValueError("num_parts must be > 0")
    if data.num_nodes is None:
        raise ValueError("data.num_nodes cannot be None")
    if num_parts > data.num_nodes:
        raise ValueError("num_parts cannot exceed number of nodes")
    if data.x is None:
        raise ValueError("data.x 不能为空（需要节点特征计算相似度）")
    if sim_metric not in ('cosine', 'euclidean'):
        raise ValueError("sim_metric must be 'cosine' or 'euclidean'")

def _make_subdata_from_indices(data: Data, subset_idx: torch.Tensor) -> Data:
    """给定原图和节点索引（长整形 tensor），返回 relabeled 的 Data，并保留 orig_node_idx"""
    n = data.num_nodes
    subset_idx = subset_idx.clone().to(torch.long)
    ei, edge_mask = subgraph(subset_idx, data.edge_index, relabel_nodes=True, num_nodes=n)
    sub_data = Data()
    if hasattr(data, 'x') and data.x is not None:
        sub_data.x = data.x[subset_idx]
    if hasattr(data, 'y') and data.y is not None:
        sub_data.y = data.y[subset_idx]
    # masks
    for k in ('train_mask', 'val_mask', 'test_mask'):
        if hasattr(data, k):
            orig_mask = getattr(data, k)
            if orig_mask is not None:
                sub_data[k] = orig_mask[subset_idx].clone()
    sub_data.edge_index = ei
    sub_data.num_nodes = subset_idx.size(0)
    sub_data.orig_node_idx = subset_idx.clone()
    return sub_data


# -----------------------
# Method A: Greedy "far-from-center" (embedding-driven)
# -----------------------
def partition_graph_equal_a(data: Data, num_parts: int, shuffle: bool = True, seed: int = None, sim_metric='cosine') -> List[Data]:
    """
    Method A: 嵌入驱动的贪婪“远离中心”采样
    思路：
      - 用节点 embedding（data.x）作为表示，按 cos/euclidean 计算相似度。
      - 初始化每个 part 一个 seed（随机或按 degree / PPR 可替换）。
      - 逐个向每个分区添加最能降低该分区平均相似度且不显著增加内部边的节点。
    复杂度：O(n * num_parts * cand)（cand 为候选池大小，默认用全部节点或可抽样）
    返回：list[Data]，每个 sub_data 含 orig_node_idx
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    _validate_inputs(data, num_parts, sim_metric)

    n = data.num_nodes
    x = data.x
    if sim_metric == 'cosine':
        xn = F.normalize(x, dim=-1)
        sim_mat = torch.matmul(xn, xn.t())  # cosine similarity
    else:
        dist = torch.cdist(x, x)
        sim_mat = -dist  # treat negative distance as similarity (larger = more similar)

    perm = torch.randperm(n) if shuffle else torch.arange(n)
    seeds = perm[:num_parts].tolist()
    clusters = [[int(s)] for s in seeds]
    assigned = set(seeds)
    remaining = [int(i) for i in perm[num_parts:].tolist()]

    # For efficiency, we'll maintain per-cluster sum of sim vectors for avg computation
    cluster_sum_sim = []
    for c in clusters:
        cluster_sum_sim.append(sim_mat[c[0], :].clone())

    # Candidate strategy: consider remaining directly (works fine for n up to several 10k),
    # or sample subset for very large graphs.
    while remaining:
        v = remaining.pop(0)  # FIFO; could use probabilistic selection
        # compute score for adding v to each cluster:
        best_score = None
        best_k = None
        for k, c in enumerate(clusters):
            # new average similarity if add v: approx by mean over sims to members
            if len(c) == 0:
                new_avg = sim_mat[v, :].mean().item()
            else:
                # compute mean similarity between v and cluster members
                mem = torch.tensor(c, dtype=torch.long)
                new_avg = float(sim_mat[v, mem].mean().item())
            # density increment: how many new internal edges would arise (approx)
            # we check existing adjacency: count edges between v and cluster members
            # build neighbor set quickly by checking data.edge_index
            # for speed, precompute adjacency set? here we compute on the fly (ok for modest sizes)
            # We'll approximate using edge_index membership test (vectorized)
            # Build mask where members appear in edge_index rows
            # Convert members to set for fast python check
            # (Keep simple: prefer low avg similarity)
            score = new_avg
            if best_score is None or score < best_score:
                best_score = score
                best_k = k
        clusters[best_k].append(v)
        # update cluster_sum_sim (not strictly necessary here)
        # cluster_sum_sim[best_k] += sim_mat[v, :]  # keep but not used further

    parts = []
    for c in clusters:
        subset = torch.tensor(c, dtype=torch.long)
        sub = _make_subdata_from_indices(data, subset)
        parts.append(sub)
    return parts


# -----------------------
# Method B: Cluster-cross sampling (KMeans + cross-cluster mixing)
# -----------------------
def partition_graph_equal_b(data: Data, num_parts: int, shuffle: bool = True, seed: int = None, sim_metric='cosine') -> List[Data]:
    """
    Method B: 基于聚类但“远离簇中心”选取 / 跨簇混合
    思路：
      - 在 node embed 上做 K_means（k = num_parts 或更大），
      - 为了让每个划分内部异质，采用跨多个簇取样策略（例如每个分区从不同聚类中抽取节点）。
      - 同时避免选出相互直接相连的节点以降低内部连通度。
    复杂度：KMeans O(n * k * iter)
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    _validate_inputs(data, num_parts, sim_metric)
    n = data.num_nodes

    x = data.x.cpu().numpy()
    # 取 cluster_count = min(num_parts*2, n) 为经验值，用更多小簇以便跨簇混合
    cluster_count = min(max(2 * num_parts, num_parts), n)
    kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(x)  # numpy array size n

    # group nodes by cluster
    clusters_by_k = {}
    for i, lab in enumerate(labels):
        clusters_by_k.setdefault(lab, []).append(i)

    # target: for each desired part, sample nodes from different kmeans-clusters
    node_pool = list(range(n))
    if shuffle:
        random.shuffle(node_pool)

    # compute desired size for each part (balanced)
    base = n // num_parts
    remainder = n % num_parts
    sizes = [base + (1 if i < remainder else 0) for i in range(num_parts)]

    parts_idx = [[] for _ in range(num_parts)]
    used = set()

    # Precompute adjacency set to avoid internal adjacency
    edge_index = data.edge_index.cpu().numpy()
    adj_set = {i: set() for i in range(n)}
    for a, b in zip(edge_index[0], edge_index[1]):
        adj_set[int(a)].add(int(b))
        adj_set[int(b)].add(int(a))

    # For each part, sample from as many distinct kmeans-clusters as possible
    k_labels = list(range(cluster_count))
    for part_id in range(num_parts):
        target_size = sizes[part_id]
        # preference: sample from clusters in round-robin to maximize heterogeneity
        kk = 0
        attempts = 0
        while len(parts_idx[part_id]) < target_size and attempts < n * 2:
            lab = k_labels[kk % cluster_count]
            kk += 1
            attempts += 1
            cand_list = clusters_by_k.get(lab, [])
            # choose candidate not used and with minimal adjacency to already chosen nodes
            random.shuffle(cand_list)
            chosen = None
            for cand in cand_list:
                if cand in used:
                    continue
                # avoid picking nodes that are directly adjacent to many nodes already in this part
                if any((cand in adj_set[other]) for other in parts_idx[part_id]):
                    continue
                chosen = cand
                break
            if chosen is None:
                # fallback: pick any unused
                for cand in cand_list:
                    if cand not in used:
                        chosen = cand
                        break
            if chosen is None:
                # pick any unused node in whole graph
                for cand in node_pool:
                    if cand not in used:
                        chosen = cand
                        break
            if chosen is None:
                # all used, break
                break
            parts_idx[part_id].append(chosen)
            used.add(chosen)

    # If any nodes remain unused, distribute them greedily to parts with smallest internal avg similarity
    unused_nodes = [i for i in range(n) if i not in used]
    # Precompute embeddings and normalized for similarity metric
    xt = data.x
    if sim_metric == 'cosine':
        xtn = F.normalize(xt, dim=-1)
    else:
        xtn = xt

    for v in unused_nodes:
        best_part = None
        best_score = None
        for pid in range(num_parts):
            if len(parts_idx[pid]) == 0:
                score = 0
            else:
                mem = torch.tensor(parts_idx[pid], dtype=torch.long)
                if sim_metric == 'cosine':
                    score = float((xtn[v:v+1] @ xtn[mem].t()).mean().item())
                else:
                    score = float(-torch.cdist(xtn[v:v+1], xtn[mem]).mean().item())  # negative dist as similarity
            if best_score is None or score < best_score:
                best_score = score
                best_part = pid
        parts_idx[best_part].append(v)
        used.add(v)

    parts = []
    for idx_list in parts_idx:
        subset = torch.tensor(idx_list, dtype=torch.long)
        sub = _make_subdata_from_indices(data, subset)
        parts.append(sub)
    return parts


# -----------------------
# Method C: Local-search approximate optimization (swap-based improvement)
# -----------------------
def partition_graph_equal_c(data: Data, num_parts: int, shuffle: bool = True, seed: int = None, sim_metric='cosine') -> List[Data]:
    """
    Method C: 用局部搜索（交换/贪心改进）近似求解一个明确的目标：
      目标函数 Score(S) = alpha * OOD(S) - beta * Density(S) - gamma * Sim(S)
    这里我们把 problem 表达为对每个分区选择节点（均匀大小），
    初始化用随机/greedy，然后做 pairwise swap 改进直到无提升或达到迭代上限。
    优点：可在中等规模上获得较好近似，适合 k=几十 的精致求解。
    注意：并非全局最优，但通常稳定且可解释。
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    _validate_inputs(data, num_parts, sim_metric)
    n = data.num_nodes
    x = data.x
    device = x.device

    # similarity matrix (小心内存：n x n)
    if sim_metric == 'cosine':
        xn = F.normalize(x, dim=-1)
        sim_mat = (xn @ xn.t()).cpu()
    else:
        dist = torch.cdist(x, x).cpu()
        sim_mat = -dist

    # adjacency matrix (sparse) for density calc
    edge_index = data.edge_index.cpu()
    adj = torch.zeros((n, n), dtype=torch.uint8)
    for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[a, b] = 1
        adj[b, a] = 1  # undirected assumption

    # partition sizes
    base = n // num_parts
    remainder = n % num_parts
    sizes = [base + (1 if i < remainder else 0) for i in range(num_parts)]

    # initialize partitions: use simple round-robin of perm (balanced)
    perm = torch.randperm(n) if shuffle else torch.arange(n)
    parts_idx = [[] for _ in range(num_parts)]
    ptr = 0
    for i in range(n):
        pid = ptr % num_parts
        if len(parts_idx[pid]) < sizes[pid]:
            parts_idx[pid].append(int(perm[i].item()))
        else:
            ptr += 1
            pid = ptr % num_parts
            parts_idx[pid].append(int(perm[i].item()))

    # helper scoring functions (we define Sim = avg pairwise sim, Density = edge_density)
    def part_sim(pid):
        mem = parts_idx[pid]
        if len(mem) <= 1:
            return 0.0
        mem_idx = torch.tensor(mem, dtype=torch.long)
        sims = sim_mat[mem_idx][:, mem_idx].float()
        # exclude diagonal
        ssum = sims.sum().item() - sims.diag().sum().item()
        pairs = len(mem) * (len(mem) - 1)
        return ssum / pairs

    def part_density(pid):
        mem = parts_idx[pid]
        if len(mem) <= 1:
            return 0.0
        cnt = 0
        for i in mem:
            for j in mem:
                if i < j:
                    if adj[i, j]:
                        cnt += 1
        # density normalized by possible edges
        denom = len(mem) * (len(mem) - 1) / 2
        return cnt / denom if denom > 0 else 0.0

    # define global objective (sum over parts) — we want to minimize Sim + lambda * Density - mu * OOD
    alpha = 1.0  # OOD weight (we approximate as distance from global mean)
    beta = 1.0   # density weight
    gamma = 1.0  # sim weight

    # precompute global mean and cov approx for OOD (use mean vector)
    global_mean = x.mean(dim=0).cpu()

    def part_oood(pid):
        mem = parts_idx[pid]
        if len(mem) == 0:
            return 0.0
        mem_idx = torch.tensor(mem, dtype=torch.long)
        mmean = x[mem_idx].mean(dim=0).cpu()
        # use simple L2 distance from global_mean as OOD proxy
        return float(torch.norm(mmean - global_mean).item())

    def global_score():
        tot = 0.0
        for pid in range(num_parts):
            tot += gamma * part_sim(pid) + beta * part_density(pid) - alpha * part_oood(pid)
        return tot

    # local search: try swapping pairs between different parts to reduce global_score
    max_iters = 2000
    improved = True
    it = 0
    cur_score = global_score()
    while improved and it < max_iters:
        improved = False
        it += 1
        # iterate over random pairs of parts and random nodes within them
        part_order = list(range(num_parts))
        random.shuffle(part_order)
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                A = part_order[i]
                B = part_order[j]
                # try all pairs of nodes (if parts small, ok); otherwise sample
                Ai = parts_idx[A]
                Bi = parts_idx[B]
                sample_pairs = []
                if len(Ai) * len(Bi) <= 200:
                    for a in Ai:
                        for b in Bi:
                            sample_pairs.append((a, b))
                else:
                    # sample limited pairs
                    for _ in range(200):
                        sample_pairs.append((random.choice(Ai), random.choice(Bi)))
                swapped = False
                for a, b in sample_pairs:
                    # perform swap a<->b and evaluate change cheaply by recomputing scores for A and B
                    # backup
                    # do swap
                    Ai_idx = Ai.index(a)
                    Bi_idx = Bi.index(b)
                    Ai[Ai_idx], Bi[Bi_idx] = b, a
                    new_score = 0.0
                    # compute local score for A and B
                    new_score += gamma * part_sim(A) + beta * part_density(A) - alpha * part_oood(A)
                    new_score += gamma * part_sim(B) + beta * part_density(B) - alpha * part_oood(B)
                    # compute old local
                    # to get old, swap back temporarily
                    Ai[Ai_idx], Bi[Bi_idx] = a, b
                    old_score = gamma * part_sim(A) + beta * part_density(A) - alpha * part_oood(A)
                    old_score += gamma * part_sim(B) + beta * part_density(B) - alpha * part_oood(B)
                    # if improvement (new < old), accept swap
                    if new_score < old_score - 1e-9:
                        Ai[Ai_idx], Bi[Bi_idx] = b, a  # keep swapped
                        cur_score = cur_score - old_score + new_score
                        improved = True
                        swapped = True
                        break
                    else:
                        # revert (already reverted)
                        pass
                if swapped:
                    # break to outer loop to restart search order (first improvement)
                    break
            if improved:
                break

    # build parts
    parts = []
    for pid in range(num_parts):
        subset = torch.tensor(parts_idx[pid], dtype=torch.long)
        sub = _make_subdata_from_indices(data, subset)
        parts.append(sub)
    return parts


# -----------------------
# Method D: "Learning-style" selector (pseudo-learned heuristic)
# -----------------------
def partition_graph_equal_d(data: Data, num_parts: int, shuffle: bool = True, seed: int = None, sim_metric='cosine') -> List[Data]:
    """
    Method D: 学习式选择的轻量实现（可作为 RL/selector 的替代 baseline）
    思路（实用近似）：
      - 计算每个节点的多种评分：degree-based (低度偏好), pagerank-like (用 reversed degree),
        embedding-centric outlier score (距离全局均值), 与局部 cluster label。
      - 把这些 score 拼成向量，做一次小型 linear scoring（权重随机/可调），再用 Farthest Point Sampling
       （基于 embedding）确保每个 partition 的内部多样性。
      - 这个函数是“可学习 selector”的工程可替代实现：你可以把 scoring 部分换成训练好的小网络。
    复杂度：O(n) ~ O(n log n)
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    _validate_inputs(data, num_parts, sim_metric)
    n = data.num_nodes
    x = data.x

    # degree-based low-degree preference
    edge_index = to_undirected(data.edge_index)
    deg = degree(edge_index[0], num_nodes=n)

    # embedding outlier score (distance to global mean)
    global_mean = x.mean(dim=0, keepdim=True)
    dist_to_mean = torch.norm(x - global_mean, dim=-1)  # larger -> more OOD

    # simple PageRank-ish score approximated by inverse-degree
    invdeg = 1.0 / (deg + 1.0)

    # combine features into node score vector (stack)
    feats = torch.stack([dist_to_mean, invdeg, -deg], dim=1)  # prefer high OOD, high invdeg, low deg
    feats = (feats - feats.mean(dim=0)) / (feats.std(dim=0) + 1e-8)
    # random initial linear weights (or could be trained offline)
    w = torch.tensor([1.0, 0.8, 0.6], dtype=torch.float32)
    node_scores = (feats * w).sum(dim=1)

    # We will form parts by repeatedly picking a seed with high node_score, then do Farthest Point Sampling
    # (FPS) in embedding space to enforce diversity (low internal sim).
    xt = F.normalize(x, dim=-1) if sim_metric == 'cosine' else x
    sizes = [n // num_parts + (1 if i < (n % num_parts) else 0) for i in range(num_parts)]
    used = set()
    parts_idx = [[] for _ in range(num_parts)]

    sorted_nodes = torch.argsort(node_scores, descending=True).tolist()
    seed_candidates = [int(u) for u in sorted_nodes if int(u) not in used]

    for pid in range(num_parts):
        # choose highest scoring unused node as seed
        seed_node = None
        for cand in seed_candidates:
            if cand not in used:
                seed_node = cand
                break
        if seed_node is None:
            # fallback
            for i in range(n):
                if i not in used:
                    seed_node = i
                    break
            if seed_node is None:
                break
        parts_idx[pid].append(seed_node)
        used.add(seed_node)

        # do FPS: iteratively add node farthest in embedding space from current set
        while len(parts_idx[pid]) < sizes[pid]:
            # compute distance of all unused nodes to current set (min distance to any selected)
            cur_mem = torch.tensor(parts_idx[pid], dtype=torch.long)
            unused = [i for i in range(n) if i not in used]
            if not unused:
                break
            # vectorized distances
            if sim_metric == 'cosine':
                # use 1 - cosine as distance
                dists = 1.0 - (xt[unused] @ xt[cur_mem].t()).max(dim=1)[0]
            else:
                dists = torch.cdist(xt[unused], xt[cur_mem]).min(dim=1)[0]
            # incorporate node_scores (bias toward high OOD even if not far)
            # normalize
            ds = dists + 0.1 * (1.0 - (node_scores[unused] - node_scores.min()) / (node_scores.max() - node_scores.min() + 1e-9))
            # pick index of max ds
            best_idx = int(torch.argmax(ds).item())
            chosen = unused[best_idx]
            parts_idx[pid].append(chosen)
            used.add(chosen)

    # if leftover nodes exist, distribute them to parts that currently have highest avg distance (to keep hetero)
    leftover = [i for i in range(n) if i not in used]
    for v in leftover:
        best_pid = None
        best_val = None
        for pid in range(num_parts):
            mem = parts_idx[pid]
            if len(mem) == 0:
                val = 0.0
            else:
                if sim_metric == 'cosine':
                    val = float((xt[v:v+1] @ xt[torch.tensor(mem, dtype=torch.long)].t()).mean().item())
                else:
                    val = float(-torch.cdist(xt[v:v+1], xt[torch.tensor(mem, dtype=torch.long)]).mean().item())
            if best_val is None or val < best_val:
                best_val = val
                best_pid = pid
        parts_idx[best_pid].append(v)

    parts = []
    for idx_list in parts_idx:
        subset = torch.tensor(idx_list, dtype=torch.long)
        sub = _make_subdata_from_indices(data, subset)
        parts.append(sub)
    return parts
def load_pyg_dataset(name: str, root: str = './data'):
    """
    加载常见数据集（Planetoid 系列）。返回 single Data 对象。
    name: 'Cora' | 'CiteSeer' | 'PubMed'
    """
    name = name if isinstance(name, str) else str(name)
    dataset = Planetoid(root=root, name=name)
    return dataset[0]

def load_npz_as_data(npz_path: str):
    """
    如果你有自制的 .npz（例如保存了 edge_index, x, y, train_mask...），
    按照常用格式加载为 Data。请根据你的 npz 内容调整键名。
    """
    import numpy as np
    from torch_geometric.data import Data

    obj = np.load(npz_path, allow_pickle=True)
    # 下面是假定键名；如不同请修改： 'edge_index','x','y','train_mask','val_mask','test_mask'
    edge_index = torch.tensor(obj['edge_index'], dtype=torch.long)
    x = torch.tensor(obj['x'], dtype=torch.float) if 'x' in obj else None
    y = torch.tensor(obj['y'], dtype=torch.long) if 'y' in obj else None

    data = Data(x=x, edge_index=edge_index, y=y)
    for k in ('train_mask', 'val_mask', 'test_mask'):
        if k in obj:
            data[k] = torch.tensor(obj[k], dtype=torch.bool)
    return data


def partition_graph_equal(data: Data, num_parts: int, shuffle: bool = True, seed: int = None, sim_metric='cosine'):
    """
    把一个 Data（大图）拆成 num_parts 个“多样化”子图：
    目标：每个子图内部节点尽量不相似，内部边尽量稀疏（低连接度）。
    思路：
        1. 计算节点相似度矩阵。
        2. 用贪心反聚类策略，把相似节点分配到不同分区。
    参数：
        sim_metric: 'cosine' 或 'euclidean' —— 节点特征相似度度量方式
    输出：
        list[Data]，每个子图结构与原函数相同，额外字段 sub_data.orig_node_idx 保留原索引。
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = data.num_nodes
    if num_parts <= 0:
        raise ValueError("num_parts must be > 0")
    if num_parts > n:
        raise ValueError("num_parts cannot exceed number of nodes")
    if data.x is None:
        raise ValueError("data.x 不能为空（需要节点特征计算相似度）")

    # === Step 1: 节点相似度矩阵 ===
    x = F.normalize(data.x, dim=-1) if sim_metric == 'cosine' else data.x
    sim = torch.matmul(x, x.t())  # [n, n]，cosine similarity
    if sim_metric == 'euclidean':
        dist = torch.cdist(x, x)  # pairwise distance
        sim = -dist  # 负距离视作“相似度低”

    # === Step 2: 多样化划分 ===
    # 初始化：先选一些节点作为每个分区的“种子”，余下节点逐个分配到与其最不相似的分区。
    perm = torch.randperm(n) if shuffle else torch.arange(n)
    seeds = perm[:num_parts]
    clusters = [[] for _ in range(num_parts)]
    for i, s in enumerate(seeds):
        clusters[i].append(s.item())

    remaining = perm[num_parts:]
    for node in remaining:
        node_sim_to_cluster = []
        for c in clusters:
            # 计算该节点对 cluster 内所有节点的平均相似度
            if len(c) == 0:
                node_sim_to_cluster.append(0)
            else:
                mean_sim = sim[node, torch.tensor(c)].mean().item()
                node_sim_to_cluster.append(mean_sim)
        # 选相似度最小的 cluster 放进去（越不相似越好）
        best_cluster = int(torch.argmin(torch.tensor(node_sim_to_cluster)))
        clusters[best_cluster].append(node.item())

    # === Step 3: 构造子图 ===
    parts = []
    for subset_list in clusters:
        subset = torch.tensor(subset_list, dtype=torch.long)
        ei, edge_mask = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=n)
        sub_data = Data()

        if data.x is not None:
            sub_data.x = data.x[subset]
        if hasattr(data, 'y') and data.y is not None:
            sub_data.y = data.y[subset]

        for k in ('train_mask', 'val_mask', 'test_mask'):
            if hasattr(data, k):
                orig_mask = getattr(data, k)
                sub_data[k] = orig_mask[subset].clone()

        sub_data.edge_index = ei
        sub_data.num_nodes = subset.size(0)
        sub_data.orig_node_idx = subset.clone()
        parts.append(sub_data)

    return parts

if __name__ == '__main__':
    # 简单示例（运行脚本会自动把 Cora 划分为 10 个子图并打印每个子图的键）
    data = load_pyg_dataset('Cora')
    subgraphs = partition_graph_equal(data, num_parts=10, shuffle=True, seed=42)
    for i, sg in enumerate(subgraphs):
        keys = list(sg.keys) if hasattr(sg, 'keys') else [k for k in sg.__dict__.keys() if not k.startswith('_')]
        # 更稳妥地列出常见键
        present_keys = [k for k in ['x','edge_index','y','train_mask','val_mask','test_mask','orig_node_idx'] if hasattr(sg, k)]
        print(f"Subgraph {i}: nodes={sg.num_nodes}, edges={sg.edge_index.size(1)}, keys={present_keys}")

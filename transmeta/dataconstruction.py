# edge_dataset_with_svd.py
#这个文件是为了加载数据集，并且进行SVD降维
import os
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.utils import from_scipy_sparse_matrix

def load_prog_data(path):
    data = np.load(path, allow_pickle=True)
    adj = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
                        shape=tuple(data['adj_shape']))
    features = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']),
                             shape=tuple(data['attr_shape']))
    # labels = data['labels']
    # idx_train = data['idx_train']
    # idx_val = data['idx_val']
    # idx_test = data['idx_test']
    return adj, features

def ensure_undirected(adj):
    adj = adj.tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1
    return adj

def load_prog_data(path, make_undirected=True):
    """
    加载图数据，支持多种格式的npz文件
    """
    print(f"尝试加载文件: {path}")
    
    # 方法1: 尝试加载包含完整图数据的格式 (adj_data, attr_data等)
    try:
        data = np.load(path, allow_pickle=True)
        print(f"文件包含的键: {list(data.keys())}")
        
        if all(k in data for k in ['adj_data','adj_indices','adj_indptr','adj_shape']) and \
           all(k in data for k in ['attr_data','attr_indices','attr_indptr','attr_shape']):
            print("检测到完整图数据格式")
            adj = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
                                shape=tuple(data['adj_shape']))
            features = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']),
                                     shape=tuple(data['attr_shape']))
            if make_undirected:
                adj = ensure_undirected(adj)
            return adj, features
    except Exception as e:
        print(f"方法1失败: {e}")
    
    # 方法2: 尝试加载标准scipy sparse格式
    try:
        print("尝试标准scipy sparse格式")
        # 不使用allow_pickle，因为scipy.sparse格式不需要pickle
        adj = sp.load_npz(path).tocsr()
        print(f"成功加载邻接矩阵，形状: {adj.shape}")
        if make_undirected:
            adj = ensure_undirected(adj)
        return adj, None
    except Exception as e:
        print(f"方法2失败: {e}")
    
    # 方法3: 尝试手动构建sparse矩阵
    try:
        print("尝试手动构建sparse矩阵")
        data = np.load(path, allow_pickle=False)  # 不使用pickle
        print(f"文件包含的键: {list(data.keys())}")
        
        if all(k in data for k in ['data', 'indices', 'indptr', 'shape']):
            adj = sp.csr_matrix((data['data'], data['indices'], data['indptr']), 
                               shape=tuple(data['shape']))
            print(f"成功构建邻接矩阵，形状: {adj.shape}")
            if make_undirected:
                adj = ensure_undirected(adj)
            return adj, None
    except Exception as e:
        print(f"方法3失败: {e}")
    
    raise RuntimeError(f"无法加载文件 {path}，尝试了所有可能的格式")

def reduce_features_svd(features, n_components=100, random_state=42, do_scale=True):
    """
    features: sparse or dense matrix (n_nodes x feat_dim)
    returns: ndarray (n_nodes x n_components)
    """
    # TruncatedSVD works with sparse input
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduced = svd.fit_transform(features)  # shape (n_nodes, n_components)
    if do_scale:
        scaler = StandardScaler()
        X_reduced = scaler.fit_transform(X_reduced)
    return X_reduced, svd  # return svd in case user wants to reuse

def construct_edge_diff_dataset(adj_clean, adj_attack, X_reduced, balance=True, balance_strategy='min', random_state=42):
    """
    构建边翻转数据集
    
    Args:
        adj_clean: 干净图的邻接矩阵
        adj_attack: 攻击后图的邻接矩阵  
        X_reduced: 降维后的节点特征
        balance: 是否平衡正负样本
        balance_strategy: 平衡策略
            - 'min': 使用较少类别的样本数量（默认，确保1:1比例）
            - 'pos': 使用正样本数量（对负样本下采样）
            - 'neg': 使用负样本数量（对正样本下采样）
        random_state: 随机种子
        
    Returns:
      - edge_pairs: np.array shape (N_samples, 2) -- node indices
      - X_pairs: np.array shape (N_samples, 2 * reduced_dim) -- concatenated features
      - labels: np.array shape (N_samples,)
    """
    rng = np.random.default_rng(random_state)
    adj_clean = adj_clean.tocoo()
    adj_attack = adj_attack.tocoo()

    # build undirected edge sets with i<j to avoid duplicates (ignore self-loops)
    clean_edges = set()
    for u, v in zip(adj_clean.row, adj_clean.col):
        if u == v: 
            continue
        a, b = (u, v) if u < v else (v, u)
        clean_edges.add((a, b))
    attack_edges = set()
    for u, v in zip(adj_attack.row, adj_attack.col):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        attack_edges.add((a, b))

    # flipped = symmetric difference -> edges that changed (added or removed)
    flipped_edges = clean_edges.symmetric_difference(attack_edges)
    pos_edges = list(flipped_edges)  # label=1

    # unchanged edges = intersection -> exist in both graphs (label=0)
    unchanged_edges = list(clean_edges.intersection(attack_edges))

    # If there are too few negatives, you may also sample from non-edges (not present in either)
    if len(unchanged_edges) == 0:
        raise ValueError("No unchanged edges found (intersection empty). Check inputs.")

    # balance positive/negative
    if balance:
        n_pos = len(pos_edges)
        n_neg = len(unchanged_edges)
        
        if n_pos == 0:
            raise ValueError("No flipped edges (pos==0). Check adj_attack vs adj_clean.")
        if n_neg == 0:
            raise ValueError("No unchanged edges (neg==0). Check inputs.")
        
        print(f"原始样本数量: 正样本 {n_pos}, 负样本 {n_neg}")
        
        # 根据平衡策略选择目标样本数
        if balance_strategy == 'min':
            target_samples = min(n_pos, n_neg)
        elif balance_strategy == 'pos':
            target_samples = n_pos
        elif balance_strategy == 'neg':
            target_samples = n_neg
        else:
            raise ValueError(f"Unknown balance_strategy: {balance_strategy}. "
                           "Choose from 'min', 'pos', 'neg'")
        
        # 对正样本进行采样
        rng.shuffle(pos_edges)
        pos_edges = pos_edges[:target_samples]
        
        # 对负样本进行采样
        rng.shuffle(unchanged_edges)
        neg_edges = unchanged_edges[:target_samples]
        
        print(f"平衡采样结果 (策略: {balance_strategy}): 正样本 {len(pos_edges)}, 负样本 {len(neg_edges)}")
    else:
        neg_edges = unchanged_edges
        print(f"不平衡模式: 正样本 {len(pos_edges)}, 负样本 {len(neg_edges)}")

    all_edges = pos_edges + neg_edges
    labels = np.array([1]*len(pos_edges) + [0]*len(neg_edges), dtype=np.int64)

    # build feature pairs (concatenate)
    # ensure X_reduced is ndarray
    if sp.issparse(X_reduced):
        X_arr = X_reduced.toarray()
    else:
        X_arr = np.asarray(X_reduced)

    X_pairs = np.stack([np.concatenate([X_arr[i], X_arr[j]]) for (i, j) in all_edges], axis=0)
    edge_pairs = np.array(all_edges, dtype=np.int64)

    # shuffle dataset
    perm = rng.permutation(len(labels))
    X_pairs = X_pairs[perm]
    labels = labels[perm]
    edge_pairs = edge_pairs[perm]

    return edge_pairs, X_pairs, labels

if __name__ == "__main__":
    # 路径按需改
    clean_path = "DeepRobust/examples/graph/tmp/cora.npz"
    attacked_path = "DeepRobust/examples/graph/tmp/cora_modified_095.npz"

    adj_clean, features, labels, idx_train, idx_val, idx_test = load_prog_data(clean_path)
    adj_attack, _, _, _, _, _ = load_prog_data(attacked_path)

    # 降维到100
    X_reduced, svd_model = reduce_features_svd(features, n_components=100, random_state=42, do_scale=True)
    print("SVD done, new shape:", X_reduced.shape)

    # 构造数据集
    edge_pairs, X_pairs, y = construct_edge_diff_dataset(adj_clean, adj_attack, X_reduced, balance=True, random_state=42)
    print("samples:", X_pairs.shape, "labels:", y.shape)

    # convert to torch tensors (for PyG GNN variant we also need edge_index)
    edge_index, _ = from_scipy_sparse_matrix(adj_clean)  # use clean graph for message passing
    x_tensor = torch.tensor(X_reduced, dtype=torch.float)  # node features after SVD
    edge_index = edge_index.long()
    # edge_pairs used for training classifier (pairs of node indices)
    edge_pairs_tensor = torch.tensor(edge_pairs, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.float)

    # quick save if you want to reuse
    np.savez("cora_pairs_svd100.npz", edge_pairs=edge_pairs, X_pairs=X_pairs, labels=y)
    print("Saved cora_pairs_svd100.npz")

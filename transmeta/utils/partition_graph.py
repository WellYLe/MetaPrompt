# partition_graph.py
import math
import numpy as np
import random
from sklearn.cluster import KMeans
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph

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

def partition_graph_equal(data: Data, num_parts: int, shuffle: bool = True, seed: int = None):
    """
    把一个 Data（大图）近似等大小拆成 num_parts 个子图（list of Data）。
    算法：对节点做随机/顺序分段，取诱导子图（induced subgraph）。
    注意：这是“节点等分”的方法；每个子图的连通性未作保证（可能断开）。
    返回：list_of_subgraphs (长度 == num_parts)
    每个子图包含额外键： 'orig_node_idx'（在原图中的节点索引）
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = data.num_nodes
    if num_parts <= 0:
        raise ValueError("num_parts must be > 0")
    if num_parts > n:
        raise ValueError("num_parts cannot exceed number of nodes")

    # 生成节点划分的索引集合（保证每一份大小尽量相等）
    if shuffle:
        perm = torch.randperm(n)
    else:
        perm = torch.arange(n)

    base = n // num_parts
    remainder = n % num_parts
    sizes = [base + (1 if i < remainder else 0) for i in range(num_parts)]

    parts = []
    offset = 0
    for part_size in sizes:
        subset = perm[offset: offset + part_size]
        offset += part_size

        # induced subgraph: 返回 relabeled edge_index 以及 edge mask 等
        sub_edge_index, sub_edge_attr = None, None
        # subgraph API returns (edge_index, edge_mask) when given edge_index and node_idx
        ei, edge_mask = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=n)
        sub_data = Data()

        # map node features and labels if exist
        if data.x is not None:
            sub_data.x = data.x[subset]
        if hasattr(data, 'y') and data.y is not None:
            sub_data.y = data.y[subset]

        # masks: 如果原图有 train/val/test_mask，则映射到子图
        for k in ('train_mask', 'val_mask', 'test_mask'):
            if hasattr(data, k):
                orig_mask = getattr(data, k)
                # subset 是原图索引，提取并转换为长度 = sub_num_nodes 的 bool tensor
                sub_mask = orig_mask[subset].clone()
                sub_data[k] = sub_mask

        # edge_index 和统计
        sub_data.edge_index = ei
        sub_data.num_nodes = subset.size(0)
        # 保存原图节点索引（便于结果回写或定位）
        sub_data.orig_node_idx = subset.clone()  # 这是原图的 node indices

        parts.append(sub_data)

    return parts

def partition_graph_equal2(data: Data, num_parts: int, shuffle: bool = True, seed: int = None):
    """
    把一个 Data（大图）近似等大小拆成 num_parts 个子图（list of Data）。
    算法：基于节点嵌入聚类划分，取诱导子图（induced subgraph）。
    注意：这是"节点近似等分"的方法；每个子图的连通性未作保证（可能断开）。
    返回：list_of_subgraphs (长度 == num_parts)
    每个子图包含额外键： 'orig_node_idx'（在原图中的节点索引）
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    n = data.num_nodes
    if num_parts <= 0:
        raise ValueError("num_parts must be > 0")
    if num_parts > n:
        raise ValueError("num_parts cannot exceed number of nodes")

    # 获取节点特征作为嵌入
    if hasattr(data, 'x') and data.x is not None:
        node_embeddings = data.x.detach().cpu().numpy()
    else:
        # 如果没有节点特征，使用节点度作为简单嵌入
        node_embeddings = _get_degree_embeddings(data)
    
    # 使用K-means聚类将节点划分为num_parts个簇
    kmeans = KMeans(n_clusters=num_parts, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(node_embeddings)
    
    # 计算每个簇的大小，确保近似相等
    cluster_sizes = [np.sum(cluster_labels == i) for i in range(num_parts)]
    print(f"Cluster sizes: {cluster_sizes}")

    parts = []
    for cluster_id in range(num_parts):
        # 获取属于当前簇的节点
        cluster_mask = cluster_labels == cluster_id
        subset = torch.tensor(np.where(cluster_mask)[0], dtype=torch.long)

        # induced subgraph: 返回 relabeled edge_index 以及 edge mask 等
        sub_edge_index, sub_edge_attr = None, None
        # subgraph API returns (edge_index, edge_mask) when given edge_index and node_idx
        ei, edge_mask = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=n)
        sub_data = Data()

        # map node features and labels if exist
        if data.x is not None:
            sub_data.x = data.x[subset]
        if hasattr(data, 'y') and data.y is not None:
            sub_data.y = data.y[subset]

        # masks: 如果原图有 train/val/test_mask，则映射到子图
        for k in ('train_mask', 'val_mask', 'test_mask'):
            if hasattr(data, k):
                orig_mask = getattr(data, k)
                # subset 是原图索引，提取并转换为长度 = sub_num_nodes 的 bool tensor
                sub_mask = orig_mask[subset].clone()
                sub_data[k] = sub_mask

        # edge_index 和统计
        sub_data.edge_index = ei
        sub_data.num_nodes = subset.size(0)
        # 保存原图节点索引（便于结果回写或定位）
        sub_data.orig_node_idx = subset.clone()  # 这是原图的 node indices

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

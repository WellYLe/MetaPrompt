# edge_dataset_with_svd.py
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
    labels = data['labels']
    idx_train = data['idx_train']
    idx_val = data['idx_val']
    idx_test = data['idx_test']
    return adj, features, labels, idx_train, idx_val, idx_test

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

def construct_edge_diff_dataset(adj_clean, adj_attack, X_reduced, balance=True, random_state=42):
    """
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
        if n_pos == 0:
            raise ValueError("No flipped edges (pos==0). Check adj_attack vs adj_clean.")
        # shuffle unchanged and pick same amount
        rng.shuffle(unchanged_edges)
        neg_edges = unchanged_edges[:n_pos]
    else:
        neg_edges = unchanged_edges

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

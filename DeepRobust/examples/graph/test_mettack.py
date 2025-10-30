import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import json
"""
确保导入本地 DeepRobust 软件包（而不是站点软件包）。
"""
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
#from deeprobust.graph.global_attack.mettack import MetaEva
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self',
        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train', 'E-Meta-Self'], help='model variant')

args = parser.parse_args()

# Respect --no-cuda flag; default to CPU for stability on Meta attacks
use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device("cuda:0" if use_cuda else "cpu")
#super(MetaEva, self).__init__(...)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)
    
# if 'E' in args.model:
#     model = MetaEva(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device)
else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)

def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()
def to_dense_numpy(adj):
    """支持 scipy.sparse 或 torch tensor 或 numpy array"""
    if sp.issparse(adj):
        return adj.toarray()
    if isinstance(adj, torch.Tensor):
        try:
            return adj.to_dense().cpu().numpy()
        except Exception:
            return adj.cpu().numpy()
    return np.array(adj)

def compute_common_neighbors(adj_dense):
    # adj_dense: numpy 0/1 matrix
    return adj_dense.dot(adj_dense)  # N x N integer matrix (counts)

def build_edge_change_dataset(adj_before, adj_after, features,
                              max_neg_ratio=None, keep_all=False,
                              add_structural_features=True):
    """
    返回 samples_dict:
      'X_pairs' -> (M, 2*F + k) numpy array (concatenated features + structural)
      'pairs' -> (M,2) node indices
      'y' -> (M,) labels 0/1
    参数:
      max_neg_ratio: 若不为None，最多负样本数量 = pos_count * max_neg_ratio
      keep_all: True 则不做负采样（危险，可能很大）
    """
    A0 = to_dense_numpy(adj_before).astype(np.int8)
    A1 = to_dense_numpy(adj_after).astype(np.int8)
    N = A0.shape[0]
    X = np.array(features)  # (N, F)

    diff = (A1 != A0).astype(np.int8)
    # 只看上三角（无向图）
    iu = np.triu_indices(N, k=1)
    all_pairs = np.vstack(iu).T  # shape (N*(N-1)/2, 2)
    diff_upper = diff[iu]
    pos_mask = diff_upper == 1
    neg_mask = diff_upper == 0

    pos_pairs = all_pairs[pos_mask]
    neg_pairs = all_pairs[neg_mask]

    P_total = len(pos_pairs)
    N_total = len(neg_pairs)
    print(f"[Dataset构造] 总节点数 N={N}")
    print(f"[Dataset构造] 所有可能的上三角对数 (i<j) = {len(all_pairs)}")
    print(f"[Dataset构造] 原始正样本(被翻转)数量 = {P_total}")
    print(f"[Dataset构造] 原始负样本(未翻转)数量 = {N_total}")

    # 负采样
    if not keep_all and max_neg_ratio is not None:
        neg_keep = min(N_total, int(P_total * max_neg_ratio))
        rng = np.random.default_rng(42)
        neg_idx = rng.choice(N_total, size=neg_keep, replace=False) if neg_keep > 0 else np.array([], dtype=int)
        neg_pairs = neg_pairs[neg_idx]
        print(f"[Dataset构造] 负采样后负样本数量 = {len(neg_pairs)} (max_neg_ratio={max_neg_ratio})")
    else:
        print(f"[Dataset构造] 保留全部负样本 (keep_all=True)")

    # 合并样本
    pairs = np.vstack([pos_pairs, neg_pairs])
    labels = np.hstack([np.ones(len(pos_pairs), dtype=np.int64),
                        np.zeros(len(neg_pairs), dtype=np.int64)])

    # 可选结构特征
    struct_feats = None
    if add_structural_features:
        deg = A0.sum(axis=1).astype(np.float32)
        cn = compute_common_neighbors(A0)
        struct_feats = []
        for i, j in pairs:
            edge_before = int(A0[i, j])
            struct_feats.append([edge_before, float(deg[i]), float(deg[j]), float(cn[i, j])])
        struct_feats = np.array(struct_feats, dtype=np.float32)

    # 节点对拼接特征
    X_pairs = np.concatenate([X[pairs[:,0]], X[pairs[:,1]]], axis=1)  # (M, 2F)
    if struct_feats is not None:
        X_pairs = np.concatenate([X_pairs, struct_feats], axis=1)

    # 打印最终统计
    total_samples = len(labels)
    pos_count = int(labels.sum())
    neg_count = total_samples - pos_count
    pos_ratio = pos_count / total_samples if total_samples > 0 else 0.0
    print(f"[Dataset构造] 最终样本总数 = {total_samples}")
    print(f"[Dataset构造] 正样本 = {pos_count}, 负样本 = {neg_count}, 正样本占比 = {pos_ratio:.6f}")

    stats = {
        'N_nodes': int(N),
        'all_candidate_pairs': int(len(all_pairs)),
        'pos_original': int(P_total),
        'neg_original': int(N_total),
        'pos_final': int(pos_count),
        'neg_final': int(neg_count),
        'pos_ratio': float(pos_ratio),
        'max_neg_ratio': max_neg_ratio,
        'keep_all': bool(keep_all),
        'struct_feats': bool(add_structural_features)
    }

    return {
        'X_pairs': X_pairs.astype(np.float32),
        'pairs': pairs.astype(np.int32),
        'y': labels.astype(np.int64),
        'stats': stats
    }

class EdgeFlipDataset(Dataset):
    def __init__(self, np_data):
        self.X = torch.from_numpy(np_data['X_pairs']).float()
        self.y = torch.from_numpy(np_data['y']).long()
        self.pairs = torch.from_numpy(np_data['pairs']).long()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.pairs[idx]

# 示例：把构造过程放进你的 main()，在 attack() 完成后调用
# adj_before: 原始 adj（scipy稀疏或 numpy）
# model.modified_adj: 攻击后邻接（torch tensor 或 scipy）
# features: numpy 或 torch (N,F)

def export_and_report_dataset(adj_before, modified_adj, features, args,
                              max_neg_ratio=3, keep_all=False, rng_seed=0):
    data_dict = build_edge_change_dataset(adj_before, modified_adj, features,
                                         max_neg_ratio=1, keep_all=False,
                                         add_structural_features=True)
    # 保存 npz
    save_path = f"{args.dataset}_edgeflip_dataset_ptbrate{int(args.ptb_rate*100):03d}.npz"
    np.savez_compressed(save_path,
                        X_pairs=data_dict['X_pairs'],
                        pairs=data_dict['pairs'],
                        y=data_dict['y'])
    # 保存统计 JSON（便于记录）
    meta_path = save_path.replace('.npz', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(data_dict['stats'], f, indent=2)
    print(f"[Dataset构造] 已保存数据到 {save_path}，统计信息保存到 {meta_path}")

    ds = EdgeFlipDataset(data_dict)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)
    return ds, loader, data_dict['stats']

def main():
    # MetaEvasion 的接口与 Metattack 不同：需要传入目标节点和整数 n_perturbations
    # if isinstance(model, MetaEva):
    #     target_nodes = idx_unlabeled
    #     model.attack(features, adj, labels, target_nodes, n_perturbations=perturbations, targeted=False)
    # else:
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    print('=== testing GCN on original(clean) graph ===')
    test(adj)
    modified_adj = model.modified_adj
    print('=== testing GCN on modified graph ===')
    test(modified_adj)
    # 计算被修改的边
    adj_before = adj.to_dense().cpu().numpy() if torch.is_tensor(adj) else adj.A
    adj_after = model.modified_adj.to_dense().cpu().numpy() if torch.is_tensor(model.modified_adj) else model.modified_adj.A

    diff = adj_after - adj_before
    # 只取上三角避免重复（无向图）
    changed_edges = np.transpose(np.nonzero(np.triu(diff != 0, 1)))

    num_changes = changed_edges.shape[0]
    print(f"=== Total {num_changes} edges have been modified ===")

    # 输出每条被修改的边及其变化方向
    for i, j in changed_edges:
        if diff[i, j] == 1:
            print(f"Added edge: ({i}, {j})")
        elif diff[i, j] == -1:
            print(f"Removed edge: ({i}, {j})")
    ds, loader, stats = export_and_report_dataset(adj, model.modified_adj, features, args, max_neg_ratio=3)
    
    
    

    #保存修改后的结构到当前目录（含数据集与比例信息）
    save_name = f'{args.dataset}_mod_adj_{int(args.ptb_rate*100):03d}'
    model.save_adj(root='./', name=save_name)
    model.save_features(root='./', name=f'{args.dataset}_mod_features')

if __name__ == '__main__':
    main()


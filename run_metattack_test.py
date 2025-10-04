# run_metattack_test.py
import os
import random
import numpy as np
import torch
import pandas as pd
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN  # surrogate & victim
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.utils import preprocess, normalize_adj_tensor
import matplotlib.pyplot as plt
import scipy.sparse as sp

# ----------------- reproducibility -----------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- config -----------------
dataset_name = 'cora'
hidden = 16
dropout = 0.0
lr_surrogate = 0.01
patience = 30
n_perturbations = 20   # attack budget
lambda_ = 0.5          # for Metattack
train_iters_inner = 50 # inner-loop steps (Metattack default may be 100, this is for speed)
attack_lr = 0.1
momentum = 0.9

# ----------------- load dataset -----------------
data = Dataset(root='/tmp/', name=dataset_name)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

# optionally preprocess features/adj into tensors for your Metattack (depends on implementation)
# Many deeprobust examples call parse / preprocess; preserve original adj/features for attack code
# Convert features to dense if sparse
if sp.issparse(features):
    features = features.todense()

# ----------------- train surrogate model (GCN) -----------------
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=hidden,
                dropout=dropout, with_relu=False, with_bias=False, device=device).to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=patience)

# Evaluate surrogate on original graph
print("Evaluate surrogate on original graph")
adj_tensor = adj  # keep original adjacency in whichever format your evaluation expects
acc_before = surrogate.test(features, adj, labels, idx_test)
print(f'Original accuracy (test): {acc_before:.4f}')

# ----------------- setup Metattack -----------------
# instantiate Metattack with the trained surrogate as "model"
attacker = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                     attack_structure=True, attack_features=False,
                     undirected=True, device=device, with_bias=False,
                     lambda_=lambda_, train_iters=train_iters_inner, lr=attack_lr, momentum=momentum).to(device)

# optional: if Metattack expects preprocessed torch tensors, ensure to pass correctly
# Run attack: produce n_perturbations
attacker.attack(ori_features=features, ori_adj=adj, labels=labels,
                idx_train=idx_train, idx_unlabeled=idx_unlabeled,
                n_perturbations=n_perturbations, ll_constraint=True, ll_cutoff=0.004)

# ----------------- get modified graph -----------------
modified_adj = attacker.modified_adj  # this should be a torch tensor or numpy array depending on implementation
# If torch tensor, convert to numpy/sparse etc for downstream
if isinstance(modified_adj, torch.Tensor):
    modified_adj = modified_adj.detach().cpu().numpy()

# binarize to get final adjacency (for undirected)
# if undirected, ensure symmetric and remove self-loops
A_mod = (modified_adj > 0.5).astype(int)
A_mod = np.triu(A_mod, 1)
A_mod = A_mod + A_mod.T

# ----------------- evaluate surrogate / victim on modified graph ---------------
# We can either evaluate the same surrogate (retrained on modified graph) or victim model frozen.
# Typical Metattack evaluation: retrain victim GCN on the *poisoned* graph and measure test accuracy drop.

victim = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=hidden,
             dropout=dropout, with_relu=False, with_bias=False, device=device).to(device)

# retrain victim on poisoned graph (simulate model trained on poisoned data)
victim.fit(features, A_mod, labels, idx_train, idx_val, patience=patience)
acc_after = victim.test(features, A_mod, labels, idx_test)
print(f'Post-attack accuracy (test): {acc_after:.4f}')
print(f'Accuracy drop: {acc_before - acc_after:.4f}')

# success rate (ASR) - common definition: fraction of initially-correct nodes that become misclassified after attack
# compute node-level predictions before & after for idx_test (or idx_unlabeled)
pred_before = surrogate.predict(features, adj) if hasattr(surrogate, 'predict') else surrogate.output.argmax(1).cpu().numpy()
# For "victim" we used retrain; predictions after:
pred_after = victim.predict(features, A_mod) if hasattr(victim, 'predict') else victim.output.argmax(1).cpu().numpy()

# choose target set to evaluate transfer: idx_unlabeled/idx_test
target_idx = idx_test
# compute ASR
orig_correct = (pred_before[target_idx] == labels[target_idx]).astype(int)
now_incorrect = (pred_after[target_idx] != labels[target_idx]).astype(int)
# success on nodes that were originally correct
if orig_correct.sum() > 0:
    ASR = (orig_correct * now_incorrect).sum() / orig_correct.sum()
else:
    ASR = 0.0
print(f'ASR (fraction of originally-correct test nodes made wrong): {ASR:.4f}')

# ----------------- stealthiness / graph stats -----------------
def degree_stats(A):
    deg = np.sum(A, axis=1)
    return deg.mean(), deg.std()

mean_before, std_before = degree_stats(adj.todense() if sp.issparse(adj) else adj)
mean_after, std_after = degree_stats(A_mod)
print('Degree mean/std before: {:.3f} / {:.3f}'.format(mean_before, std_before))
print('Degree mean/std after : {:.3f} / {:.3f}'.format(mean_after, std_after))

# save history if attacker stored it (we recommended adding history earlier)
if hasattr(attacker, 'history'):
    pd.DataFrame(attacker.history).to_csv('attack_history.csv', index=False)
    print('Saved attack history to attack_history.csv')

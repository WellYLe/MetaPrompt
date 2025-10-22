import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import scipy.sparse as sp
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from deeprobust.graph.data import Dataset

d = Dataset(root='/tmp/', name='cora', setting='prognn')
features = d.features
labels = d.labels
idx_train = d.idx_train
idx_val = d.idx_val
idx_test = d.idx_test

# 读取原文件
data = np.load('cora.npz', allow_pickle=True)

new_adj = sp.load_npz('../../../cora_mod_adj_00001.npz').tocsr()
new_adj = new_adj.tocsr()

features = features.tocsr()
np.savez('cora_modified_00001.npz',
         adj_data=new_adj.data,
         adj_indices=new_adj.indices,
         adj_indptr=new_adj.indptr,
         adj_shape=new_adj.shape,
         attr_data=features.data,
         attr_indices=features.indices,
         attr_indptr=features.indptr,
         attr_shape=features.shape,
         labels=labels,
         idx_train=idx_train,
         idx_val=idx_val,
         idx_test=idx_test)
print('保存完成: cora_modified.npz')
print('111')

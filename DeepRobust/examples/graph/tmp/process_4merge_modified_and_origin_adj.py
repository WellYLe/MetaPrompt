import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import scipy.sparse as sp
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from deeprobust.graph.data import Dataset

d = Dataset(root='/tmp/', name='pubmed', setting='prognn')
features = d.features
labels = d.labels
idx_train = d.idx_train
idx_val = d.idx_val
idx_test = d.idx_test

# 加载修改后的邻接矩阵
# 加载修改后的邻接矩阵
new_adj = sp.load_npz('pubmed_mod_adj_005.npz').tocsr()

# 确保 features 是稀疏矩阵格式

features = features.tocsr()
np.savez('pubmed_modified_005.npz',
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
print('保存完成: pubmed_modified_005.npz')
print('111')

import numpy as np
import scipy.sparse as sp
import os
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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

# 拿到原特征和标签等
features = data['features'].item() if isinstance(data['features'], np.ndarray) else data['features']
labels = data['labels']
idx_train = data['idx_train']
idx_val = data['idx_val']
idx_test = data['idx_test']

# # 这里是你新的邻接矩阵 (numpy 或 scipy.sparse 格式都可以)
# new_adj = sp.csr_matrix(new_adj_dense)   # 建议保持 csr 格式以节约空间
new_adj = sp.load_npz('./mod_adj.npz')
new_adj = new_adj.tocsr()
new_features = sp.load_npz('./mod_features.npz')
new_features = new_features.tocsr()
# # 重新打包保存
# np.savez('cora_modified.npz',
#          adj=new_adj,
#          features=features,
#          labels=labels,
#          idx_train=idx_train,
#          idx_val=idx_val,
#          idx_test=idx_test)
np.savez('cora_modified.npz',
         adj=new_adj,
         features=features,
         labels=labels,
         idx_train=idx_train,
         idx_val=idx_val,
         idx_test=idx_test)
print('保存完成: cora_modified.npz')

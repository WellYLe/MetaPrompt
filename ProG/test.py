import numpy as np

data1 = np.load("C:/Users/11326/Desktop/MetaPrompt/DeepRobust/mod_adj-005.npz")  # 你的文件路径
data2 = np.load("C:/Users/11326/Desktop/MetaPrompt/DeepRobust/examples/graph/tmp/cora.npz") 
data3 = np.load("C:/Users/11326/Desktop/MetaPrompt/DeepRobust/examples/graph/tmp/cora_modified_005.npz") 
print(data1.files)  # 打印所有键名
print(data2.files)  # 打印所有键名
print(data3.files)  # 打印所有键名
#cora 和Cora扰动后，好像几个idx不同了，之前是idx_to_node和idx_to_class，现在是idx_train, idx_val, idx_test
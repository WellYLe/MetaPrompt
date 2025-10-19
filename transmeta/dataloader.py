import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch_geometric.datasets import Planetoid
import torch

import numpy as np
from sklearn.model_selection import train_test_split

# 加载Cora数据集
dataset = Planetoid(root='./data/Planetoid', name='Cora')
data = dataset[0]

def generate_edge_samples(data, neg_ratio=1.0):
    # 获取现有边（正样本）
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # 创建边的字典，用于快速查找
    edge_dict = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edge_dict[(src, dst)] = 1
        edge_dict[(dst, src)] = 1  # 无向图需要双向添加
    
    # 收集正样本（现有边）
    pos_edges = []
    for src, dst in edge_dict.keys():
        if src < dst:  # 避免重复边
            pos_edges.append((src, dst))
    
    # 生成负样本（不存在的边）
    neg_edges = []
    num_neg_samples = int(len(pos_edges) * neg_ratio)
    while len(neg_edges) < num_neg_samples:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in edge_dict and (dst, src) not in edge_dict:
            neg_edges.append((src, dst))
            edge_dict[(src, dst)] = 0
            edge_dict[(dst, src)] = 0
    
    # 合并正负样本
    all_edges = pos_edges + neg_edges
    all_labels = [1] * len(pos_edges) + [0] * len(neg_edges)
    
    # 转换为张量
    edge_samples = torch.tensor(all_edges, dtype=torch.long).t()
    edge_labels = torch.tensor(all_labels, dtype=torch.long)
    
    return edge_samples, edge_labels

def split_edge_data(edge_samples, edge_labels, shot_num=5, num_classes=2):
    # 按类别分组
    pos_indices = (edge_labels == 1).nonzero(as_tuple=True)[0]
    neg_indices = (edge_labels == 0).nonzero(as_tuple=True)[0]
    
    # 随机打乱索引，保证每个task是随机的
    pos_indices = pos_indices[torch.randperm(pos_indices.size(0))]
    neg_indices = neg_indices[torch.randperm(neg_indices.size(0))]
    
    # 每个类别选择shot_num个样本作为训练集
    train_pos = pos_indices[:shot_num]
    train_neg = neg_indices[:shot_num]
    
    # 剩余样本作为测试集
    test_pos = pos_indices[shot_num:]
    test_neg = neg_indices[shot_num:]
    
    test_neg = neg_indices[shot_num:]
    min_len = min(test_pos.size(0), test_neg.size(0))
    test_pos = test_pos[:min_len]
    test_neg = test_neg[:min_len]
    
    # 合并训练和测试索引
    train_idx = torch.cat([train_pos, train_neg])
    test_idx = torch.cat([test_pos, test_neg])
    
    return train_idx, test_idx

def save_edge_data(edge_samples, edge_labels, train_idx, test_idx, dataset_name='Cora', shot_num=5, task_id=1):
    # 创建目录
    save_dir = f"./Experiment/sample_data/Link/{dataset_name}/{shot_num}_shot/{task_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取训练和测试数据
    train_labels = edge_labels[train_idx]
    test_labels = edge_labels[test_idx]
    
    # 保存数据
    torch.save(train_idx, f"{save_dir}/train_idx.pt")
    torch.save(train_labels, f"{save_dir}/train_labels.pt")
    torch.save(test_idx, f"{save_dir}/test_idx.pt")
    torch.save(test_labels, f"{save_dir}/test_labels.pt")
    
    # 额外保存边的样本，以便后续使用
    torch.save(edge_samples, f"{save_dir}/edge_samples.pt")
    torch.save(edge_labels, f"{save_dir}/edge_labels.pt")
    
    print(f"Saved edge data to {save_dir}")
    
def generate_edge_tasks(data, dataset_name='Cora', shot_nums=[1, 5, 10], task_num=5):
    edge_samples, edge_labels = generate_edge_samples(data, neg_ratio=1.0)
    
    for shot_num in shot_nums:
        for task_id in range(1, task_num + 1):
            # 为每个任务重新随机划分数据
            train_idx, test_idx = split_edge_data(edge_samples, edge_labels, shot_num)
            save_edge_data(edge_samples, edge_labels, train_idx, test_idx, 
                          dataset_name, shot_num, task_id)

# 执行生成
#generate_edge_tasks(data)
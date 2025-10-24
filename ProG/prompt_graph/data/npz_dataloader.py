import torch
import numpy as np
import pickle
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, k_hop_subgraph
from scipy.sparse import csr_matrix
from copy import deepcopy
from Dataset import GraphDataset


class NPZDataLoader:
    """
    从NPZ文件加载图数据并生成兼容的数据加载器
    功能类似于node_task.py中的train_loader，但支持从npz文件加载数据
    """
    
    def __init__(self, dataset_name, npz_file_path, device='cpu', batch_size=32, 
                 smallest_size=10, largest_size=300):
        """
        初始化NPZ数据加载器
        
        Args:
            dataset_name: 数据集名称
            npz_file_path: npz文件路径
            device: 设备类型 ('cpu' 或 'cuda')
            batch_size: 批处理大小
            smallest_size: 子图最小节点数
            largest_size: 子图最大节点数
        """
        self.dataset_name = dataset_name
        self.npz_file_path = npz_file_path
        self.device = device
        self.batch_size = batch_size
        self.smallest_size = smallest_size
        self.largest_size = largest_size
        
        # 加载数据
        self.data = self._load_npz_data()
        self.graphs_list = None
        
    def _load_npz_data(self):
        """
        从NPZ文件加载图数据并转换为PyTorch Geometric格式
        
        Returns:
            Data: PyTorch Geometric数据对象
        """
        # 加载npz文件
        npz_data = np.load(self.npz_file_path)
        
        # 重构稀疏邻接矩阵
        adj_data = npz_data['adj_data']
        adj_indices = npz_data['adj_indices']
        adj_indptr = npz_data['adj_indptr']
        adj_shape = npz_data['adj_shape']
        
        # 创建CSR格式的稀疏矩阵
        adj_matrix = csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)
        
        # 转换为COO格式获取边索引
        adj_coo = adj_matrix.tocoo()
        edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
        
        # 重构节点特征矩阵
        attr_data = npz_data['attr_data']
        attr_indices = npz_data['attr_indices']
        attr_indptr = npz_data['attr_indptr']
        attr_shape = npz_data['attr_shape']
        
        # 创建特征矩阵
        attr_matrix = csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)
        x = torch.tensor(attr_matrix.toarray(), dtype=torch.float)
        
        # 加载标签
        labels = torch.tensor(npz_data['labels'], dtype=torch.long)
        
        # 加载训练/验证/测试索引
        idx_train = torch.tensor(npz_data['idx_train'], dtype=torch.long)
        idx_val = torch.tensor(npz_data['idx_val'], dtype=torch.long)
        idx_test = torch.tensor(npz_data['idx_test'], dtype=torch.long)
        
        # 创建PyTorch Geometric数据对象
        data = Data(x=x, edge_index=edge_index, y=labels)
        data.idx_train = idx_train
        data.idx_val = idx_val
        data.idx_test = idx_test
        
        return data.to(self.device)
    
    def _create_induced_graphs(self):
        """
        创建induced graphs，类似于induced_graph.py中的split_induced_graphs方法
        
        Returns:
            list: 包含所有induced graphs的列表
        """
        induced_graph_list = []
        
        for index in range(self.data.x.size(0)):
            current_label = self.data.y[index].item()
            
            # 使用k-hop子图提取
            current_hop = 2
            subset, _, _, _ = k_hop_subgraph(
                node_idx=index, 
                num_hops=current_hop,
                edge_index=self.data.edge_index, 
                relabel_nodes=True
            )
            
            # 如果子图太小，增加hop数
            while len(subset) < self.smallest_size and current_hop < 5:
                current_hop += 1
                subset, _, _, _ = k_hop_subgraph(
                    node_idx=index, 
                    num_hops=current_hop,
                    edge_index=self.data.edge_index
                )
            
            # 如果仍然太小，添加同类节点
            if len(subset) < self.smallest_size:
                need_node_num = self.smallest_size - len(subset)
                pos_nodes = torch.argwhere(self.data.y == int(current_label))
                pos_nodes = pos_nodes.to('cpu')
                subset = subset.to('cpu')
                candidate_nodes = torch.from_numpy(
                    np.setdiff1d(pos_nodes.numpy(), subset.numpy())
                )
                if len(candidate_nodes) > 0:
                    candidate_nodes = candidate_nodes[
                        torch.randperm(candidate_nodes.shape[0])
                    ][:need_node_num]
                    subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])
            
            # 如果太大，随机采样
            if len(subset) > self.largest_size:
                subset = subset[torch.randperm(subset.shape[0])][:self.largest_size - 1]
                subset = torch.unique(torch.cat([
                    torch.LongTensor([index]).to(self.device), 
                    torch.flatten(subset)
                ]))
            
            # 确保在正确设备上
            subset = subset.to(self.device)
            
            # 提取子图
            sub_edge_index, _ = subgraph(subset, self.data.edge_index, relabel_nodes=True)
            sub_edge_index = sub_edge_index.to(self.device)
            
            # 提取节点特征
            x = self.data.x[subset]
            
            # 创建induced graph，添加index属性以保持兼容性
            induced_graph = Data(
                x=x, 
                edge_index=sub_edge_index, 
                y=current_label, 
                index=index
            )
            
            induced_graph_list.append(induced_graph)
            
            if index % 500 == 0:
                print(f"Processed {index} nodes...")
        
        return induced_graph_list
    
    def get_data_loaders(self):
        """
        获取训练和测试数据加载器，兼容node_task.py中的使用方式
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        # 创建induced graphs
        if self.graphs_list is None:
            print("Creating induced graphs...")
            self.graphs_list = self._create_induced_graphs()
            print("Done!")
        
        # 分离训练和测试图
        train_graphs = []
        test_graphs = []
        
        print('Distinguishing the train dataset and test dataset...')
        for graph in self.graphs_list:
            if graph.index in self.data.idx_train:
                train_graphs.append(graph)
            elif graph.index in self.data.idx_test:
                test_graphs.append(graph)
        print('Done!!!')
        
        # 创建数据集
        train_dataset = GraphDataset(train_graphs)
        test_dataset = GraphDataset(test_graphs)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print("Prepare induced graph data is finished!")
        
        return train_loader, test_loader
    
    def get_original_data(self):
        """
        获取原始图数据
        
        Returns:
            Data: 原始图数据对象
        """
        return self.data
    
    def save_induced_graphs(self, save_path):
        """
        保存induced graphs到文件
        
        Args:
            save_path: 保存路径
        """
        if self.graphs_list is None:
            self.graphs_list = self._create_induced_graphs()
        
        # 转换到CPU以便保存
        saved_graph_list = [deepcopy(graph).to('cpu') for graph in self.graphs_list]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(saved_graph_list, f)
        print(f"Induced graph data has been written to {save_path}")


def create_npz_dataloader(dataset_name, npz_file_path, device='cpu', batch_size=32,
                         smallest_size=10, largest_size=300):
    """
    便捷函数：创建NPZ数据加载器
    
    Args:
        dataset_name: 数据集名称
        npz_file_path: npz文件路径
        device: 设备类型
        batch_size: 批处理大小
        smallest_size: 子图最小节点数
        largest_size: 子图最大节点数
    
    Returns:
        NPZDataLoader: 数据加载器实例
    """
    return NPZDataLoader(
        dataset_name=dataset_name,
        npz_file_path=npz_file_path,
        device=device,
        batch_size=batch_size,
        smallest_size=smallest_size,
        largest_size=largest_size
    )
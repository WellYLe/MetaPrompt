"""
边扰动预测模型训练脚本
用于训练一个能够预测每条边是否应该被扰动的模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm


class EdgePerturbationPredictor(nn.Module):
    """边扰动预测模型"""
    
    def __init__(self, feature_dim, hidden_dim=64, dropout=0.1):
        super(EdgePerturbationPredictor, self).__init__()
        
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 图级别特征编码器（可选）
        self.graph_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(1 + hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features, edge_index, return_edge_features=False):
        """
        Args:
            node_features: [num_nodes, feature_dim]
            edge_index: [2, num_edges]
        Returns:
            edge_probs: [num_edges] 边扰动概率
        """
        # 获取边的端点特征
        src_features = node_features[edge_index[0]]  # [num_edges, feature_dim]
        dst_features = node_features[edge_index[1]]  # [num_edges, feature_dim]
        
        # 拼接端点特征
        edge_features = torch.cat([src_features, dst_features], dim=1)  # [num_edges, 2*feature_dim]
        
        # 编码边特征
        edge_encoded = self.edge_encoder(edge_features)  # [num_edges, 1]
        
        # 编码图级别特征（全局平均）
        graph_features = torch.mean(node_features, dim=0, keepdim=True)  # [1, feature_dim]
        graph_encoded = self.graph_encoder(graph_features)  # [1, hidden_dim//4]
        graph_encoded = graph_encoded.expand(edge_encoded.size(0), -1)  # [num_edges, hidden_dim//4]
        
        # 组合特征
        combined_features = torch.cat([edge_encoded, graph_encoded], dim=1)  # [num_edges, 1 + hidden_dim//4]
        
        # 预测扰动概率
        edge_probs = self.predictor(combined_features).squeeze(-1)  # [num_edges]
        
        if return_edge_features:
            return edge_probs, edge_features
        return edge_probs


class EdgePerturbationDataset:
    """边扰动数据集生成器"""
    
    def __init__(self, node_features, adj_matrix, perturbation_ratio=0.1):
        self.node_features = node_features
        self.adj_matrix = adj_matrix
        self.perturbation_ratio = perturbation_ratio
        
        # 获取所有可能的边
        self.all_edges = self._get_all_possible_edges()
        self.existing_edges = self._get_existing_edges()
        
    def _get_all_possible_edges(self):
        """获取所有可能的边（不包括自环）"""
        num_nodes = self.adj_matrix.shape[0]
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # 无向图，只考虑上三角
                edges.append([i, j])
        return np.array(edges)
    
    def _get_existing_edges(self):
        """获取现有的边"""
        edges = np.nonzero(np.triu(self.adj_matrix, k=1))
        return np.column_stack([edges[0], edges[1]])
    
    def generate_perturbation_labels(self, strategy='random'):
        """
        生成边扰动标签
        Args:
            strategy: 'random', 'degree_based', 'centrality_based'
        Returns:
            edge_index: [2, num_edges]
            labels: [num_edges] 0表示不扰动，1表示扰动
        """
        if strategy == 'random':
            return self._random_perturbation()
        elif strategy == 'degree_based':
            return self._degree_based_perturbation()
        elif strategy == 'centrality_based':
            return self._centrality_based_perturbation()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _random_perturbation(self):
        """随机选择边进行扰动"""
        num_perturb = int(len(self.existing_edges) * self.perturbation_ratio)
        
        # 随机选择要扰动的边
        perturb_indices = np.random.choice(len(self.existing_edges), num_perturb, replace=False)
        
        labels = np.zeros(len(self.existing_edges))
        labels[perturb_indices] = 1
        
        edge_index = torch.tensor(self.existing_edges.T, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)
        
        return edge_index, labels
    
    def _degree_based_perturbation(self):
        """基于度的扰动：优先扰动连接高度节点的边"""
        degrees = np.sum(self.adj_matrix, axis=1)
        
        # 计算每条边的度分数（端点度的乘积）
        edge_scores = []
        for edge in self.existing_edges:
            score = degrees[edge[0]] * degrees[edge[1]]
            edge_scores.append(score)
        
        edge_scores = np.array(edge_scores)
        
        # 选择分数最高的边进行扰动
        num_perturb = int(len(self.existing_edges) * self.perturbation_ratio)
        perturb_indices = np.argsort(edge_scores)[-num_perturb:]
        
        labels = np.zeros(len(self.existing_edges))
        labels[perturb_indices] = 1
        
        edge_index = torch.tensor(self.existing_edges.T, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)
        
        return edge_index, labels
    
    def _centrality_based_perturbation(self):
        """基于中心性的扰动：优先扰动连接中心节点的边"""
        # 简单的度中心性
        degrees = np.sum(self.adj_matrix, axis=1)
        centrality = degrees / (len(degrees) - 1)
        
        # 计算每条边的中心性分数
        edge_scores = []
        for edge in self.existing_edges:
            score = (centrality[edge[0]] + centrality[edge[1]]) / 2
            edge_scores.append(score)
        
        edge_scores = np.array(edge_scores)
        
        # 选择分数最高的边进行扰动
        num_perturb = int(len(self.existing_edges) * self.perturbation_ratio)
        perturb_indices = np.argsort(edge_scores)[-num_perturb:]
        
        labels = np.zeros(len(self.existing_edges))
        labels[perturb_indices] = 1
        
        edge_index = torch.tensor(self.existing_edges.T, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)
        
        return edge_index, labels


def train_edge_predictor(node_features, adj_matrix, 
                        num_epochs=200, lr=0.001, 
                        device='cpu', save_path=None):
    """
    训练边扰动预测模型
    
    Args:
        node_features: numpy array [num_nodes, feature_dim]
        adj_matrix: numpy array [num_nodes, num_nodes]
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
        save_path: 模型保存路径
    
    Returns:
        trained_model: 训练好的模型
        training_history: 训练历史
    """
    
    # 转换为tensor
    node_features = torch.tensor(node_features, dtype=torch.float).to(device)
    
    # 创建数据集
    dataset = EdgePerturbationDataset(node_features.cpu().numpy(), adj_matrix)
    
    # 初始化模型
    feature_dim = node_features.size(1)
    model = EdgePerturbationPredictor(feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_auc': [],
        'train_ap': []
    }
    
    print("开始训练边扰动预测模型...")
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        
        # 生成训练数据（每个epoch使用不同的扰动策略）
        strategies = ['random', 'degree_based', 'centrality_based']
        strategy = strategies[epoch % len(strategies)]
        edge_index, labels = dataset.generate_perturbation_labels(strategy)
        edge_index, labels = edge_index.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        pred_probs = model(node_features, edge_index)
        loss = criterion(pred_probs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            pred_probs_np = pred_probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            if len(np.unique(labels_np)) > 1:  # 确保有正负样本
                auc = roc_auc_score(labels_np, pred_probs_np)
                ap = average_precision_score(labels_np, pred_probs_np)
            else:
                auc = ap = 0.0
        
        # 记录历史
        history['train_loss'].append(loss.item())
        history['train_auc'].append(auc)
        history['train_ap'].append(ap)
        
        # 打印进度
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Loss: {loss.item():.4f}')
            print(f'  AUC: {auc:.4f}')
            print(f'  AP: {ap:.4f}')
    
    # 保存模型
    if save_path:
        torch.save(model, save_path)
        print(f"模型已保存到: {save_path}")
    
    return model, history


def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # AUC
    axes[1].plot(history['train_auc'])
    axes[1].set_title('Training AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].grid(True)
    
    # AP
    axes[2].plot(history['train_ap'])
    axes[2].set_title('Training AP')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Average Precision')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # 示例：使用随机数据训练模型
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 生成示例数据
    num_nodes = 100
    feature_dim = 16
    
    # 随机节点特征
    node_features = np.random.randn(num_nodes, feature_dim)
    
    # 随机邻接矩阵（稀疏）
    adj_matrix = np.random.rand(num_nodes, num_nodes) < 0.1
    adj_matrix = adj_matrix.astype(float)
    adj_matrix = np.triu(adj_matrix, k=1)  # 上三角
    adj_matrix = adj_matrix + adj_matrix.T  # 对称化
    
    # 训练模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model, history = train_edge_predictor(
        node_features=node_features,
        adj_matrix=adj_matrix,
        num_epochs=100,
        lr=0.001,
        device=device,
        save_path='edge_predictor_model.pth'
    )
    
    # 绘制训练历史
    plot_training_history(history, 'training_history.png')
    
    print("训练完成！")
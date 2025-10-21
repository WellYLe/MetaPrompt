import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

# 添加ProG路径
current_dir = os.path.dirname(os.path.abspath(__file__))
prog_path = os.path.join(os.path.dirname(current_dir), 'ProG')
if prog_path not in sys.path:
    sys.path.insert(0, prog_path)

from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from prompt_graph.pretrain.base import PreTrain

def initialize_gnn_encoder(gnn_type, input_dim, hid_dim, num_layer, device):
    """初始化GNN编码器"""
    if gnn_type == 'GAT':
        gnn = GAT(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GCN':
        gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GraphSAGE':
        gnn = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GIN':
        gnn = GIN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GCov':
        gnn = GCov(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GraphTransformer':
        gnn = GraphTransformer(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    else:
        raise ValueError(f"Unsupported GNN type: {gnn_type}")
    gnn.to(device)
    return gnn

class EdgeFlipMAELoss(nn.Module):
    """基于MAE思想的边翻转检测损失函数"""
    
    def __init__(self, encoder, edge_classifier, node_feat_dim, hid_dim, 
                 mask_rate=0.15, noise_rate=0.1, device='cpu'):
        super(EdgeFlipMAELoss, self).__init__()
        self.encoder = encoder
        self.edge_classifier = edge_classifier
        self.mask_rate = mask_rate
        self.noise_rate = noise_rate
        self.device = device
        
        # 掩码token用于节点特征掩蔽
        self.mask_token = nn.Parameter(torch.zeros(1, node_feat_dim))
        
        # 重构头：从隐藏表示重构原始节点特征
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, node_feat_dim)
        ).to(device)
        
        # 损失函数
        self.edge_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.MSELoss()
        
    def forward(self, data, edge_pairs, edge_labels):
        """
        Args:
            data: PyG Data对象，包含x和edge_index
            edge_pairs: tensor [N, 2] - 边的节点对
            edge_labels: tensor [N] - 边翻转标签 (1=翻转, 0=未翻转)
        """
        # 1. 节点特征掩蔽 (MAE风格)
        masked_x, mask_indices = self.mask_node_features(data.x)
        
        # 2. 编码器前向传播
        node_embeddings = self.encoder(x=masked_x, edge_index=data.edge_index)
        
        # 3. 边分类任务
        edge_logits = self.classify_edges(node_embeddings, edge_pairs)
        edge_loss = self.edge_criterion(edge_logits, edge_labels.float())
        
        # 4. 节点特征重构任务 (MAE风格的辅助任务)
        if len(mask_indices) > 0:
            reconstructed_features = self.reconstruction_head(node_embeddings[mask_indices])
            original_features = data.x[mask_indices]
            recon_loss = self.recon_criterion(reconstructed_features, original_features)
        else:
            recon_loss = torch.tensor(0.0, device=self.device)
        
        # 5. 总损失 = 边分类损失 + 重构损失
        total_loss = edge_loss + 0.1 * recon_loss  # 重构损失权重较小
        
        return total_loss, edge_loss, recon_loss, edge_logits
    
    def mask_node_features(self, x):
        """随机掩蔽部分节点特征"""
        num_nodes = x.size(0)
        num_mask = int(self.mask_rate * num_nodes)
        
        if num_mask == 0:
            return x, []
        
        # 随机选择要掩蔽的节点
        perm = torch.randperm(num_nodes, device=x.device)
        mask_indices = perm[:num_mask]
        
        # 创建掩蔽后的特征
        masked_x = x.clone()
        
        # 部分用mask token替换，部分用噪声替换
        num_noise = int(self.noise_rate * num_mask)
        if num_noise > 0:
            noise_indices = mask_indices[:num_noise]
            token_indices = mask_indices[num_noise:]
            
            # 噪声替换：用其他随机节点的特征
            noise_nodes = torch.randperm(num_nodes, device=x.device)[:num_noise]
            masked_x[noise_indices] = x[noise_nodes]
        else:
            token_indices = mask_indices
        
        # mask token替换
        if len(token_indices) > 0:
            masked_x[token_indices] = self.mask_token
        
        return masked_x, mask_indices
    
    def classify_edges(self, node_embeddings, edge_pairs):
        """基于节点嵌入对边进行分类"""
        # 获取边两端节点的嵌入
        src_embeddings = node_embeddings[edge_pairs[:, 0]]  # [N, hid_dim]
        dst_embeddings = node_embeddings[edge_pairs[:, 1]]  # [N, hid_dim]
        
        # 边表示：拼接两个节点的嵌入
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)  # [N, 2*hid_dim]
        
        # 边分类
        edge_logits = self.edge_classifier(edge_embeddings).squeeze()  # [N]
        
        return edge_logits

class EdgeFlipMAE(PreTrain):
    """基于MAE框架的边翻转检测模型"""
    
    def __init__(self, gnn_type='GCN', dataset_name='Cora', input_dim=100, hid_dim=64, 
                 num_layer=2, device=0, mask_rate=0.15, noise_rate=0.1, 
                 learning_rate=0.001, weight_decay=5e-4, epochs=200, **kwargs):
        
        # 初始化基类
        super().__init__(gnn_type=gnn_type, dataset_name=dataset_name, 
                        input_dim=input_dim, hid_dim=hid_dim, gln=num_layer,
                        num_epoch=epochs, device=device, **kwargs)
        
        # 模型参数
        self.mask_rate = mask_rate
        self.noise_rate = noise_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 初始化模型组件
        self.encoder = initialize_gnn_encoder(gnn_type, input_dim, hid_dim, num_layer, self.device)
        
        # 边分类器：输入是两个节点嵌入的拼接
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, 1)
        ).to(self.device)
        
        # 损失函数
        self.loss_fn = EdgeFlipMAELoss(
            encoder=self.encoder,
            edge_classifier=self.edge_classifier,
            node_feat_dim=input_dim,
            hid_dim=hid_dim,
            mask_rate=mask_rate,
            noise_rate=noise_rate,
            device=self.device
        )
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.edge_classifier.parameters()) + 
            list(self.loss_fn.reconstruction_head.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def load_triplet_data(self, edge_pairs, X_pairs, labels, graph_data, 
                         train_ratio=0.7, val_ratio=0.15):
        """加载三元组数据并划分训练/验证/测试集"""
        
        # 数据划分
        n_samples = len(labels)
        n_train = int(train_ratio * n_samples)
        n_val = int(val_ratio * n_samples)
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # 转换为tensor
        edge_pairs_tensor = torch.tensor(edge_pairs, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        
        # 创建数据集
        self.train_dataset = TensorDataset(
            edge_pairs_tensor[train_indices], 
            labels_tensor[train_indices]
        )
        self.val_dataset = TensorDataset(
            edge_pairs_tensor[val_indices], 
            labels_tensor[val_indices]
        )
        self.test_dataset = TensorDataset(
            edge_pairs_tensor[test_indices], 
            labels_tensor[test_indices]
        )
        
        # 图数据
        self.graph_data = graph_data
        
        print(f"数据加载完成: 训练集 {len(self.train_dataset)}, "
              f"验证集 {len(self.val_dataset)}, 测试集 {len(self.test_dataset)}")
    
    def train_one_epoch(self, dataloader):
        """训练一个epoch"""
        self.encoder.train()
        self.edge_classifier.train()
        
        total_loss = 0
        total_edge_loss = 0
        total_recon_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_edge_pairs, batch_labels in dataloader:
            batch_edge_pairs = batch_edge_pairs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            loss, edge_loss, recon_loss, logits = self.loss_fn(
                self.graph_data, batch_edge_pairs, batch_labels
            )
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_edge_loss += edge_loss.item()
            total_recon_loss += recon_loss.item()
            
            # 收集预测结果
            with torch.no_grad():
                preds = torch.sigmoid(logits)
                all_preds.append(preds.cpu())
                all_labels.append(batch_labels.cpu())
        
        # 计算指标
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        metrics = self.calculate_metrics(all_preds, all_labels)
        
        return {
            'total_loss': total_loss / len(dataloader),
            'edge_loss': total_edge_loss / len(dataloader),
            'recon_loss': total_recon_loss / len(dataloader),
            **metrics
        }
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.encoder.eval()
        self.edge_classifier.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_edge_pairs, batch_labels in dataloader:
                batch_edge_pairs = batch_edge_pairs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                loss, _, _, logits = self.loss_fn(
                    self.graph_data, batch_edge_pairs, batch_labels
                )
                
                total_loss += loss.item()
                
                preds = torch.sigmoid(logits)
                all_preds.append(preds.cpu())
                all_labels.append(batch_labels.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        metrics = self.calculate_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def calculate_metrics(self, preds, labels, threshold=0.5):
        """计算分类指标"""
        preds_binary = (preds > threshold).float()
        
        accuracy = accuracy_score(labels, preds_binary)
        precision = precision_score(labels, preds_binary, zero_division=0)
        recall = recall_score(labels, preds_binary, zero_division=0)
        f1 = f1_score(labels, preds_binary, zero_division=0)
        
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def pretrain(self, batch_size=64):
        """预训练模型"""
        
        # 创建数据加载器
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_f1 = 0
        patience = 20
        wait = 0
        
        print(f"开始训练EdgeFlipMAE模型...")
        print(f"数据集: {self.dataset_name}, GNN类型: {self.gnn_type}")
        print(f"隐藏维度: {self.hid_dim}, 掩码率: {self.mask_rate}")
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # 训练
            train_metrics = self.train_one_epoch(train_loader)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{self.epochs} | Time: {epoch_time:.2f}s")
            print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # 早停
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                wait = 0
                # 保存最佳模型
                self.save_model()
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print("-" * 60)
    
    def save_model(self):
        """保存模型"""
        folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 保存编码器
        encoder_path = f"{folder_path}/EdgeFlipMAE.{self.gnn_type}.{self.hid_dim}hidden_dim.encoder.pth"
        torch.save(self.encoder.state_dict(), encoder_path)
        
        # 保存边分类器
        classifier_path = f"{folder_path}/EdgeFlipMAE.{self.gnn_type}.{self.hid_dim}hidden_dim.classifier.pth"
        torch.save(self.edge_classifier.state_dict(), classifier_path)
        
        print(f"模型已保存: {encoder_path}")
        print(f"分类器已保存: {classifier_path}")
    
    def load_model(self, encoder_path, classifier_path):
        """加载预训练模型"""
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.edge_classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        print(f"模型加载完成: {encoder_path}, {classifier_path}")
    
    def predict_edge_flips(self, edge_pairs, graph_data):
        """预测边是否被翻转"""
        self.encoder.eval()
        self.edge_classifier.eval()
        
        edge_pairs_tensor = torch.tensor(edge_pairs, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # 获取节点嵌入
            node_embeddings = self.encoder(x=graph_data.x, edge_index=graph_data.edge_index)
            
            # 边分类
            src_embeddings = node_embeddings[edge_pairs_tensor[:, 0]]
            dst_embeddings = node_embeddings[edge_pairs_tensor[:, 1]]
            edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
            
            logits = self.edge_classifier(edge_embeddings).squeeze()
            probs = torch.sigmoid(logits)
        
        return probs.cpu().numpy()
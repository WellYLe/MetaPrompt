"""
Enhanced Metattack with GPF (Graph Prompt Feature) Layer and Edge Perturbation Predictor
基于原始Metattack，集成了可训练的特征提示和冻结的边扰动预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack
import math


class GPF(nn.Module):
    """Graph Prompt Feature Layer - 可训练的特征提示层"""
    
    def __init__(self, in_channels: int, prompt_type='global'):
        super(GPF, self).__init__()
        self.prompt_type = prompt_type
        
        if prompt_type == 'global':
            # 全局特征提示：所有节点共享同一个提示向量
            self.global_prompt = Parameter(torch.Tensor(1, in_channels))
        elif prompt_type == 'node_specific':
            # 节点特定提示：每个节点有独立的提示向量（内存消耗大）
            self.node_prompts = None  # 需要在forward时根据节点数初始化
        elif prompt_type == 'learnable_embedding':
            # 可学习嵌入提示：通过小型网络生成提示
            self.prompt_generator = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.ReLU(),
                nn.Linear(in_channels // 2, in_channels)
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if hasattr(self, 'global_prompt'):
            nn.init.xavier_uniform_(self.global_prompt)
        elif hasattr(self, 'prompt_generator'):
            for layer in self.prompt_generator:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 节点特征矩阵 [num_nodes, in_channels]
        Returns:
            prompted_x: 添加提示后的特征矩阵
        """
        if self.prompt_type == 'global':
            return x + self.global_prompt
        elif self.prompt_type == 'node_specific':
            if self.node_prompts is None:
                self.node_prompts = Parameter(torch.Tensor(x.size(0), x.size(1))).to(x.device)
                nn.init.xavier_uniform_(self.node_prompts)
            return x + self.node_prompts
        elif self.prompt_type == 'learnable_embedding':
            prompt = self.prompt_generator(x)
            return x + prompt
        else:
            return x


class EdgePerturbationPredictor(nn.Module):
    """边扰动预测模型 - 预测每条边被扰动的概率"""
    
    def __init__(self, feature_dim, hidden_dim=64):
        super(EdgePerturbationPredictor, self).__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features, edge_index):
        """
        Args:
            node_features: [num_nodes, feature_dim]
            edge_index: [2, num_edges] 边索引
        Returns:
            edge_probs: [num_edges] 每条边的扰动概率
        """
        # 获取边的端点特征
        src_features = node_features[edge_index[0]]  # [num_edges, feature_dim]
        dst_features = node_features[edge_index[1]]  # [num_edges, feature_dim]
        
        # 拼接端点特征
        edge_features = torch.cat([src_features, dst_features], dim=1)  # [num_edges, 2*feature_dim]
        
        # 预测扰动概率
        edge_probs = self.edge_encoder(edge_features).squeeze(-1)  # [num_edges]
        
        return edge_probs


class EnhancedMetattack(BaseAttack):
    """增强版Metattack，集成GPF层和边扰动预测"""
    
    def __init__(self, model, nnodes, feature_shape=None, 
                 edge_predictor_path=None, 
                 attack_structure=True, attack_features=False, 
                 undirected=True, device='cpu', 
                 lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9,
                 gpf_lr=0.01, edge_weight=1.0):
        
        super(EnhancedMetattack, self).__init__(model, nnodes, attack_structure, attack_features, device)
        
        self.lambda_ = lambda_
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.gpf_lr = gpf_lr
        self.edge_weight = edge_weight  # 边预测损失的权重
        
        # 初始化GPF层
        if feature_shape is not None:
            self.gpf = GPF(in_channels=feature_shape[1], prompt_type='global').to(device)
            self.gpf_optimizer = torch.optim.Adam(self.gpf.parameters(), lr=gpf_lr)
        
        # 加载冻结的边扰动预测模型
        if edge_predictor_path is not None:
            self.edge_predictor = torch.load(edge_predictor_path, map_location=device)
            self.edge_predictor.eval()
            for param in self.edge_predictor.parameters():
                param.requires_grad = False
        else:
            self.edge_predictor = None
        
        # 初始化攻击参数
        if attack_structure:
            self.undirected = undirected
            self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
            self.adj_changes.data.fill_(0)
        
        if attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)
        
        # 初始化代理模型权重
        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass
        self.with_bias = False
        self.with_relu = model.with_relu
        
        self._init_surrogate_weights()
    
    def _init_surrogate_weights(self):
        """初始化代理模型的权重和偏置"""
        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []
        
        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(self.device))
            w_velocity = torch.zeros(weight.shape).to(self.device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)
            
            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(self.device))
                b_velocity = torch.zeros(bias.shape).to(self.device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)
            
            previous_size = nhid
        
        # 输出层
        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(self.device))
        output_w_velocity = torch.zeros(output_weight.shape).to(self.device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)
        
        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(self.device))
            output_b_velocity = torch.zeros(output_bias.shape).to(self.device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重参数"""
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)
        
        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(b.size(0))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)
    
    def get_modified_adj(self, ori_adj):
        """获取修改后的邻接矩阵"""
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj
    
    def self_training_label(self, labels, idx_train):
        """生成自训练标签"""
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training
    
    def inner_train_with_gpf(self, features, adj_norm, idx_train, idx_unlabeled, labels, create_graph=True):
        """使用GPF层的内层训练"""
        self._initialize_weights()
        
        # 重新设置梯度追踪
        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True
            
            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True
        
        # 应用GPF层
        prompted_features = self.gpf(features)
        
        for j in range(self.train_iters):
            hidden = prompted_features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                hidden = adj_norm @ hidden @ w + b
                
                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)
            
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            
            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=create_graph)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=create_graph)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]
            
            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]
        
        return prompted_features
    
    def get_meta_grad_with_edge_guidance(self, features, adj_norm, idx_train, idx_unlabeled, 
                                       labels, labels_self_training, edge_index=None):
        """计算带边指导的元梯度"""
        # 前向传播
        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)
        
        output = F.log_softmax(hidden, dim=1)
        
        # 计算基础损失
        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        
        # 组合攻击损失
        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled
        
        # 如果有边预测模型，添加边指导损失
        if self.edge_predictor is not None and edge_index is not None:
            edge_probs = self.edge_predictor(features, edge_index)
            # 边指导损失：鼓励高概率边被扰动
            edge_guidance_loss = -torch.mean(edge_probs)  # 最大化边扰动概率
            attack_loss = attack_loss + self.edge_weight * edge_guidance_loss
        
        print(f'GCN loss on unlabeled data: {loss_test_val.item():.4f}')
        print(f'GCN acc on unlabeled data: {utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item():.4f}')
        print(f'Attack loss: {attack_loss.item():.4f}')
        
        # 计算梯度
        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        
        return adj_grad, feature_grad, attack_loss
    
    def get_adj_score(self, adj_grad, modified_adj, ori_adj):
        """计算邻接矩阵分数"""
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        adj_meta_grad = adj_meta_grad - adj_meta_grad.min()
        adj_meta_grad = adj_meta_grad - torch.diag(torch.diag(adj_meta_grad, 0))
        return adj_meta_grad
    
    def attack_with_gpf(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, 
                       n_perturbations, edge_index=None):
        """使用GPF层的攻击方法"""
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        
        # 获取边索引（如果未提供）
        if edge_index is None and self.edge_predictor is not None:
            edge_index = torch.nonzero(ori_adj).t().contiguous()
        
        labels_self_training = self.self_training_label(labels, idx_train)
        
        for i in tqdm(range(n_perturbations), desc="Enhanced Metattack with GPF"):
            # 获取修改后的邻接矩阵
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
            else:
                modified_adj = ori_adj
            
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            
            # 使用GPF层进行内层训练
            prompted_features = self.inner_train_with_gpf(
                ori_features, adj_norm, idx_train, idx_unlabeled, labels, create_graph=True
            )
            
            # 计算带边指导的元梯度
            adj_grad, feature_grad, attack_loss = self.get_meta_grad_with_edge_guidance(
                prompted_features, adj_norm, idx_train, idx_unlabeled, 
                labels, labels_self_training, edge_index
            )
            
            # 优化GPF层参数
            self.gpf_optimizer.zero_grad()
            attack_loss.backward(retain_graph=True)
            self.gpf_optimizer.step()
            
            # 更新结构扰动
            if self.attack_structure and adj_grad is not None:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj)
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
        
        # 保存最终结果
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        
        return self.modified_adj, self.gpf
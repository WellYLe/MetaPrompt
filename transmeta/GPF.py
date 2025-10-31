import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_geometric.data import Data
class GPF(torch.nn.Module):
    def __init__(self, in_channels: int, attacker=None):
        super(GPF, self).__init__()
        self.global_emb = torch.nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()
        self.attacker = attacker

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        # 确保输入是 torch.Tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.global_emb.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.global_emb.device)
        
        # 确保 x 在正确的设备上
        if x.device != self.global_emb.device:
            x = x.to(self.global_emb.device)
            
        return x + self.global_emb
    
    def train(self, train_graphs, attacker, surrogate, answering, optimizer, device='cuda',
              budget_ratio=0.50,     # 预算比例，例如5%边可攻击
              mu_init=0.0,           # 初始乘子
              mu_lr=1e-3):           # 乘子学习率
        """训练GPF提示"""
        total_loss = 0.0    
        mu = torch.tensor(mu_init, device=device)  # 拉格朗日乘子 (非负标量)
        
        # 初始化最佳prompt保存
        if not hasattr(self, 'best_loss'):
            self.best_loss = float('inf')
            self.best_prompt_state = None
            
        for batch in train_graphs:  
            optimizer.zero_grad() 
            batch = batch.to(device)
            
            # 检查batch是否有必要的属性
            if not hasattr(batch, 'x') or not hasattr(batch, 'edge_index'):
                # 从子图数据构造PyG Data格式
                if hasattr(batch, 'orig_node_idx'):
                    # 这是partition_graph_equal返回的子图
                    feature = batch.x  # 节点特征
                    edge_index = batch.edge_index  # 边索引
                else:
                    print("警告: batch缺少必要的图数据属性")
                    continue
            else:
                feature = batch.x
                edge_index = batch.edge_index
            
            # 构造图数据用于预测（使用原始特征而不是编码后的嵌入）
            graph_data = Data(x=feature, edge_index=edge_index)
            
            # 使用EdgeFlipMAE预测边翻转概率
            flip_probs = attacker.predict_all_edges(graph_data)
            
            # 通过编码器获得节点嵌入（用于后续的提示处理）
            node_embeddings = attacker.encoder(feature, edge_index)
            
            # 添加提示
            prompted_embeddings = self.add(node_embeddings)
            
            # 构造扰动后的邻接矩阵（软扰动，用于梯度传播）
            num_nodes = feature.shape[0]
            adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
            adj_matrix[edge_index[0], edge_index[1]] = 1.0
            
            # 软扰动：使用概率而不是硬决策
            flip_probs_tensor = torch.tensor(flip_probs, device=device)
            edge_probs_matrix = torch.zeros_like(adj_matrix)
            edge_probs_matrix[edge_index[0], edge_index[1]] = flip_probs_tensor
            
            # 扰动后的软邻接矩阵
            perturbed_adj = adj_matrix * (1 - edge_probs_matrix) + (1 - adj_matrix) * edge_probs_matrix
            
            # 归一化邻接矩阵
            adj_norm = self._normalize_adj(perturbed_adj)
            
            # 通过代理模型预测
            surrogate_output = surrogate(prompted_embeddings, adj_norm)
            final_output = answering(surrogate_output)
            
            # 计算损失（这里需要真实标签，假设batch.y存在）
            if hasattr(batch, 'y'):
                criterion = nn.CrossEntropyLoss()
                loss = criterion(final_output, batch.y)
            else:
                # 如果没有标签，可以使用自监督损失或跳过
                print("警告: batch缺少标签信息")
                continue
            # --- 拉格朗日预算约束 ---
            num_edges = edge_index.size(1)
            budget = budget_ratio * num_edges  # 允许攻击的期望边数
            g = flip_probs_tensor.sum() - budget   # 约束项：sum(p) - B
            budget_loss = mu * g                   # 拉格朗日项 μ * g

            # --- 总损失 ---
            loss = loss + budget_loss
            loss.backward()  
            optimizer.step()
            # --- 更新拉格朗日乘子 ---
            with torch.no_grad():
                mu = torch.clamp(mu + mu_lr * g, min=0.0)
            total_loss += loss.item()  
        
        # 计算平均损失
        avg_loss = total_loss / len(train_graphs) if len(train_graphs) > 0 else 0.0
        
        # 保存最佳prompt状态
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_prompt_state = self.global_emb.clone().detach()
            print(f"保存最佳prompt，损失: {avg_loss:.4f}")
            
        return avg_loss
    
    def _normalize_adj(self, adj):
        """归一化邻接矩阵"""
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
        D = torch.sum(adj, dim=1)
        D_inv = torch.pow(D, -0.5)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)
        adj_norm = D_mat_inv @ adj @ D_mat_inv
        return adj_norm 
    
    def load_best_prompt(self):
        """加载最佳prompt状态"""
        if hasattr(self, 'best_prompt_state') and self.best_prompt_state is not None:
            self.global_emb.data = self.best_prompt_state.clone()
            print(f"加载最佳prompt，最佳损失: {self.best_loss:.4f}")
        else:
            print("警告: 没有找到最佳prompt状态")
    
    def reset_best_prompt(self):
        """重置最佳prompt记录"""
        self.best_loss = float('inf')
        self.best_prompt_state = None 
    
    
class GPF_plus(torch.nn.Module): 
    def __init__(self, in_channels: int, p_num: int, original_dim=None):
        super(GPF_plus, self).__init__()
        self.in_channels = in_channels
        self.p_num = p_num

        # 多个可学习的prompt参数
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        # 一个线性层，用于根据节点特征计算每个prompt的权重
        self.a = torch.nn.Linear(in_channels, p_num)
        
        # 如果输入维度和in_channels不一致，添加投影层
        self.projection = None
        if original_dim is not None and original_dim != in_channels:
            self.projection = torch.nn.Linear(original_dim, in_channels)
            print(f"Created projection layer: {original_dim} -> {in_channels}")
            
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()
        if self.projection is not None:
            self.projection.reset_parameters()

    def add(self, x: torch.Tensor):
        """根据节点特征自适应生成Prompt"""
        # 维度对齐
        if self.projection is not None and x.shape[1] != self.in_channels:
            x = self.projection(x)
            
        # 每个节点计算对每个prompt的注意力得分
        score = self.a(x)
        weight = F.softmax(score, dim=1)  # [num_nodes, p_num]

        # 加权融合所有prompt
        p = weight.mm(self.p_list)        # [num_nodes, in_channels]
        return x + p

    def _normalize_adj(self, adj):
        """归一化邻接矩阵"""
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
        D = torch.sum(adj, dim=1)
        D_inv = torch.pow(D, -0.5)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)
        adj_norm = D_mat_inv @ adj @ D_mat_inv
        return adj_norm

    def train(self, train_graphs, attacker, surrogate, answering, optimizer, device='cuda',
              budget_ratio=0.05,     # 预算比例，例如5%边可攻击
              mu_init=0.0,           # 初始乘子
              mu_lr=1e-2):           # 乘子学习率
        """
        GPF+的训练过程
        与GPF的区别在于：这里每个节点都有自己组合出来的prompt p，而不是全局共享的global_emb
        """
        total_loss = 0.0
        mu = torch.tensor(mu_init, device=device)  # 拉格朗日乘子 (非负标量)
        
        # 初始化最佳prompt保存
        if not hasattr(self, 'best_loss'):
            self.best_loss = float('inf')
            self.best_prompt_state = None
            
        for batch in train_graphs:
            optimizer.zero_grad()
            batch = batch.to(device)

            # 检查batch合法性
            if not hasattr(batch, 'x') or not hasattr(batch, 'edge_index'):
                print("警告: batch缺少必要的图数据属性")
                continue

            feature = batch.x
            edge_index = batch.edge_index

            # EdgeFlipMAE预测边翻转概率
            graph_data = Data(x=feature, edge_index=edge_index)
            flip_probs = attacker.predict_all_edges(graph_data)

            # 通过编码器获得节点嵌入
            node_embeddings = attacker.encoder(feature, edge_index)

            # 使用GPF_plus添加prompt（节点自适应prompt）
            prompted_embeddings = self.add(node_embeddings)

            # 构造扰动的软邻接矩阵
            num_nodes = feature.shape[0]
            adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
            adj_matrix[edge_index[0], edge_index[1]] = 1.0

            flip_probs_tensor = torch.tensor(flip_probs, device=device)
            edge_probs_matrix = torch.zeros_like(adj_matrix)
            edge_probs_matrix[edge_index[0], edge_index[1]] = flip_probs_tensor

            perturbed_adj = adj_matrix * (1 - edge_probs_matrix) + (1 - adj_matrix) * edge_probs_matrix
            adj_norm = self._normalize_adj(perturbed_adj)

            # 通过代理模型和answering层
            surrogate_output = surrogate(prompted_embeddings, adj_norm)
            final_output = answering(surrogate_output)

            # 计算损失
            if hasattr(batch, 'y'):
                criterion = nn.CrossEntropyLoss()
                loss = criterion(final_output, batch.y)
            else:
                print("警告: batch缺少标签信息")
                continue
                    
            # --- 拉格朗日预算约束 ---
            num_edges = edge_index.size(1)
            budget = budget_ratio * num_edges  # 允许攻击的期望边数
            g = flip_probs_tensor.sum() - budget   # 约束项：sum(p) - B
            budget_loss = mu * g                   # 拉格朗日项 μ * g

            # --- 总损失 ---
            loss = loss + budget_loss
            loss.backward()
            optimizer.step()
            # --- 更新拉格朗日乘子 ---
            with torch.no_grad():
                mu = torch.clamp(mu + mu_lr * g, min=0.0)
            total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(train_graphs) if len(train_graphs) > 0 else 0.0
        
        def load_best_prompt(self):
            """加载最佳prompt状态"""
            if hasattr(self, 'best_prompt_state') and self.best_prompt_state is not None:
                self.p_list.data = self.best_prompt_state['p_list'].clone()
                self.a.weight.data = self.best_prompt_state['a_weight'].clone()
                if self.best_prompt_state['a_bias'] is not None:
                    self.a.bias.data = self.best_prompt_state['a_bias'].clone()
                
                if self.projection is not None and 'projection_weight' in self.best_prompt_state:
                    self.projection.weight.data = self.best_prompt_state['projection_weight'].clone()
                    if self.best_prompt_state['projection_bias'] is not None:
                        self.projection.bias.data = self.best_prompt_state['projection_bias'].clone()
                print(f"加载最佳prompt，最佳损失: {self.best_loss:.4f}")
            else:
                print("警告: 没有找到最佳prompt状态")
        
        def reset_best_prompt(self):
            """重置最佳prompt记录"""
            self.best_loss = float('inf')
            self.best_prompt_state = None
            
        # 保存最佳prompt状态
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            # 保存所有prompt相关参数
            self.best_prompt_state = {
                'p_list': self.p_list.clone().detach(),
                'a_weight': self.a.weight.clone().detach(),
                'a_bias': self.a.bias.clone().detach() if self.a.bias is not None else None
            }
            if self.projection is not None:
                self.best_prompt_state['projection_weight'] = self.projection.weight.clone().detach()
                self.best_prompt_state['projection_bias'] = self.projection.bias.clone().detach() if self.projection.bias is not None else None
            print(f"保存最佳prompt，损失: {avg_loss:.4f}")

        return avg_loss
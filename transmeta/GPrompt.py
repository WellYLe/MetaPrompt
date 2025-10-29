import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot

class Gprompt(torch.nn.Module):
    """
    Gprompt: 对每个通道做可学习的缩放（feature-wise gating / multiplicative prompt）。
    weight: 形状 [1, input_dim]，在forward时对节点嵌入按通道逐元素相乘。
    """
    def __init__(self, input_dim):
        super(Gprompt, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.max_n_num = input_dim
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, node_embeddings):
        """
        node_embeddings: [num_nodes, input_dim]
        返回: node_embeddings * weight（按通道缩放）
        """
        # 支持 numpy / 不同设备的输入
        if isinstance(node_embeddings, (list, tuple)):
            node_embeddings = torch.stack(node_embeddings, dim=0)
        if not isinstance(node_embeddings, torch.Tensor):
            node_embeddings = torch.tensor(node_embeddings, dtype=torch.float32, device=self.weight.device)

        if node_embeddings.device != self.weight.device:
            node_embeddings = node_embeddings.to(self.weight.device)

        # broadcasting: [num_nodes, input_dim] * [1, input_dim] -> [num_nodes, input_dim]
        return node_embeddings * self.weight


    def _normalize_adj(self, adj):
        """与之前相同的对称归一化"""
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
        D = torch.sum(adj, dim=1)
        D_inv = torch.pow(D, -0.5)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)
        adj_norm = D_mat_inv @ adj @ D_mat_inv
        return adj_norm


    def train(self, train_graphs, attacker, surrogate, answering, optimizer, device='cuda', l2_reg=1e-4,
              budget_ratio=0.05,     # 预算比例，例如5%边可攻击
              mu_init=0.0,           # 初始乘子
              mu_lr=1e-2):  
        """
        与GPF / GPF_plus一致的训练接口：
        train_graphs: iterable of Data objects (PyG)
        attacker: 需要实现 predict_all_edges(graph_data) 和 encoder(features, edge_index)
        surrogate: 接受 (node_embeddings, adj_norm) -> logits
        answering: 将surrogate输出映射成最终分类logits（可是identity）
        optimizer: 优化器，包含本模块参数及其他可学习参数
        device: 'cuda' or 'cpu'
        l2_reg: 对 weight 做小的 L2 正则（可选）
        """
        total_loss = 0.0
        n_batches = 0
        mu = torch.tensor(mu_init, device=device)  # 拉格朗日乘子 (非负标量)    

        for batch in train_graphs:
            optimizer.zero_grad()
            batch = batch.to(device)

            if not hasattr(batch, 'x') or not hasattr(batch, 'edge_index'):
                print("警告: batch缺少必要的图数据属性")
                continue

            feature = batch.x
            edge_index = batch.edge_index

            # 1) 预测边翻转概率
            graph_data = Data(x=feature, edge_index=edge_index)
            flip_probs = attacker.predict_all_edges(graph_data)  # 应返回与edge_index长度一致的概率向量

            # 2) 编码器获取节点嵌入
            node_embeddings = attacker.encoder(feature, edge_index)  # [num_nodes, embed_dim]

            # 3) 使用 Gprompt（逐通道缩放）
            prompted_embeddings = self.forward(node_embeddings)

            # 4) 构造软扰动邻接矩阵（保持可微）
            num_nodes = feature.shape[0]
            adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
            adj_matrix[edge_index[0], edge_index[1]] = 1.0

            flip_probs_tensor = torch.tensor(flip_probs, device=device)
            edge_probs_matrix = torch.zeros_like(adj_matrix)
            edge_probs_matrix[edge_index[0], edge_index[1]] = flip_probs_tensor

            perturbed_adj = adj_matrix * (1 - edge_probs_matrix) + (1 - adj_matrix) * edge_probs_matrix
            adj_norm = self._normalize_adj(perturbed_adj)

            # 5) 通过代理模型和 answering 层
            surrogate_output = surrogate(prompted_embeddings, adj_norm)
            final_output = answering(surrogate_output)

            # 6) 损失计算（交叉熵）
            if hasattr(batch, 'y'):
                criterion = nn.CrossEntropyLoss()
                loss = criterion(final_output, batch.y)
            else:
                print("警告: batch缺少标签信息")
                continue

            # 7) 可选正则：避免weight尺度爆炸或收敛到0/很大值
            reg_loss = l2_reg * torch.sum(self.weight ** 2)
            total_batch_loss = loss + reg_loss
            
            # 8) 拉格朗日预算约束
            num_edges = edge_index.size(1)
            budget = budget_ratio * num_edges  # 允许攻击的期望边数
            g = flip_probs_tensor.sum() - budget   # 约束项：sum(p) - B
            budget_loss = mu * g                   # 拉格朗日项 μ * g
            total_batch_loss += budget_loss
            
            total_batch_loss.backward()
            optimizer.step()
            
            # 9) 更新拉格朗日乘子
            with torch.no_grad():
                mu = torch.clamp(mu + mu_lr * g, min=0.0)
            
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

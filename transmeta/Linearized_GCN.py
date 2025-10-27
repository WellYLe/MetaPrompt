import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import sys
# 修正模块搜索路径，确保能导入 ProG 下的 prompt_graph
root_dir = os.path.dirname(os.path.dirname(__file__))
prog_path = os.path.join(root_dir, 'ProG')
if prog_path not in sys.path:
    sys.path.insert(0, prog_path)

from prompt_graph.data import load4node, load4graph
from prompt_graph.utils import act
from utils.edge_index_to_adjacency_matrix import edge_index_to_adjacency_matrix
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention, global_mean_pool
from torch_geometric.nn.inits import glorot

# 使用方式见Test()

class GCNConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GCNConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))       #这构建了一个要求梯度的 参数矩阵
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, out_features))               #偏置项
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)                           
        if self.bias is not None:                       
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    


class Linearized_GCN(torch.nn.Module):    
    def __init__(self, input_dim, hid_dim, out_dim=None, num_layer=2, bias=False):
        super().__init__()

        GraphConv = GCNConvolution
        
        assert 1<num_layer, "please set num_layer>1"

        if out_dim is None:
            out_dim = hid_dim
        
        if num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim, bias), GraphConv(hid_dim, out_dim, bias)])
        else:
            layers = [GraphConv(input_dim, hid_dim, bias)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim, bias))
            layers.append(GraphConv(hid_dim, out_dim, bias))
            self.conv_layers = torch.nn.ModuleList(layers)

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()

    def forward(self, x, adj_norm):
        for conv in self.conv_layers:
            x = conv(x, adj_norm)
        return x
    
    def train(self, 
                   dataset_name='Cora', 
                   task_type='node',  # 'node' 或 'graph'
                   learning_rate=0.05, 
                   weight_decay=1e-4, 
                   epochs=100, 
                   device=None,
                   verbose=True,
                   early_stopping=False,
                   patience=10,
                   min_delta=1e-4):
        """
        训练 Linearized_GCN 模型
        
        Args:
            dataset_name: 数据集名称
            task_type: 任务类型 ('node' 为节点分类, 'graph' 为图分类)
            learning_rate: 学习率
            weight_decay: 权重衰减
            epochs: 训练轮数
            device: 设备 ('cpu', 'cuda', 或 None 自动选择)
            verbose: 是否打印训练过程
            early_stopping: 是否启用早停
            patience: 早停耐心值
            min_delta: 早停最小改善值
            
        Returns:
            Dict: 包含训练结果的字典
        """
        import torch.optim as optim
        
        # 设备选择
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        self.to(device)
        
        if verbose:
            print(f"使用设备: {device}")
            print(f"任务类型: {'节点分类' if task_type == 'node' else '图分类'}")
            print(f"训练数据集: {dataset_name}")
        
        try:
            if task_type == 'node':
                return self._train_node_classification(
                    dataset_name, learning_rate, weight_decay, epochs, 
                    device, verbose, early_stopping, patience, min_delta
                )
            elif task_type == 'graph':
                return self._train_graph_classification(
                    dataset_name, learning_rate, weight_decay, epochs, 
                    device, verbose, early_stopping, patience, min_delta
                )
            else:
                raise ValueError("task_type 必须是 'node' 或 'graph'")
                
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            raise
    
    def _train_node_classification(self, dataset_name, learning_rate, weight_decay, 
                                 epochs, device, verbose, early_stopping, patience, min_delta):
        """节点分类训练"""
        # 加载数据
        data, feature_dim, out_dim = load4node(dataset_name)
        x = torch.FloatTensor(data.x).to(device)
        y = data.y.to(device)
        edge_index = data.edge_index
        
        if verbose:
            print(f"数据集信息:")
            print(f"  - 节点数: {x.shape[0]}")
            print(f"  - 特征维度: {feature_dim}")
            print(f"  - 类别数: {out_dim}")
            print(f"  - 边数: {edge_index.shape[1]}")
        
        # 构建归一化邻接矩阵
        adj = edge_index_to_adjacency_matrix(edge_index, x.shape[0])
        adj_ = adj + torch.eye(adj.shape[0])
        D = torch.sum(adj_, dim=1)
        D_inv = torch.pow(D, -1/2)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)
        adj_norm = D_mat_inv @ adj_ @ D_mat_inv
        adj_norm = adj_norm.to(device)
        
        # 创建优化器
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if verbose:
            print(f"模型参数数量: {sum(p.numel() for p in self.parameters())}")
            print("开始训练...")
        
        # 训练记录
        train_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        # 训练循环
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            
            # 前向传播
            output = F.log_softmax(self(x, adj_norm), dim=1)
            loss = F.nll_loss(output, y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # 打印训练信息
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs}, Loss: {loss.item():.6f}")
            
            # 早停检查
            if early_stopping:
                if loss.item() < best_loss - min_delta:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    if verbose:
                        print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
        
        # 计算最终准确率
        self.eval()
        with torch.no_grad():
            output = self(x, adj_norm)
            pred = output.argmax(dim=1)
            accuracy = (pred == y).float().mean().item()
        
        if verbose:
            print(f"训练完成!")
            print(f"最终损失: {train_losses[-1]:.6f}")
            print(f"训练准确率: {accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'final_loss': train_losses[-1],
            'accuracy': accuracy,
            'epochs_trained': len(train_losses),
            'task_type': 'node_classification',
            'dataset_name': dataset_name
        }
    
    def _train_graph_classification(self, dataset_name, learning_rate, weight_decay, 
                                  epochs, device, verbose, early_stopping, patience, min_delta):
        """图分类训练"""
        from torch_geometric.loader import DataLoader
        
        # 加载数据
        feature_dim, out_dim, dataset = load4graph(dataset_name)
        
        if verbose:
            print(f"数据集信息:")
            print(f"  - 图数量: {len(dataset)}")
            print(f"  - 特征维度: {feature_dim}")
            print(f"  - 类别数: {out_dim}")
        
        # 数据分割 (80% 训练, 20% 测试)
        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 创建优化器
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if verbose:
            print(f"训练集大小: {len(train_dataset)}")
            print(f"测试集大小: {len(test_dataset)}")
            print(f"模型参数数量: {sum(p.numel() for p in self.parameters())}")
            print("开始训练...")
        
        # 训练记录
        train_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        # 训练循环
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # 构建批次邻接矩阵
                adj_norm = self._build_batch_adj_norm(batch, device)
                
                # 前向传播
                output = self(batch.x, adj_norm)
                
                # 图级别池化 (简单平均池化)
                batch_size = batch.num_graphs
                node_counts = torch.bincount(batch.batch, minlength=batch_size)
                graph_embeddings = torch.zeros(batch_size, output.size(1)).to(device)
                
                for i in range(batch_size):
                    mask = (batch.batch == i)
                    graph_embeddings[i] = output[mask].mean(dim=0)
                
                # 计算损失
                loss = F.cross_entropy(graph_embeddings, batch.y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # 打印训练信息
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs}, Loss: {avg_loss:.6f}")
            
            # 早停检查
            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    if verbose:
                        print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
        
        # 计算测试准确率
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                adj_norm = self._build_batch_adj_norm(batch, device)
                output = self(batch.x, adj_norm)
                
                # 图级别池化
                batch_size = batch.num_graphs
                graph_embeddings = torch.zeros(batch_size, output.size(1)).to(device)
                
                for i in range(batch_size):
                    mask = (batch.batch == i)
                    graph_embeddings[i] = output[mask].mean(dim=0)
                
                pred = graph_embeddings.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        accuracy = correct / total
        
        if verbose:
            print(f"训练完成!")
            print(f"最终损失: {train_losses[-1]:.6f}")
            print(f"测试准确率: {accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'final_loss': train_losses[-1],
            'accuracy': accuracy,
            'epochs_trained': len(train_losses),
            'task_type': 'graph_classification',
            'dataset_name': dataset_name
        }
    
    def _build_batch_adj_norm(self, batch, device):
        """为批次数据构建归一化邻接矩阵"""
        num_nodes = batch.x.size(0)
        adj = torch.zeros(num_nodes, num_nodes).to(device)
        
        # 填充邻接矩阵
        edge_index = batch.edge_index
        adj[edge_index[0], edge_index[1]] = 1.0
        
        # 归一化
        adj_ = adj + torch.eye(num_nodes).to(device)
        D = torch.sum(adj_, dim=1)
        D_inv = torch.pow(D, -1/2)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)
        adj_norm = D_mat_inv @ adj_ @ D_mat_inv
        
        return adj_norm
    
    def predict(self, x, adj_norm, task_type='node', batch=None):
        """
        使用训练好的模型进行预测
        
        Args:
            x: 节点特征
            adj_norm: 归一化邻接矩阵
            task_type: 任务类型 ('node' 或 'graph')
            batch: 图分类时的批次信息
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            output = self(x, adj_norm)
            
            if task_type == 'node':
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                return preds, probs, output
            
            elif task_type == 'graph':
                if batch is None:
                    raise ValueError("图分类预测需要提供 batch 信息")
                
                # 图级别池化
                batch_size = batch.max().item() + 1
                graph_embeddings = torch.zeros(batch_size, output.size(1)).to(output.device)
                
                for i in range(batch_size):
                    mask = (batch == i)
                    graph_embeddings[i] = output[mask].mean(dim=0)
                
                probs = F.softmax(graph_embeddings, dim=1)
                preds = graph_embeddings.argmax(dim=1)
                return preds, probs, graph_embeddings
    
    

def Test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, feature_dim, out_dim = load4node('Cora')
    x = data.x
    y = data.y
    edge_index = data.edge_index

    x = torch.FloatTensor(data.x).to(device)
    y = y.to(device)

    adj = edge_index_to_adjacency_matrix(edge_index, x.shape[0])

    adj_ = adj + torch.eye(adj.shape[0])
    D = torch.sum(adj_, dim=1)
    D_inv = torch.pow(D, -1/2)
    D_inv[torch.isinf(D_inv)] = 0.
    D_mat_inv = torch.diag(D_inv)

    adj_norm = D_mat_inv @ adj_ @ D_mat_inv   # GCN的归一化方式
    adj_norm = adj_norm.to(device)

    model = Linearized_GCN(feature_dim, hid_dim=128, out_dim=out_dim).to(device)
    #print(model.conv_layers[0].weight, model.conv_layers[1].weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)

    for i in range(100):
        model.train()
        optimizer.zero_grad()
        output = F.log_softmax(model(x, adj_norm))
        loss = F.nll_loss(output, y)
        loss.backward()
        print("loss is:", loss)
        optimizer.step()
    

if __name__ == '__main__':
    Test()
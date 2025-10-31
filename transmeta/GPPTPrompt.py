import torch
from sklearn.cluster import KMeans
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class SimpleMeanConv(MessagePassing):
    def __init__(self):
        # 初始化时指定聚合方式为 'mean'，即平均聚合
        super(SimpleMeanConv, self).__init__(aggr='mean')  # 'mean'聚合。

    def forward(self, x, edge_index):
        # x 代表节点特征矩阵，edge_index 是图的边索引列表

        # 在边索引中添加自环，这样在聚合时，节点也会考虑自己的特征
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 开始消息传递过程，其中x是每个节点的特征，edge_index定义了节点间的连接关系
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j 表示邻居节点的特征，这里直接返回，因为我们使用的是 'mean' 聚合
        return x_j
    
# def perform_kmeans(h, num_clusters, device):
#     # gpu kmeans
#     cluster_ids, cluster_centers = kmeans(X=h, num_clusters=num_clusters, distance='euclidean', device=device)
#     return cluster_centers

class GPPTPrompt(torch.nn.Module):
    def __init__(self, n_hidden, center_num, n_classes, device):
        super(GPPTPrompt, self).__init__()
        self.center_num = center_num
        self.n_classes = n_classes
        self.device = device
        self.StructureToken = torch.nn.Linear(n_hidden, center_num, bias=False)
        self.StructureToken=self.StructureToken.to(device)  # structure token
        self.TaskToken = torch.nn.ModuleList()
        for i in range(center_num):
            self.TaskToken.append(torch.nn.Linear(2 * n_hidden, n_classes, bias=False))  #task token
        self.TaskToken = self.TaskToken.to(device)

    def _initialize_weights(self, layer):
        import torch.nn.init as init
        if isinstance(layer, nn.Linear):
            # You can choose any initialization method. Here, we use Xavier initialization.
            init.xavier_uniform_(layer.weight)
            # If you have bias, you can initialize it as well, but in this case, bias is False.
            # if layer.bias is not None:
            #     init.constant_(layer.bias, 0)        

    def weigth_init(self, h, edge_index, label, index):
        # 对于图中的每一个节点，将其特征（'h'）发送给所有邻居节点，然后每个节点会计算所有收到的邻居特征的平均值，并将这个平均值存储为自己的新特征在'neighbor'下

        conv = SimpleMeanConv()
        # 使用这个层进行前向传播，得到聚合后的节点特征
        h = conv(h, edge_index)
        
        features=h[index]
        labels=label[index.long()]  # labels变量的类别不全

        cluster = KMeans(n_clusters=self.center_num,random_state=0).fit(features.detach().cpu())
        temp=torch.FloatTensor(cluster.cluster_centers_).to(self.device)
        self.StructureToken.weight.data = temp.clone().detach()

        p=[]
        for i in range(self.n_classes):
            p.append(features[labels==i].mean(dim=0).view(1,-1))
        temp=torch.cat(p,dim=0).to(self.device)
        for i in range(self.center_num):
            self.TaskToken[i].weight.data = temp.clone().detach()
            # enzymes 600张图。1-shot，6种节点类型，筛选出6张图。do.. while() --> 6张图的Batch bg，bg.y,
    
    def update_StructureToken_weight(self, h):

        if h.size(0)>20000:
            cluster_ids_x, cluster_centers = kmeans(X=h, num_clusters=self.center_num, distance='euclidean', device=self.device)
            self.StructureToken.weight.data = cluster_centers.clone()
        else:
            cluster = KMeans(n_clusters=self.center_num,random_state=0).fit(h.detach().cpu())
            temp = torch.FloatTensor(cluster.cluster_centers_).to(self.device)
            self.StructureToken.weight.data = temp.clone()

    def get_TaskToken(self):
        pros=[]
        for name,param in self.named_parameters():
            if name.startswith('TaskToken.'):
                pros.append(param)
        return pros
        
    def get_StructureToken(self):
        for name,param in self.named_parameters():
            if name.startswith('StructureToken.weight'):
                pro=param
        return pro
    
    def get_mid_h(self):
        return self.fea
    
    def load_best_prompt(self):
        """加载最佳prompt状态"""
        if hasattr(self, 'best_prompt_state') and self.best_prompt_state is not None:
            self.StructureToken.weight.data = self.best_prompt_state['structure_weight'].clone()
            for i, task_token in enumerate(self.TaskToken):
                task_token.weight.data = self.best_prompt_state['task_weights'][i].clone()
            print(f"加载最佳prompt，最佳损失: {self.best_loss:.4f}")
        else:
            print("警告: 没有找到最佳prompt状态")
    
    def reset_best_prompt(self):
        """重置最佳prompt记录"""
        self.best_loss = float('inf')
        self.best_prompt_state = None

    def forward(self, h, edge_index):       
        device = h.device
        conv = SimpleMeanConv()
        # 使用这个层进行前向传播，得到聚合后的节点特征
        h = conv(h, edge_index)
        self.fea = h 
        out = self.StructureToken(h)
        index = torch.argmax(out, dim=1)
        out = torch.zeros(h.shape[0],self.n_classes).to(device)
        for i in range(self.center_num):
            out[index==i]=self.TaskToken[i](h[index==i])
        return out
    

def kmeans(X, num_clusters, distance='euclidean', device='cuda', max_iter=100, tol=1e-4):
    """
    Perform KMeans clustering on the input data X.

    Parameters:
    X : torch.Tensor
        Input data, shape [n_samples, n_features]
    num_clusters : int
        Number of clusters
    distance : str
        Distance metric ('euclidean' is currently supported)
    device : str
        Device to use ('cuda' or 'cpu')
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence

    Returns:
    cluster_ids_x : torch.Tensor
        Cluster assignment for each sample
    cluster_centers : torch.Tensor
        Cluster centers
    """

    if distance != 'euclidean':
        raise NotImplementedError("Currently only 'euclidean' distance is supported.")

    X = X.to(device)
    n_samples, n_features = X.shape

    # Randomly initialize cluster centers
    random_indices = torch.randperm(n_samples)[:num_clusters]
    cluster_centers = X[random_indices]

    for i in range(max_iter):
        # Compute distances and assign clusters
        distances = torch.cdist(X, cluster_centers)
        cluster_ids_x = torch.argmin(distances, dim=1)

        # Compute new cluster centers
        new_cluster_centers = torch.zeros_like(cluster_centers)
        for k in range(num_clusters):
            cluster_k = X[cluster_ids_x == k]
            if len(cluster_k) > 0:
                new_cluster_centers[k] = cluster_k.mean(dim=0)

        # Check for convergence
        if torch.norm(new_cluster_centers - cluster_centers) < tol:
            break

        cluster_centers = new_cluster_centers

    return cluster_ids_x, cluster_centers

# # Example usage
# h = torch.randn(160000, 128).to('cuda')
# cluster_ids_x, cluster_centers = kmeans(X=h, num_clusters=10, distance='euclidean', device='cuda')
    def train(self, train_graphs, attacker, surrogate, answering, optimizer, device='cuda',
              budget_ratio=0.05,     # 预算比例，例如5%边可攻击
              mu_init=0.0,           # 初始乘子
              mu_lr=1e-2):
        total_loss = 0.0
        mu = torch.tensor(mu_init, device=device)  # 拉格朗日乘子 (非负标量)  
        for batch in train_graphs:
            optimizer.zero_grad()
            batch = batch.to(device)
    
            feature, edge_index = batch.x, batch.edge_index
            graph_data = Data(x=feature, edge_index=edge_index)
    
            # 预测扰动概率
            flip_probs = attacker.predict_all_edges(graph_data)
    
            # 获得节点嵌入
            node_emb = attacker.encoder(feature, edge_index)
    
            # GPPT动态提示传播
            prompted_emb = self.prompt_transformer(node_emb, edge_index)
    
            # 软扰动邻接矩阵
            num_nodes = feature.shape[0]
            adj = torch.zeros(num_nodes, num_nodes, device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            flip_probs_tensor = torch.tensor(flip_probs, device=device)
            edge_probs_matrix = torch.zeros_like(adj)
            edge_probs_matrix[edge_index[0], edge_index[1]] = flip_probs_tensor
            perturbed_adj = adj * (1 - edge_probs_matrix) + (1 - adj) * edge_probs_matrix
            adj_norm = self._normalize_adj(perturbed_adj)
    
            # 代理模型预测 + 答案生成
            surrogate_out = surrogate(prompted_emb, adj_norm)
            final_out = answering(surrogate_out)
    
            # 损失计算
            if hasattr(batch, 'y'):
                criterion = nn.CrossEntropyLoss()
                loss = criterion(final_out, batch.y)
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
            else:
                print("Warning: missing labels, skip batch")
                continue
    
        # 计算平均损失
        avg_loss = total_loss / len(train_graphs) if len(train_graphs) > 0 else 0.0
        
        # 保存最佳prompt状态
        if not hasattr(self, 'best_loss') or avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_prompt_state = {
                'structure_weight': self.StructureToken.weight.clone().detach(),
                'task_weights': [task_token.weight.clone().detach() for task_token in self.TaskToken]
            }
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

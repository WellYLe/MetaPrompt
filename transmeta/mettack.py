"""
    Adversarial Attacks on Graph Neural Networks via Meta Learning. ICLR 2019
        https://openreview.net/pdf?id=Bylnx209YX
    Author Tensorflow implementation:
        https://github.com/danielzuegner/gnn-meta-attack
"""
import sys
import os
from DeepRobust.examples.graph.test_visualization import clean_adj
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

# 设置DeepRobust路径并添加到Python路径中
# REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeepRobust'))
# sys.path.insert(0, REPO_ROOT)

# 现在可以正确导入deeprobust包
from DeepRobust.deeprobust.graph import utils
from DeepRobust.deeprobust.graph.global_attack import BaseAttack
from DeepRobust.deeprobust.graph.data import Dataset
from edge_flip_mae_example import load_and_predict_example
from torch_geometric.utils import degree

from DeepRobust.deeprobust.graph.defense.gcn import GCN
from EdgeFlipMAE import EdgeFlipMAE


class BaseMeta(BaseAttack):
    """元攻击抽象基类. Adversarial Attacks on Graph Neural
    Networks via Meta Learning, ICLR 2019,
    https://openreview.net/pdf?id=Bylnx209YX

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    lambda_ : float
        lambda_ is used to weight the two objectives in Eq. (10) in the paper.
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    undirected : bool
        whether the graph is undirected
    device: str
        'cpu' or 'cuda'



    """

    def __init__(self, model=None, nnodes=None, feature_shape=None, lambda_=0.5, attack_structure=True, attack_features=False, undirected=True, device='cpu'):

        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            self.undirected = undirected
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert feature_shape is not None, 'Please give feature_shape='
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

        self.with_relu = model.with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def get_modified_adj(self, ori_adj):#不动，依旧如此，因为这里只是接受安排，然后给个结果
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        # ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj
    


    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

    def filter_potential_singletons(self, modified_adj):
        """
        计算可能导致单节点的条目掩码,即对应于条目的两个节点之一的度为1,并且这两个节点之间有一条边。
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask

    def self_training_label(self, labels, idx_train):
        # 预测未标记节点的标签，用于自我训练。
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        计算导致对数似然约束被违反的条目的掩码。

        注意，不同的数据类型（float、double）可能会影响最终结果。
        """
        t_d_min = torch.tensor(2.0).to(self.device)
        if self.undirected:
            t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        else:
            t_possible_edges = np.array((np.ones((self.nnodes, self.nnodes)) - np.eye(self.nnodes)).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff, undirected=self.undirected)
        return allowed_mask, current_ratio



    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad
class GPF(torch.nn.Module):
        def __init__(self, in_channels: int):
            super(GPF, self).__init__()
            self.global_emb = torch.nn.Parameter(torch.Tensor(1,in_channels))
            self.reset_parameters()

        def reset_parameters(self):
            glorot(self.global_emb)

        def add(self, x: torch.Tensor):
            return x + self.global_emb
        
        def GPFTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            
            for batch in train_loader:  
                  self.optimizer.zero_grad() 
                  batch = batch.to(self.device)
                  batch.x = self.prompt.add(batch.x)
                  out = self.gnntemp(self.weights,self.biases, batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
                  out = self.answering(out)
                  loss = self.criterion(out, batch.y)  
                  loss.backward()  
                  self.optimizer.step()  
                  total_loss += loss.item()  
            return total_loss / len(train_loader) 
        
############TODO####################################################
        def gnntemp(self, weights, biases, x, edge_index, batch, prompt = None, prompt_type = None):
            for ix, w in enumerate(weights):
                b = biases[ix] if self.with_bias else 0
                x = self.gnnlayer(x, edge_index, w, b)
                if self.with_relu:
                    x = F.relu(x)
            return x
############TODO#####################################################

class Metattack(BaseMeta):
    """Meta attack. Adversarial Attacks on Graph Neural Networks
    via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import Metattack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, 
                 device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9,attacker_encoder=None,attacker_classifier=None,
                 prompt_type = None):

        super(Metattack, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum# 动量因子
        self.lr = lr
        self.train_iters = train_iters# 训练迭代次数
        self.with_bias = with_bias# 是否包含偏置项

        self.weights = []# 模型权重列表
        self.biases = []# 模型偏置项列表
        self.w_velocities = []# 权重动量列表
        self.b_velocities = []# 偏置项动量列表

        self.hidden_sizes = self.surrogate.hidden_sizes# 隐藏层大小列表
        self.nfeat = self.surrogate.nfeat# 输入特征维度
        self.nclass = self.surrogate.nclass# 输出类别数
        self.model = model# 受害者模型
        self.attacker_encoder = attacker_encoder# 攻击模型编码器
        self.attacker_classifier = attacker_classifier# 攻击模型分类器
        self.prompt_type = prompt_type# prompt类型

        previous_size = self.nfeat# 上一层的特征维度
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))# 隐藏层权重
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)# 隐藏层权重
            self.w_velocities.append(w_velocity)

            if self.with_bias:# 是否包含偏置项
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid# 当前层的特征维度

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def initialize_optimizer(self):
            if self.prompt_type == 'None':
                if self.pre_train_model_path == 'None':
                    model_param_group = []
                    model_param_group.append({"params": self.gnn.parameters()})
                    model_param_group.append({"params": self.answering.parameters()})
                    self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
                else:
                    model_param_group = []
                    model_param_group.append({"params": self.gnn.parameters()})
                    model_param_group.append({"params": self.answering.parameters()})
                    self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
                    # self.optimizer = optim.Adam(self.answering.parameters(), lr=self.lr, weight_decay=self.wd)

            elif self.prompt_type == 'All-in-one':
                self.pg_opi = optim.Adam( self.prompt.parameters(), lr=1e-6, weight_decay= self.wd)
                self.answer_opi = optim.Adam( self.answering.parameters(), lr=self.lr, weight_decay= self.wd)
            elif self.prompt_type in ['GPF', 'GPF-plus']:
                model_param_group = []
                model_param_group.append({"params": self.prompt.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
            elif self.prompt_type in ['Gprompt']:
                self.pg_opi = optim.Adam(self.prompt.parameters(), lr=self.lr, weight_decay=self.wd)
            elif self.prompt_type in ['GPPT']:
                self.pg_opi = optim.Adam(self.prompt.parameters(), lr=2e-3, weight_decay=5e-4)
            elif self.prompt_type == 'MultiGprompt':
                self.optimizer = optim.Adam([*self.DownPrompt.parameters(),*self.feature_prompt.parameters()], lr=self.lr)

    def pretrainGNNGPL(self):
        self.prompt = GPF(self.input_dim).to(self.device)#初始化prompt
        out = self.prompt.add(self.modified_adj)#获得prompt层的嵌入表示
        out = load_model(out, self.gnnPath)#用节点嵌入再去查阅我预训练的输出
        return out
    

        
 
        
    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.

        """

        self.sparse_features = sp.issparse(ori_features)# 检查是否为稀疏矩阵
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)# 转换为张量
        labels_self_training = self.self_training_label(labels, idx_train)# 自训练标签
        modified_adj = ori_adj# 复制原始邻接矩阵
        modified_features = ori_features# 复制原始特征矩阵
        self.surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
            nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
        self.surrogate.fit(features, adj_norm, labels, idx_train, idx_unlabeled, patience=30)
        #这里在clean graph上训练GCN
        #加载 cora 数据集
        data = Dataset(root='/tmp/', name='cora')
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
#####################FINISH TRAINING SURROGATE MODEL################################
        #这里导入在上游训练的判断图的Attacker模型，用于生成扰动
        attacker = EdgeFlipMAE(ori_adj, ori_features, labels, idx_train, idx_unlabeled, self.device)
        attacker.load_model(self.attacker_encoder, self.attacker_classifier)
        
        # modified_adj = attacker.modified_adj# 扰动后的邻接矩阵
        # modified_features = attacker.modified_features# 扰动后的特征矩阵
        self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                    torch.nn.Softmax(dim=1)).to(self.device) 
        self.prompt = GPF(self.input_dim).to(self.device)
        self.gnn = attacker
        #为什么好像别的文件调用self.prompt的时候都不需要再init函数中声明这个变量？
        self.initialize_optimizer()
        
#####################FINISH INITIALIZING THE PROMPT################################     
   
#####################FOLLOWING BEGIN TO LOAD THE DATA################################          
        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)# 更新扰动后的邻接矩阵

            if self.attack_features:
                modified_features = ori_features + self.feature_changes
            
            adj_norm = utils.normalize_adj_tensor(modified_adj)# 归一化扰动后的邻接矩阵
            adj_grad = get_grad(self.surrogate.predict(modified_features, adj_norm, idx_train, idx_unlabeled, labels))# 用GCN模型预测并求梯度
            #这里不训练Victim，只是用预训练好的GPL来预测，而不更新GNN参数
            GPFTrain(adj_grad)#在这个里面p被更新。  
            #这里还没做，因为我还没实现prompt的训练
            #这里用Victim的反馈来求梯度，并且更新prompt参数
            
            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                #adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
                adj_meta_score = attacker.predict_graph_with_decisions_with_get_all_edges(modified_adj)['flip_probabilities']
                #返回一个矩阵，意味着此时带有提示的GNN对每条边的预测结果
                #但是接受的参数还没对齐，可能需要写一个dataloader
                
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)
            
            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)#在所有边里找最佳扰动点
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()
            
    
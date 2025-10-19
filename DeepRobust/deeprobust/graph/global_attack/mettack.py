"""
    Adversarial Attacks on Graph Neural Networks via Meta Learning. ICLR 2019
        https://openreview.net/pdf?id=Bylnx209YX
    Author Tensorflow implementation:
        https://github.com/danielzuegner/gnn-meta-attack
"""

import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack
import sys
from torch_geometric.utils import degree


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

    def get_modified_adj(self, ori_adj):
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
        计算可能导致单节点的条目掩码，即对应于条目的两个节点之一的度为1，并且这两个节点之间有一条边。
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
        # Predict the labels of the unlabeled nodes to use them for self-training.
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

    def get_adj_score(self, adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff, node_loss_weight=None):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_meta_grad = adj_meta_grad - adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad = adj_meta_grad - torch.diag(torch.diag(adj_meta_grad, 0))
        # # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad *  singleton_mask

        # 可选：按节点损失加权分数，将节点损失映射为边对权重
        if node_loss_weight is not None:
            # Normalize to [0,1] to keep scale reasonable
            nlw = node_loss_weight
            nlw = nlw - nlw.min()
            denom = nlw.max() + 1e-12
            nlw = nlw / denom if denom > 0 else nlw
            # Build pair weight: average of endpoint weights
            pair_weight = (nlw.view(-1, 1) + nlw.view(1, -1)) * 0.5
            adj_meta_grad = adj_meta_grad * pair_weight
            # Ensure self-loops stay zero
            adj_meta_grad = adj_meta_grad - torch.diag(torch.diag(adj_meta_grad, 0))

        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
            allowed_mask = allowed_mask.to(self.device)
            adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad


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

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):

        super(Metattack, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

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

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

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

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad

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

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
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
class MetaApprox(BaseMeta):
    """Approximated version of Meta Attack. Adversarial Attacks on
    Graph Neural Networks via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import MetaApprox
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = MetaApprox(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=True)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.01):

        super(MetaApprox, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

        self.lr = lr
        self.train_iters = train_iters
        self.adj_meta_grad = None
        self.features_meta_grad = None
        if self.attack_structure:
            self.adj_grad_sum = torch.zeros(nnodes, nnodes).to(device)
        if self.attack_features:
            self.feature_grad_sum = torch.zeros(feature_shape).to(device)

        self.with_bias = with_bias

        self.weights = []
        self.biases = []

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
        self.weights.append(output_weight)
        self.biases.append(output_bias)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=lr) # , weight_decay=5e-4)
        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            # w.data.fill_(1)
            # b.data.fill_(1)
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr)

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        for j in range(self.train_iters):
            # hidden = features
            # for w, b in zip(self.weights, self.biases):
            #     if self.sparse_features:
            #         hidden = adj_norm @ torch.spmm(hidden, w) + b
            #     else:
            #         hidden = adj_norm @ hidden @ w + b
            #     if self.with_relu:
            #         hidden = F.relu(hidden)

            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)

            if self.attack_structure:
                self.adj_changes.grad.zero_()
                self.adj_grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            if self.attack_features:
                self.feature_changes.grad.zero_()
                self.feature_grad_sum += torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

            self.optimizer.step()


        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))


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
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        self.sparse_features = sp.issparse(ori_features)
        modified_adj = ori_adj
        modified_features = ori_features

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            self._initialize()

            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
                self.adj_grad_sum.data.fill_(0)
            if self.attack_features:
                modified_features = ori_features + self.feature_changes
                self.feature_grad_sum.data.fill_(0)

            self.inner_train(modified_features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)

            if self.attack_structure:
                adj_meta_score = self.get_adj_score(self.adj_grad_sum, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(self.feature_grad_sum, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
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
# class MetaEva(BaseMeta):
#     """Approximated version of  Meta Attack. Adversarial Attacks on
#     Graph Neural Networks via Meta Learning, ICLR 2019.

#     Examples
#     --------

#     >>> import numpy as np
#     >>> from deeprobust.graph.data import Dataset
#     >>> from deeprobust.graph.defense import GCN
#     >>> from deeprobust.graph.global_attack import MetaEva
#     >>> from deeprobust.graph.utils import preprocess
#     >>> data = Dataset(root='/tmp/', name='cora')
#     >>> adj, features, labels = data.adj, data.features, data.labels
#     >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
#     >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
#     >>> idx_unlabeled = np.union1d(idx_val, idx_test)
#     >>> # Setup Surrogate model
#     >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
#                     nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
#     >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
#     >>> # Setup Attack Model
#     >>> model = MetaApprox(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
#             attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
#     >>> # Attack
#     >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=True)
#     >>> modified_adj = model.modified_adj

#     """

#     def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.01):

#         super(MetaEva, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

#         self.lr = lr
#         self.train_iters = train_iters
#         self.adj_meta_grad = None
#         self.features_meta_grad = None
#         if self.attack_structure:
#             self.adj_grad_sum = torch.zeros(nnodes, nnodes).to(device)
#         if self.attack_features:
#             self.feature_grad_sum = torch.zeros(feature_shape).to(device)

#         self.with_bias = with_bias

#         self.weights = []
#         self.biases = []

#         previous_size = self.nfeat
#         for ix, nhid in enumerate(self.hidden_sizes):
#             weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
#             bias = Parameter(torch.FloatTensor(nhid).to(device))
#             previous_size = nhid

#             self.weights.append(weight)
#             self.biases.append(bias)

#         output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
#         output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
#         self.weights.append(output_weight)
#         self.biases.append(output_bias)

#         self.optimizer = optim.Adam(self.weights + self.biases, lr=lr) # , weight_decay=5e-4)
#         self._initialize()

#     def _initialize(self):
#         for w, b in zip(self.weights, self.biases):
#             # w.data.fill_(1)
#             # b.data.fill_(1)
#             stdv = 1. / math.sqrt(w.size(1))
#             w.data.uniform_(-stdv, stdv)
#             b.data.uniform_(-stdv, stdv)

#         self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr)

#     def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
#         adj_norm = utils.normalize_adj_tensor(modified_adj)
#         for j in range(self.train_iters):
#             # hidden = features
#             # for w, b in zip(self.weights, self.biases):
#             #     if self.sparse_features:
#             #         hidden = adj_norm @ torch.spmm(hidden, w) + b
#             #     else:
#             #         hidden = adj_norm @ hidden @ w + b
#             #     if self.with_relu:
#             #         hidden = F.relu(hidden)

#             hidden = features
#             for ix, w in enumerate(self.weights):
#                 b = self.biases[ix] if self.with_bias else 0
#                 if self.sparse_features:
#                     hidden = adj_norm @ torch.spmm(hidden, w) + b
#                 else:
#                     hidden = adj_norm @ hidden @ w + b
#                 if self.with_relu:
#                     hidden = F.relu(hidden)

#             output = F.log_softmax(hidden, dim=1)
#             loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
#             loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

#             if self.lambda_ == 1:
#                 attack_loss = loss_labeled
#             elif self.lambda_ == 0:
#                 attack_loss = loss_unlabeled
#             else:
#                 attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

#             self.optimizer.zero_grad()
#             loss_labeled.backward(retain_graph=True)

#             if self.attack_structure:
#                 self.adj_changes.grad.zero_()
#                 self.adj_grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
#             if self.attack_features:
#                 self.feature_changes.grad.zero_()
#                 self.feature_grad_sum += torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

#             self.optimizer.step()


#         loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
#         print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
#         print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
#     def _single_step_feedback(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
#         """
#         单步反馈：只做一次前向和一次反向，计算关于 adj_changes / feature_changes 的梯度并累加到 adj_grad_sum/feature_grad_sum。
#         优先使用 self.surrogate(如果存在并可调用)进行前向，否则使用内部 weights/biases 做前向，但确保不做 optimizer.step().
#         """
#         # 确保 adj_changes/feature_changes 需要梯度
#         if self.attack_structure:
#             self.adj_changes.requires_grad_(True)
#         if self.attack_features:
#             self.feature_changes.requires_grad_(True)

#         # 计算被扰动的邻接与归一化
#         modified_adj = modified_adj
#         adj_norm = utils.normalize_adj_tensor(modified_adj)

#         # ---- 使用 surrogate (if available) ----
#         use_surrogate = hasattr(self, 'surrogate') and self.surrogate is not None
#         output = None
#         if use_surrogate:
#             # 代理模型应该被冻结（已训练完）。我们只用它的预测，不对其参数求梯度。
#             # 多数deeprobust的GCN类实现了 __call__ 或 forward 两种调用方式，尝试兼容两者。
#             try:
#                 # 尝试直接调用
#                 out = self.surrogate(features, modified_adj)
#             except Exception:
#                 try:
#                     out = self.surrogate.forward(features, modified_adj)
#                 except Exception:
#                     out = None
#                     print("Warning: surrogate model forward method not found. Falling back to internal weights.")
#                     sys.exit(0)
#             if out is not None:
#                 # 如果 surrogate 返回了 logits
#                 output = F.log_softmax(out, dim=1)

#         # ---- 若 surrogate 不可用或不可调用，则用内部权重做前向（但禁止对权重求梯度） ----
#         if output is None:
#             # 保存原来的 requires_grad 状态
#             requires_backup = [w.requires_grad for w in self.weights] + [b.requires_grad for b in self.biases]
#             # 禁止 weights / biases 的梯度计算
#             for w in self.weights:
#                 w.requires_grad_(False)
#             for b in self.biases:
#                 b.requires_grad_(False)

#             # 前向（与原 inner_train 相同的前向逻辑）
#             hidden = features
#             for ix, w in enumerate(self.weights):
#                 b = self.biases[ix] if self.with_bias else 0
#                 if self.sparse_features:
#                     hidden = adj_norm @ torch.spmm(hidden, w) + b
#                 else:
#                     hidden = adj_norm @ hidden @ w + b
#                 if self.with_relu:
#                     hidden = F.relu(hidden)
#             output = F.log_softmax(hidden, dim=1)

#             # 恢复 requires_grad 状态（以免影响外部流程）
#             for param, r in zip(self.weights + self.biases, requires_backup):
#                 param.requires_grad_(r)

#         # ---- 计算攻击损失（single-step） ----
#         # 这里我们沿用你原来的 lambda 策略：loss_labeled / loss_unlabeled 的线性组合
#         loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
#         loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
#         if self.lambda_ == 1:
#             attack_loss = loss_labeled
#         elif self.lambda_ == 0:
#             attack_loss = loss_unlabeled
#         else:
#             attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

#         # ---- 对 adj_changes / feature_changes 求一次梯度 ----
#         # 不对 weights/biases 求梯度与 optimizer.step()
#         if self.attack_structure:
#             # 清空旧 grad（保险起见）
#             if self.adj_changes.grad is not None:
#                 self.adj_changes.grad.zero_()
#             grad_adj = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=False, allow_unused=True)[0]
#             if grad_adj is None:
#                 grad_adj = torch.zeros_like(self.adj_changes)
#             # 累积梯度
#             self.adj_grad_sum += grad_adj

#         if self.attack_features:
#             if self.feature_changes.grad is not None:
#                 self.feature_changes.grad.zero_()
#             grad_feat = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=False, allow_unused=True)[0]
#             if grad_feat is None:
#                 grad_feat = torch.zeros_like(self.feature_changes)
#             self.feature_grad_sum += grad_feat

#         # 清理 requires_grad 标记
#         if self.attack_structure:
#             self.adj_changes.requires_grad_(False)
#         if self.attack_features:
#             self.feature_changes.requires_grad_(False)
#         loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
#         acc_test_val = utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()
#         print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
#         print('GCN acc on unlabled data: {}'.format(acc_test_val))
#         # 返回一下用于 debug 的值
#         #return attack_loss.item()
#     def _normalize_tensor(self, x, eps=1e-8):
#         """把任意 tensor 归一化到 [0,1]，返回同 shape tensor（cpu/cuda 保持不变）。"""
#         x = x.clone()
#         x_min = x.min()
#         x = x - x_min
#         x_max = x.max()
#         if (x_max.item() == 0):
#             return x  # 全零
#         return x / (x_max + eps)

    
#     def get_node_scores(self, adj, x, device):
#      adj = adj.to(device); x = x.to(device)
#      edge_index = adj.nonzero(as_tuple=False).t()
#      num_nodes = x.size(0)
#     # cosine sim per node (归一化到 [0,1])
#      x_norm = x / torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-8)
#      e = torch.sum(x_norm[edge_index[0]] * x_norm[edge_index[1]], dim=1).unsqueeze(-1)  # in [-1,1]
#      e01 = (e + 1) / 2.0  # now in [0,1]
#      row, col = edge_index
#      c = torch.zeros(num_nodes, 1, device=device)
#      c = c.scatter_add_(0, col.unsqueeze(1), e01)
#      deg = degree(col, num_nodes, dtype=x.dtype).unsqueeze(-1).to(device)
#      csim = (c / deg.clamp(min=1)).squeeze()  # in [0,1] roughly

#     # degree norm to [0,1]
#      deg_scalar = degree(col, num_nodes, dtype=x.dtype).to(device)
#      norm_deg = (deg_scalar - deg_scalar.min()) / (deg_scalar.max() - deg_scalar.min() + 1e-8)

#     # ood distance norm to [0,1]
#      mean_feat = x.mean(dim=0, keepdim=True)
#      dist = torch.norm(x - mean_feat, p=2, dim=1)
#      norm_dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)

#     # Decide intended direction:
#     # - If you want to target nodes that are *dissimilar* to neighbors, use (1 - csim)
#     # - If you want to target low-degree nodes, use (1 - norm_deg)
#     # - If you want to target OOD (outliers), use norm_dist (bigger = more OOD)
#     # Example: focus on nodes that are dissimilar and OOD:
#      alpha, beta, gamma = 1,0,0
#      node_score = alpha * (1 - csim) + beta * (1 - norm_deg) + gamma * norm_dist
#      return node_score


       
#     def map_node_to_adj_score(self, node_scores, adj, ori_adj, ll_constraint, ll_cutoff):
#         """
#         将节点分数映射为邻接矩阵得分。
#         通常边的得分 = 节点分数之和 / 2。
#         """
#         num_nodes = node_scores.size(0)
#         adj_meta_score = torch.zeros_like(adj)

#         row, col = adj.nonzero(as_tuple=True)
#         edge_scores = (node_scores[row] + node_scores[col]) / 2.0
#         adj_meta_score[row, col] = edge_scores

#         # 根据原有约束处理（ll_constraint）
#         if ll_constraint:
#             mask = utils.likelihood_ratio_filter(ori_adj, adj_meta_score, ll_cutoff)
#             adj_meta_score = adj_meta_score * mask  

#         return adj_meta_score
#     def _sample_edge_from_score(self, adj_meta_score, ori_shape, k=100, temp=0.5):
#         """
#         从 adj_meta_score 中以 top-k + softmax(multinomial) 的方式抽取一个边的索引。
#         参数：
#         adj_meta_score: tensor, shape (N, N) 或 展平后的 (N*N,)
#         ori_shape: 原始 adjacency 的形状 (通常是 (N, N))
#         k: top-k 候选数量
#         temp: 软化温度，越小越 greedy，越大越随机
#         返回:
#         row_idx, col_idx (int)
#         """
#         # 保证是二维 (N, N)
#         if adj_meta_score.dim() == 1:
#             flat = adj_meta_score
#             n = int(math.sqrt(flat.numel()))
#             # 防护：如果与 ori_shape 不一致，以 ori_shape 为准
#             if ori_shape is not None:
#                 n = ori_shape[0]
#                 flat = adj_meta_score.view(-1)
#         else:
#             flat = adj_meta_score.view(-1)

#         device = flat.device
#         numel = flat.numel()
#         kk = min(k, numel)
#         # 取 top-k
#         topk_vals, topk_idx = torch.topk(flat, kk)
#         # softmax 采样（加温度）
#         probs = torch.softmax(topk_vals / float(temp), dim=0)
#         chosen = topk_idx[torch.multinomial(probs, 1)].item()
#         # 转回 row, col
#         row_idx, col_idx = utils.unravel_index(chosen, ori_shape)
#         return row_idx, col_idx

#     def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
#         """Generate n_perturbations on the input graph.

#         Parameters
#         ----------
#         ori_features :
#             Original (unperturbed) node feature matrix
#         ori_adj :
#             Original (unperturbed) adjacency matrix
#         labels :
#             node labels
#         idx_train :
#             node training indices
#         idx_unlabeled:
#             unlabeled nodes indices
#         n_perturbations : int
#             Number of perturbations on the input graph. Perturbations could
#             be edge removals/additions or feature removals/additions.
#         ll_constraint: bool
#             whether to exert the likelihood ratio test constraint
#         ll_cutoff : float
#             The critical value for the likelihood ratio test of the power law distributions.
#             See the Chi square distribution with one degree of freedom. Default value 0.004
#             corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
#             is False.

#         """
#         ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
#         labels_self_training = self.self_training_label(labels, idx_train)
#         self.sparse_features = sp.issparse(ori_features)
#         modified_adj = ori_adj
#         modified_features = ori_features

#         for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
#             self._initialize()#每次迭代都要重新初始化，否则梯度会累加

#             if self.attack_structure:
#                 modified_adj = self.get_modified_adj(ori_adj)#重新计算修改后的邻接矩阵，否则梯度会累加
#                 self.adj_grad_sum.data.fill_(0)#梯度置0
#             if self.attack_features:
#                 modified_features = ori_features + self.feature_changes
#                 self.feature_grad_sum.data.fill_(0)

#             self._single_step_feedback(modified_features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)
#         #这里不调用inner_train，因为我们只需要计算梯度，不需要训练，那么要改为用Surrogate model的预测结果来评分
#             adj_meta_score = torch.tensor(0.0).to(self.device)#邻接矩阵梯度评分
#             feature_meta_score = torch.tensor(0.0).to(self.device)#特征矩阵梯度评分

#             if self.attack_structure:
#                 adj_meta_score = self.get_node_scores(modified_adj,modified_features, device=self.device)
#                 adj_meta_score = self.map_node_to_adj_score(adj_meta_score, modified_adj, ori_adj, ll_constraint, ll_cutoff)
#             if self.attack_features:
#                 feature_meta_score = self.get_feature_score(self.feature_grad_sum, modified_features)
#             #就是这个位置要把get_feature_score函数改写成能获得特征梯度的函数，然后用特征梯度来评分
#             if adj_meta_score.max() >= feature_meta_score.max():
#                 adj_meta_argmax = torch.argmax(adj_meta_score)
#                 row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
#                 self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
#                 if self.undirected:
#                     self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
#                     # ---- Debug: 检查当前是加边还是删边 ----
#                 delta = (-2 * modified_adj[row_idx][col_idx] + 1).item()
#                 if delta == 1:
#                     print(f"[DEBUG] step {i}: 添加边 ({row_idx}, {col_idx})")
#                 else:
#                     print(f"[DEBUG] step {i}: 删除边 ({row_idx}, {col_idx})")

#             else:
#                 feature_meta_argmax = torch.argmax(feature_meta_score)
#                 row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
#                 self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

            
       

#         if self.attack_structure:
#             self.modified_adj = self.get_modified_adj(ori_adj).detach()
#         if self.attack_features:
#             self.modified_features = self.get_modified_features(ori_features).detach()
    
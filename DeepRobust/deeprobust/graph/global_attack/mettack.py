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

    def get_adj_score(self, adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_meta_grad = adj_meta_grad - adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad = adj_meta_grad - torch.diag(torch.diag(adj_meta_grad, 0))
        # # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad *  singleton_mask

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
class MetaEva(BaseMeta):
    """Approximated version of  Meta Attack. Adversarial Attacks on
    Graph Neural Networks via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import MetaEva
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

        super(MetaEva, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

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
    def _single_step_feedback(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        """
        单步反馈：只做一次前向和一次反向，计算关于 adj_changes / feature_changes 的梯度并累加到 adj_grad_sum/feature_grad_sum。
        优先使用 self.surrogate(如果存在并可调用)进行前向，否则使用内部 weights/biases 做前向，但确保不做 optimizer.step().
        """
        # 确保 adj_changes/feature_changes 需要梯度
        if self.attack_structure:
            self.adj_changes.requires_grad_(True)
        if self.attack_features:
            self.feature_changes.requires_grad_(True)

        # 计算被扰动的邻接与归一化
        modified_adj = modified_adj
        adj_norm = utils.normalize_adj_tensor(modified_adj)

        # ---- 使用 surrogate (if available) ----
        use_surrogate = hasattr(self, 'surrogate') and self.surrogate is not None
        output = None
        if use_surrogate:
            # 代理模型应该被冻结（已训练完）。我们只用它的预测，不对其参数求梯度。
            # 多数deeprobust的GCN类实现了 __call__ 或 forward 两种调用方式，尝试兼容两者。
            try:
                # 尝试直接调用
                out = self.surrogate(features, modified_adj)
            except Exception:
                try:
                    out = self.surrogate.forward(features, modified_adj)
                except Exception:
                    out = None
                    print("Warning: surrogate model forward method not found. Falling back to internal weights.")
                    sys.exit(0)
            if out is not None:
                # 如果 surrogate 返回了 logits
                output = F.log_softmax(out, dim=1)

        # ---- 若 surrogate 不可用或不可调用，则用内部权重做前向（但禁止对权重求梯度） ----
        if output is None:
            # 保存原来的 requires_grad 状态
            requires_backup = [w.requires_grad for w in self.weights] + [b.requires_grad for b in self.biases]
            # 禁止 weights / biases 的梯度计算
            for w in self.weights:
                w.requires_grad_(False)
            for b in self.biases:
                b.requires_grad_(False)

            # 前向（与原 inner_train 相同的前向逻辑）
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

            # 恢复 requires_grad 状态（以免影响外部流程）
            for param, r in zip(self.weights + self.biases, requires_backup):
                param.requires_grad_(r)

        # ---- 计算攻击损失（single-step） ----
        # 这里我们沿用你原来的 lambda 策略：loss_labeled / loss_unlabeled 的线性组合
        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        # ---- 对 adj_changes / feature_changes 求一次梯度 ----
        # 不对 weights/biases 求梯度与 optimizer.step()
        if self.attack_structure:
            # 清空旧 grad（保险起见）
            if self.adj_changes.grad is not None:
                self.adj_changes.grad.zero_()
            grad_adj = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=False, allow_unused=True)[0]
            if grad_adj is None:
                grad_adj = torch.zeros_like(self.adj_changes)
            # 累积梯度
            self.adj_grad_sum += grad_adj

        if self.attack_features:
            if self.feature_changes.grad is not None:
                self.feature_changes.grad.zero_()
            grad_feat = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=False, allow_unused=True)[0]
            if grad_feat is None:
                grad_feat = torch.zeros_like(self.feature_changes)
            self.feature_grad_sum += grad_feat

        # 清理 requires_grad 标记
        if self.attack_structure:
            self.adj_changes.requires_grad_(False)
        if self.attack_features:
            self.feature_changes.requires_grad_(False)
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        acc_test_val = utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()
        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(acc_test_val))
        # 返回一下用于 debug 的值
        #return attack_loss.item()

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
            self._initialize()#每次迭代都要重新初始化，否则梯度会累加

            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)#重新计算修改后的邻接矩阵，否则梯度会累加
                self.adj_grad_sum.data.fill_(0)#梯度置0
            if self.attack_features:
                modified_features = ori_features + self.feature_changes
                self.feature_grad_sum.data.fill_(0)

            self._single_step_feedback(modified_features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)
        #这里不调用inner_train，因为我们只需要计算梯度，不需要训练，那么要改为用Surrogate model的预测结果来评分
            adj_meta_score = torch.tensor(0.0).to(self.device)#邻接矩阵梯度评分
            feature_meta_score = torch.tensor(0.0).to(self.device)#特征矩阵梯度评分

            if self.attack_structure:
                adj_meta_score = self.get_adj_score(self.adj_grad_sum, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(self.feature_grad_sum, modified_features)
#就是这个位置要把get_feature_score函数改写成能获得特征梯度的函数，然后用特征梯度来评分
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
    #def attack(self, ori_features, ori_adj, labels, idx_test, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
# class MetaEvasion(BaseMeta):
#     """
#     Evasion-style attack: do NOT retrain surrogate. Compute gradients
#     w.r.t adj_changes via a single forward/backward pass on the (modified) graph.
#     适合测试时逃逸攻击（no training set access）。
#     参数含义与父类一致。

#     usage:
#       attacker = MetaEvasion(surrogate, nnodes=..., attack_structure=True, device='cpu')
#       # 若不知道真实标签，idx_attack为待攻击节点索引
#       attacker.attack(features, adj, labels, idx_attack, n_perturbations=10, targeted=False)
#       modified_adj = attacker.modified_adj
#     """
#     def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', lr=0.01):
#         super(MetaEvasion, self).__init__(model, nnodes, feature_shape, lambda_=0.5, attack_structure=attack_structure, attack_features=attack_features, undirected=undirected, device=device)
#         # lr 用于在 adj_changes 上做小步更新（梯度上升）
#         self.lr = lr
#         # 保证 adj_changes 可求梯度（父类已创建 Parameter）
#         if attack_structure:
#             self.adj_changes.requires_grad = True
#     # ----- helper: get k-hop subgraph -----
#     def k_hop_subgraph(adj_matrix, seed_nodes, k=2, undirected=True):
#         """
#         adj_matrix: scipy sparse or torch tensor (dense) adjacency (on CPU)
#         seed_nodes: list or 1D np array of node indices (int)
#         sub_nodes (np.array), node_map (dict old->new)
#         """
#         import collections
#         if sp.issparse(adj_matrix):
#             A = adj_matrix.tocsr()
#         else:
#         # ensure numpy adjacency for neighborhood traversal
#             A = sp.csr_matrix(adj_matrix.cpu().numpy()) if isinstance(adj_matrix, torch.Tensor) else sp.csr_matrix(adj_matrix)
#         visited = set(seed_nodes)
#         frontier = set(seed_nodes)
#         for _ in range(k):
#             next_front = set()
#             for u in frontier:
#                 row = A.getrow(u).indices
#                 next_front.update(row)
#             new = next_front - visited
#             visited |= next_front
#             frontier = new
#         sub_nodes = np.array(sorted(list(visited)), dtype=int)
#         node_map = {int(n): i for i, n in enumerate(sub_nodes)}
#         return sub_nodes, node_map

#     def get_modified_adj_tensor(self, ori_adj):
#         # 与父类 get_modified_adj 类似，但返回 tensor 用于前向
#         adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes))
#         if self.undirected:
#             adj_changes_square = adj_changes_square + adj_changes_square.t()
#         adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
#         modified_adj = adj_changes_square + ori_adj
#         return modified_adj

#     def get_evasion_grad(self, features, modified_adj, target_idx, target_labels=None, targeted=False, loss_type='ce'):
#         """
#         计算单次前向下 adj_changes 的梯度。
#         - target_idx: 要攻击的节点索引（array-like）
#         - target_labels: 若已知真实标签，传入真实标签；若为 None，则使用 surrogate 的预测 label 作为 pseudo-label（untargeted场景）
#         - targeted: 若 True 执行有目标攻击（想让模型预测为给定 label）；若 False 则 untargeted（想改变 / 降低对原类置信度）
#         - loss_type: 'ce' 或 'entropy' 等（目前实现 ce 和 confidence）
#         返回：adj_grad (和 feature_grad 如果需要)
#         """
#         # 准备输入
#         adj_norm = utils.normalize_adj_tensor(modified_adj)
#         # 前向：采用 surrogate 的 forward（假设 surrogate 存在 .forward 接受 features, adj_norm）
#         # 这里直接调用 self.surrogate.output 风格或 surrogate.forward
#         # 为通用，尝试 self.surrogate.output first, else call surrogate.forward
#         self.surrogate.eval()
#         for p in self.surrogate.parameters():
#             p.requires_grad = False

#         # compute hidden/output in surrogate style; try to use existing interface:
#         try:
#             output = self.surrogate.output  # 如果在外部已经有 output buffer（不常见）
#             # 如果 output 存在，但可能是上次训练的，需要重新前向:
#             # fallback to forward below
#             raise Exception()
#         except Exception:
#             # 一般 surrogate 有 forward(features, adj) 或 fit/forward 实现
#             # 为最小依赖，尝试 self.surrogate.predict_logits 或直接 call model
#             # 大多数实现：model(features, adj_norm)
#             output = self.surrogate(features, adj_norm)  # 假设返回 logits
#             if output.dim() == 2:
#                 logp = F.log_softmax(output, dim=1)
#             else:
#                 logp = F.log_softmax(output, dim=1)

#         # 若没有 target_labels，使用 pseudo-labels（当前预测）
#         if target_labels is None:
#             pseudo = logp.argmax(dim=1)
#             orig_labels = pseudo[target_idx]
#         else:
#             orig_labels = target_labels[target_idx]

#         # 构造损失：
#         if targeted:
#             # 若 targeted，目标 label 应由 target_labels 提供（或由外部指定）
#             loss = F.nll_loss(logp[target_idx], orig_labels)
#             # 对 targeted 攻击，想最大化目标类概率 -> 最小化负 log prob，等同于最小化 nll_loss
#             # but we'll perform gradient ascent on adj_changes to increase target prob
#         else:
#             # untargeted: 我们想降低模型对原类别的置信度 —— maximize loss
#             loss = F.nll_loss(logp[target_idx], orig_labels)
#             # 对 adj_changes 做梯度上升以增大该 loss (从而降低原类置信度)

#         # compute gradient wrt adj_changes
#         if self.attack_structure:
#             if self.adj_changes.grad is not None:
#                 self.adj_changes.grad.zero_()
#             # ensure adj_changes participates in computation graph: modified_adj depends on adj_changes
#             # We used modified_adj earlier; ensure it was computed from self.adj_changes
#             adj_grad = torch.autograd.grad(loss, self.adj_changes, retain_graph=False, allow_unused=True)[0]
#         else:
#             adj_grad = None

#         return adj_grad

#     def attack(self, ori_features, ori_adj, labels, idx_attack, n_perturbations, k_hop=2, targeted=False, target_labels=None, ll_constraint=True, ll_cutoff=0.004):
#         """
#         使用 k-hop 子图来做攻击，每次只在目标节点的子图上选择一条边翻转。
#         """
#         # 确保原始在 CPU（用于子图抽取），也保持 ori_adj 的稀疏形式
#         # ori_adj, ori_features, labels 已通过 utils.to_tensor 前置（你可按需要调整）
#         self.sparse_features = sp.issparse(ori_features)
#         ori_adj_tensor, ori_features_tensor, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
#         # 但是用于子图提取需要 CPU/sparse 形式：
#         ori_adj_cpu = ori_adj if sp.issparse(ori_adj) else sp.csr_matrix(ori_adj.cpu().numpy())

#         # 确保 adj_changes 存在于 CPU 并初始化（全图 CPU 存储）
#         if not hasattr(self, 'adj_changes') or self.adj_changes is None:
#             self.adj_changes = Parameter(torch.zeros(self.nnodes, self.nnodes), requires_grad=False)
#             self.adj_changes.data.fill_(0.0)
#             self.adj_changes.data = self.adj_changes.data.to('cpu')  # keep on CPU
#         else:
#             # 若已存在，确保搬到 CPU，避免后续 NumPy 转换报错
#             self.adj_changes.data = self.adj_changes.data.detach().cpu()

#         # 将 idx_attack 标准化为列表
#         if isinstance(idx_attack, (int, np.integer)):
#             idx_list = [int(idx_attack)]
#         else:
#             idx_list = list(idx_attack)

#         # 对每个攻击步骤，先选一个 target node（可以使用轮换或随机）
#         for step in range(n_perturbations):
#             # 这里我们轮流攻击 idx_list 中的节点，或随机选择一个
#             tgt = idx_list[step % len(idx_list)]
#             # 抽取子图节点与 node_map
#             sub_nodes, node_map = MetaEvasion.k_hop_subgraph(ori_adj_cpu, [tgt], k=k_hop, undirected=self.undirected)
#             n_sub = len(sub_nodes)
#             if n_sub <= 1:
#                 continue

#             # 构建子图的邻接/feature（把子图数据放到设备）
#             # ori_adj_cpu is sparse CSR -> get submatrix
#             rows = sub_nodes
#             cols = sub_nodes
#             adj_sub = ori_adj_cpu[rows][:, cols]  # still sparse
#             # convert to dense torch on device (n_sub small)
#             adj_sub_dense = torch.FloatTensor(adj_sub.toarray()).to(self.device)
#             if sp.issparse(ori_features):
#                 feat_sub = ori_features.tocsr()[rows].toarray()
#                 feat_sub = torch.FloatTensor(feat_sub).to(self.device)
#             else:
#                 feat_sub = ori_features[rows].to(self.device)

#             # 创建子图层面的 adj_changes_sub（在 GPU 上）
#             adj_changes_sub = torch.zeros((n_sub, n_sub), device=self.device, requires_grad=True)
#             # 注意：如果你想保留以前翻转的效果，需要把全局 adj_changes 的相关 entries 映射到这里并初始化
#             # global flips -> transfer
#             global_changes_slice = self.adj_changes.data[sub_nodes[:, None], sub_nodes]  # on CPU
#             if torch.is_tensor(global_changes_slice):
#                 adj_changes_sub.data = global_changes_slice.to(self.device)

#             # modified_adj_sub = adj_sub_dense + adj_changes_sub
#             modified_adj_sub = torch.clamp(adj_changes_sub - torch.diag(torch.diag(adj_changes_sub)), -1, 1)
#             if self.undirected:
#                 modified_adj_sub = modified_adj_sub + modified_adj_sub.t()
#             modified_adj_sub = modified_adj_sub + adj_sub_dense

#             # 前向得到 loss（与 get_evasion_grad 类似，但只在子图上）
#             adj_norm_sub = utils.normalize_adj_tensor(modified_adj_sub)
#             out = self.surrogate(feat_sub, adj_norm_sub)  # logits
#             logp = F.log_softmax(out, dim=1)

#             # determine labels for nodes in subgraph
#             if target_labels is None:
#                 pseudo = logp.argmax(dim=1)
#                 orig_label = pseudo[node_map[tgt]]
#             else:
#                 orig_label = target_labels[tgt]

#             loss = F.nll_loss(logp[node_map[tgt]].unsqueeze(0), orig_label.unsqueeze(0)) if isinstance(orig_label, torch.Tensor) else F.nll_loss(logp[node_map[tgt]].unsqueeze(0), torch.LongTensor([int(orig_label)]).to(self.device))

#             # 计算 adj_changes_sub 的梯度
#             adj_grad_sub = torch.autograd.grad(loss, adj_changes_sub, retain_graph=False, allow_unused=True)[0]
#             if adj_grad_sub is None:
#                 continue

#             # 得到局部 adj_meta_grad（用 same transform）
#             adj_meta_grad_sub = adj_grad_sub * (-2 * modified_adj_sub + 1)
#             adj_meta_grad_sub = adj_meta_grad_sub - adj_meta_grad_sub.min()
#             adj_meta_grad_sub = adj_meta_grad_sub - torch.diag(torch.diag(adj_meta_grad_sub))

#             # 过滤单节点（在子图尺度上）
#             # 这里可以调用同样的 filter_potential_singletons 但要把输入换成子图形式
#             # 简化：直接禁止自环（对角）并只选择上三角
#             adj_meta_grad_sub = torch.triu(adj_meta_grad_sub, diagonal=1)

#             # pick best local edge to flip
#             flat_idx = torch.argmax(adj_meta_grad_sub)
#             local_row = (flat_idx // n_sub).item()
#             local_col = (flat_idx % n_sub).item()
#             u = int(sub_nodes[local_row])
#             v = int(sub_nodes[local_col])

#             # apply this flip to global adj_changes (which is on CPU)
#             # delta = (-2 * modified_adj_sub[local_row, local_col] + 1)
#             # note: read modified_adj_sub value, move to cpu scalar
#             delta = float((-2 * modified_adj_sub[local_row, local_col] + 1).detach().cpu().item())
#             # update CPU global adj_changes
#             self.adj_changes.data[u, v] += delta
#             if self.undirected:
#                 self.adj_changes.data[v, u] += delta

#             # optional: free GPU memory for this loop
#             del adj_changes_sub, adj_grad_sub, adj_meta_grad_sub, modified_adj_sub, adj_norm_sub, out, logp
#             torch.cuda.empty_cache()

#         # after loop, produce full modified_adj if needed (but keep it on CPU or produce small dense if required)
#         if self.attack_structure:
#             # you can compute modified_adj_cpu = ori_adj_cpu + self.adj_changes.data (both on CPU)
#             # or convert to sparse flips list for downstream
#             # convert to NumPy on CPU explicitly to avoid CUDA→NumPy error
#             self.modified_adj = (ori_adj_cpu + sp.csr_matrix(self.adj_changes.detach().cpu().numpy())).astype(float)


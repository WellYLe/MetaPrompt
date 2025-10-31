import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import json
"""
确保导入本地 DeepRobust 软件包（而不是站点软件包）。
"""
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DeepRobust'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from deeprobust.graph.defense import GCN
from transfermeta import Metattack
#from deeprobust.graph.global_attack.mettack import MetaEva
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.10,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self',
        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train', 'E-Meta-Self'], help='model variant')

args = parser.parse_args()

# Respect --no-cuda flag; default to CPU for stability on Meta attacks
use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device("cuda:0" if use_cuda else "cpu")
#super(MetaEva, self).__init__(...)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

# 使用Metattack作为默认攻击模型
model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  
                 attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)

def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    import json
    # 执行十次攻击实验
    num_attacks = 10
    results = []
    
    print('=== testing GCN on original(clean) graph ===')
    original_acc = test(adj)
    print(f'Original accuracy: {original_acc:.4f}')
    
    print(f'\n=== Starting {num_attacks} attack experiments ===')
    
    for i in range(num_attacks):
        print(f'\n--- Attack {i+1}/{num_attacks} ---')
        
        # 重新初始化攻击模型以确保每次攻击的独立性
        if 'Self' in args.model:
            lambda_ = 0
        if 'Train' in args.model:
            lambda_ = 1
        if 'Both' in args.model:
            lambda_ = 0.5

        # 创建新的攻击模型实例
        attack_model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  
                               attack_structure=True, attack_features=False, device=device, lambda_=lambda_)
        
        attack_model = attack_model.to(device)
        
        # 执行攻击
        attack_model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
        
        # 获取修改后的邻接矩阵
        modified_adj = attack_model.modified_adj
        
        # 测试攻击后的性能
        print(f'Testing GCN on modified graph (attack {i+1})')
        attacked_acc = test(modified_adj)
        
        # 计算性能下降
        acc_drop = original_acc - attacked_acc
        
        # 记录结果
        result = {
            'attack_id': i+1,
            'original_acc': original_acc,
            'attacked_acc': attacked_acc,
            'acc_drop': acc_drop,
            'perturbations': perturbations
        }
        results.append(result)
        
        print(f'Attack {i+1} - Accuracy drop: {acc_drop:.4f} ({original_acc:.4f} -> {attacked_acc:.4f})')
        
        # 保存每次攻击的扰动图
        attack_model.save_adj(root='./', name=f'{args.dataset}_attack_{i+1}_mod_adj')
        # attack_model.save_features(root='./', name=f'attack_{i+1}_mod_features')
    
    # 计算统计结果
    attacked_accs = [r['attacked_acc'] for r in results]
    acc_drops = [r['acc_drop'] for r in results]
    
    mean_attacked_acc = np.mean(attacked_accs)
    std_attacked_acc = np.std(attacked_accs)
    mean_acc_drop = np.mean(acc_drops)
    std_acc_drop = np.std(acc_drops)
    
    # 生成最终报告
    print('\n' + '='*60)
    print('FINAL ATTACK REPORT')
    print('='*60)
    print(f'Dataset: {args.dataset}')
    print(f'Model: {args.model}')
    print(f'Perturbation rate: {args.ptb_rate}')
    print(f'Number of perturbations: {perturbations}')
    print(f'Number of attacks: {num_attacks}')
    print('-'*60)
    print(f'Original accuracy: {original_acc:.4f}')
    print(f'Mean attacked accuracy: {mean_attacked_acc:.4f} ± {std_attacked_acc:.4f}')
    print(f'Mean accuracy drop: {mean_acc_drop:.4f} ± {std_acc_drop:.4f}')
    print(f'Attack success rate: {sum(1 for drop in acc_drops if drop > 0) / num_attacks * 100:.1f}%')
    print('-'*60)
    
    # 详细结果
    print('Detailed results:')
    for result in results:
        print(f"Attack {result['attack_id']}: {result['original_acc']:.4f} -> {result['attacked_acc']:.4f} "
              f"(drop: {result['acc_drop']:.4f})")
    
    print('='*60)
    
    # 保存结果到文件
    report = {
        'experiment_config': {
            'dataset': args.dataset,
            'model': args.model,
            'perturbation_rate': args.ptb_rate,
            'num_perturbations': perturbations,
            'num_attacks': num_attacks
        },
        'summary': {
            'original_accuracy': original_acc,
            'mean_attacked_accuracy': mean_attacked_acc,
            'std_attacked_accuracy': std_attacked_acc,
            'mean_accuracy_drop': mean_acc_drop,
            'std_accuracy_drop': std_acc_drop,
            'attack_success_rate': sum(1 for drop in acc_drops if drop > 0) / num_attacks
        },
        'detailed_results': results
    }
    
    with open(f'{args.dataset}_{args.model}_attack_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f'Report saved to: {args.dataset}_{args.model}_attack_report.json')

if __name__ == '__main__':
    main()


"""
完整的图对抗攻击测试代码
测试MetAttack框架的完整流程
"""

import os
import sys
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入必要的模块
from mettack import MetAttack
from EdgeFlipMAE import EdgeFlipMAE
from Linearized_GCN import Linearized_GCN
from DeepRobust.deeprobust.graph.data import Dataset
from DeepRobust.deeprobust.graph.defense.gcn import GCN
from utils.partition_graph import partition_graph_equal

def load_test_data():
    """加载测试数据"""
    print("加载Cora数据集...")
    data = Dataset(root='/tmp/', name='cora', setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    
    print(f"图统计信息:")
    print(f"  节点数: {adj.shape[0]}")
    print(f"  边数: {adj.nnz // 2}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  类别数: {labels.max().item() + 1}")
    print(f"  训练节点: {len(idx_train)}")
    print(f"  验证节点: {len(idx_val)}")
    print(f"  测试节点: {len(idx_test)}")
    
    return adj, features, labels, idx_train, idx_val, idx_test

def create_mock_edgeflip_model(device):
    """创建模拟的EdgeFlipMAE模型（用于测试）"""
    print("创建模拟EdgeFlipMAE模型...")
    
    # 创建简单的编码器
    class MockEncoder(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            
        def forward(self, x, edge_index):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    # 创建简单的分类器
    class MockClassifier(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.linear = torch.nn.Linear(hidden_dim * 2, 1)
            
        def forward(self, edge_embeddings):
            return torch.sigmoid(self.linear(edge_embeddings))
    
    # 创建EdgeFlipMAE实例
    model = EdgeFlipMAE(
        gnn_type='GCN',
        dataset_name='Cora',
        input_dim=1433,  # Cora特征维度
        hid_dim=64,
        num_layer=2,
        device=device
    )
    
    # 替换为模拟模型
    model.encoder = MockEncoder(1433, 64).to(device)
    model.edge_classifier = MockClassifier(64).to(device)
    
    return model

def evaluate_attack_performance(clean_adj, clean_features, perturbed_adj, perturbed_features, 
                              labels, idx_train, idx_val, idx_test, device):
    """评估攻击效果"""
    print("\n评估攻击效果...")
    
    # 初始化GCN模型
    gcn = GCN(nfeat=clean_features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5,
              device=device)
    
    # 转换为张量
    if sp.issparse(clean_adj):
        clean_adj_tensor = torch.FloatTensor(clean_adj.toarray()).to(device)
    else:
        clean_adj_tensor = torch.FloatTensor(clean_adj).to(device)
        
    if sp.issparse(clean_features):
        clean_features_tensor = torch.FloatTensor(clean_features.toarray()).to(device)
    else:
        clean_features_tensor = torch.FloatTensor(clean_features).to(device)
    
    perturbed_adj_tensor = torch.FloatTensor(perturbed_adj).to(device)
    perturbed_features_tensor = torch.FloatTensor(perturbed_features).to(device)
    
    labels_tensor = torch.LongTensor(labels).to(device)
    
    # 在干净图上训练
    print("在干净图上训练GCN...")
    gcn.fit(clean_features_tensor, clean_adj_tensor, labels_tensor, idx_train, idx_val, 
            train_iters=200, verbose=False)
    
    # 在干净图上测试
    gcn.eval()
    clean_output = gcn.predict(clean_features_tensor, clean_adj_tensor)
    clean_acc = (clean_output[idx_test].argmax(1) == labels_tensor[idx_test]).float().mean()
    
    # 在扰动图上测试
    perturbed_output = gcn.predict(perturbed_features_tensor, perturbed_adj_tensor)
    perturbed_acc = (perturbed_output[idx_test].argmax(1) == labels_tensor[idx_test]).float().mean()
    
    # 计算攻击成功率
    attack_success_rate = (clean_acc - perturbed_acc) / clean_acc
    
    print(f"干净图准确率: {clean_acc:.4f}")
    print(f"扰动图准确率: {perturbed_acc:.4f}")
    print(f"准确率下降: {clean_acc - perturbed_acc:.4f}")
    print(f"攻击成功率: {attack_success_rate:.4f}")
    
    return {
        'clean_accuracy': clean_acc.item(),
        'perturbed_accuracy': perturbed_acc.item(),
        'accuracy_drop': (clean_acc - perturbed_acc).item(),
        'attack_success_rate': attack_success_rate.item()
    }

def analyze_perturbations(clean_adj, clean_features, perturbed_adj, perturbed_features):
    """分析扰动统计"""
    print("\n分析扰动统计...")
    
    # 邻接矩阵扰动
    if sp.issparse(clean_adj):
        clean_adj_dense = clean_adj.toarray()
    else:
        clean_adj_dense = clean_adj
        
    adj_diff = np.abs(perturbed_adj - clean_adj_dense)
    edge_changes = np.sum(adj_diff) / 2  # 除以2因为对称矩阵
    
    # 特征扰动
    if sp.issparse(clean_features):
        clean_features_dense = clean_features.toarray()
    else:
        clean_features_dense = clean_features
        
    feature_diff = np.abs(perturbed_features - clean_features_dense)
    feature_changes = np.sum(feature_diff > 1e-6)
    
    print(f"边变化数量: {edge_changes}")
    print(f"特征变化数量: {feature_changes}")
    print(f"边变化比例: {edge_changes / (clean_adj_dense.shape[0] * (clean_adj_dense.shape[0] - 1) / 2):.6f}")
    print(f"特征变化比例: {feature_changes / (clean_features_dense.shape[0] * clean_features_dense.shape[1]):.6f}")
    
    return {
        'edge_changes': edge_changes,
        'feature_changes': feature_changes,
        'edge_change_ratio': edge_changes / (clean_adj_dense.shape[0] * (clean_adj_dense.shape[0] - 1) / 2),
        'feature_change_ratio': feature_changes / (clean_features_dense.shape[0] * clean_features_dense.shape[1])
    }

def main():
    """主测试函数"""
    print("=" * 60)
    print("MetAttack 图对抗攻击框架测试")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 1. 加载数据
        adj, features, labels, idx_train, idx_val, idx_test = load_test_data()
        
        # 2. 创建攻击器
        print("\n初始化MetAttack攻击器...")
        attacker = MetAttack(
            model=None,  # 将在attack方法中初始化
            nnodes=adj.shape[0],
            attack_structure=True,
            attack_features=True,
            device=device
        )
        
        # 3. 执行攻击
        print("\n开始执行攻击...")
        n_perturbations = 5  # 减少扰动数量以加快测试
        
        perturbed_adj, perturbed_features = attacker.attack(
            ori_features=features,
            ori_adj=adj,
            labels=labels,
            idx_train=idx_train,
            n_perturbations=n_perturbations
        )
        
        # 4. 分析扰动
        perturbation_stats = analyze_perturbations(adj, features, perturbed_adj, perturbed_features)
        
        # 5. 评估攻击效果
        attack_results = evaluate_attack_performance(
            adj, features, perturbed_adj, perturbed_features,
            labels, idx_train, idx_val, idx_test, device
        )
        
        # 6. 输出总结
        print("\n" + "=" * 60)
        print("攻击测试完成!")
        print("=" * 60)
        print("扰动统计:")
        print(f"  边变化: {perturbation_stats['edge_changes']}")
        print(f"  特征变化: {perturbation_stats['feature_changes']}")
        print(f"  边变化比例: {perturbation_stats['edge_change_ratio']:.6f}")
        print(f"  特征变化比例: {perturbation_stats['feature_change_ratio']:.6f}")
        
        print("\n攻击效果:")
        print(f"  干净图准确率: {attack_results['clean_accuracy']:.4f}")
        print(f"  扰动图准确率: {attack_results['perturbed_accuracy']:.4f}")
        print(f"  准确率下降: {attack_results['accuracy_drop']:.4f}")
        print(f"  攻击成功率: {attack_results['attack_success_rate']:.4f}")
        
        # 7. 保存结果
        results = {
            'perturbation_stats': perturbation_stats,
            'attack_results': attack_results,
            'n_perturbations': n_perturbations
        }
        
        torch.save(results, 'attack_test_results.pt')
        print(f"\n结果已保存到: attack_test_results.pt")
        
        return True
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 攻击测试成功完成!")
    else:
        print("\n❌ 攻击测试失败!")
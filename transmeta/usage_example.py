"""
增强版Metattack使用示例
展示如何使用GPF层和边扰动预测模型进行图攻击
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 导入我们的模块
from enhanced_mettack import EnhancedMetattack, GPF, EdgePerturbationPredictor
from edge_predictor_trainer import train_edge_predictor, EdgePerturbationDataset

# 导入DeepRobust的GCN模型
try:
    from deeprobust.graph.defense import GCN
    from deeprobust.graph.data import Dataset
except ImportError:
    print("请安装DeepRobust: pip install deeprobust")
    exit(1)


class SimpleGCN(torch.nn.Module):
    """简单的GCN模型作为代理模型"""
    
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_relu=True, with_bias=True):
        super(SimpleGCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.hidden_sizes = [nhid]
        self.with_relu = with_relu
        self.with_bias = with_bias
        
        self.gc1 = torch.nn.Linear(nfeat, nhid, bias=with_bias)
        self.gc2 = torch.nn.Linear(nhid, nclass, bias=with_bias)
        self.dropout = dropout
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x)
        x = adj @ x
        if self.with_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        x = adj @ x
        self.output = F.log_softmax(x, dim=1)
        return self.output


def load_cora_data():
    """加载Cora数据集"""
    print("加载Cora数据集...")
    
    # 使用PyTorch Geometric加载数据
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    # 转换为numpy格式
    adj = to_dense_adj(data.edge_index)[0].numpy()
    features = data.x.numpy()
    labels = data.y.numpy()
    
    # 划分训练/验证/测试集
    idx_train = np.where(data.train_mask.numpy())[0]
    idx_val = np.where(data.val_mask.numpy())[0]
    idx_test = np.where(data.test_mask.numpy())[0]
    idx_unlabeled = np.union1d(idx_val, idx_test)
    
    print(f"节点数: {features.shape[0]}")
    print(f"特征维度: {features.shape[1]}")
    print(f"类别数: {len(np.unique(labels))}")
    print(f"训练集大小: {len(idx_train)}")
    print(f"验证集大小: {len(idx_val)}")
    print(f"测试集大小: {len(idx_test)}")
    
    return adj, features, labels, idx_train, idx_val, idx_test, idx_unlabeled


def train_surrogate_model(features, adj, labels, idx_train, idx_val, device='cpu'):
    """训练代理模型"""
    print("训练代理GCN模型...")
    
    # 转换为tensor
    features = torch.FloatTensor(features).to(device)
    adj = torch.FloatTensor(adj).to(device)
    labels = torch.LongTensor(labels).to(device)
    
    # 创建模型
    model = SimpleGCN(
        nfeat=features.shape[1],
        nhid=16,
        nclass=labels.max().item() + 1,
        dropout=0.5,
        with_relu=False,  # Metattack要求
        with_bias=False   # Metattack要求
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 训练
    model.train()
    best_val_acc = 0
    patience = 0
    max_patience = 100
    
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        
        # 验证
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                output = model(features, adj)
                pred = output.argmax(dim=1)
                val_acc = accuracy_score(labels[idx_val].cpu(), pred[idx_val].cpu())
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience = 0
                    torch.save(model.state_dict(), 'best_surrogate_model.pth')
                else:
                    patience += 1
                
                if patience >= max_patience:
                    break
            
            model.train()
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_surrogate_model.pth'))
    model.eval()
    
    print(f"代理模型训练完成，最佳验证准确率: {best_val_acc:.4f}")
    return model


def evaluate_attack_performance(original_adj, modified_adj, features, labels, 
                              idx_train, idx_test, surrogate_model, device='cpu'):
    """评估攻击效果"""
    print("评估攻击效果...")
    
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    
    # 原始图性能
    original_adj_tensor = torch.FloatTensor(original_adj).to(device)
    with torch.no_grad():
        original_output = surrogate_model(features, original_adj_tensor)
        original_pred = original_output.argmax(dim=1)
        original_acc = accuracy_score(labels[idx_test].cpu(), original_pred[idx_test].cpu())
    
    # 攻击后图性能
    modified_adj_tensor = torch.FloatTensor(modified_adj).to(device)
    with torch.no_grad():
        modified_output = surrogate_model(features, modified_adj_tensor)
        modified_pred = modified_output.argmax(dim=1)
        modified_acc = accuracy_score(labels[idx_test].cpu(), modified_pred[idx_test].cpu())
    
    # 计算攻击成功率
    attack_success_rate = (original_acc - modified_acc) / original_acc * 100
    
    print(f"原始图测试准确率: {original_acc:.4f}")
    print(f"攻击后测试准确率: {modified_acc:.4f}")
    print(f"准确率下降: {original_acc - modified_acc:.4f}")
    print(f"攻击成功率: {attack_success_rate:.2f}%")
    
    return {
        'original_acc': original_acc,
        'modified_acc': modified_acc,
        'accuracy_drop': original_acc - modified_acc,
        'attack_success_rate': attack_success_rate
    }


def visualize_attack_results(results, save_path=None):
    """可视化攻击结果"""
    methods = list(results.keys())
    original_accs = [results[method]['original_acc'] for method in methods]
    modified_accs = [results[method]['modified_acc'] for method in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, original_accs, width, label='原始准确率', alpha=0.8)
    bars2 = ax.bar(x + width/2, modified_accs, width, label='攻击后准确率', alpha=0.8)
    
    ax.set_xlabel('攻击方法')
    ax.set_ylabel('准确率')
    ax.set_title('不同攻击方法的效果比较')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果图已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    print("=== 增强版Metattack演示 ===\n")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 1. 加载数据
    adj, features, labels, idx_train, idx_val, idx_test, idx_unlabeled = load_cora_data()
    
    # 2. 训练代理模型
    surrogate_model = train_surrogate_model(features, adj, labels, idx_train, idx_val, device)
    
    # 3. 训练边扰动预测模型
    print("\n训练边扰动预测模型...")
    edge_predictor, _ = train_edge_predictor(
        node_features=features,
        adj_matrix=adj,
        num_epochs=50,
        lr=0.001,
        device=device,
        save_path='edge_predictor_cora.pth'
    )
    
    # 4. 设置攻击参数
    n_perturbations = 50  # 扰动数量
    
    results = {}
    
    # 5. 原始Metattack（无GPF层和边预测）
    print(f"\n=== 原始Metattack攻击 ===")
    try:
        from mettack import Metattack  # 假设原始实现存在
        
        original_attacker = Metattack(
            model=surrogate_model,
            nnodes=adj.shape[0],
            attack_structure=True,
            attack_features=False,
            device=device,
            lambda_=0.5
        )
        
        original_attacker.attack(
            ori_features=features,
            ori_adj=adj,
            labels=labels,
            idx_train=idx_train,
            idx_unlabeled=idx_unlabeled,
            n_perturbations=n_perturbations
        )
        
        original_modified_adj = original_attacker.modified_adj.cpu().numpy()
        results['原始Metattack'] = evaluate_attack_performance(
            adj, original_modified_adj, features, labels, 
            idx_train, idx_test, surrogate_model, device
        )
        
    except ImportError:
        print("原始Metattack不可用，跳过比较")
    
    # 6. 增强版Metattack（仅GPF层）
    print(f"\n=== 增强版Metattack (仅GPF层) ===")
    
    enhanced_attacker_gpf = EnhancedMetattack(
        model=surrogate_model,
        nnodes=adj.shape[0],
        feature_shape=features.shape,
        edge_predictor_path=None,  # 不使用边预测
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=0.5,
        gpf_lr=0.01
    )
    
    # 获取边索引
    edge_index = torch.nonzero(torch.FloatTensor(adj)).t().contiguous()
    
    modified_adj_gpf, trained_gpf = enhanced_attacker_gpf.attack_with_gpf(
        ori_features=features,
        ori_adj=adj,
        labels=labels,
        idx_train=idx_train,
        idx_unlabeled=idx_unlabeled,
        n_perturbations=n_perturbations,
        edge_index=edge_index
    )
    
    results['增强版(GPF层)'] = evaluate_attack_performance(
        adj, modified_adj_gpf.cpu().numpy(), features, labels,
        idx_train, idx_test, surrogate_model, device
    )
    
    # 7. 增强版Metattack（GPF层 + 边预测）
    print(f"\n=== 增强版Metattack (GPF层 + 边预测) ===")
    
    enhanced_attacker_full = EnhancedMetattack(
        model=surrogate_model,
        nnodes=adj.shape[0],
        feature_shape=features.shape,
        edge_predictor_path='edge_predictor_cora.pth',
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=0.5,
        gpf_lr=0.01,
        edge_weight=0.1  # 边预测损失权重
    )
    
    modified_adj_full, trained_gpf_full = enhanced_attacker_full.attack_with_gpf(
        ori_features=features,
        ori_adj=adj,
        labels=labels,
        idx_train=idx_train,
        idx_unlabeled=idx_unlabeled,
        n_perturbations=n_perturbations,
        edge_index=edge_index
    )
    
    results['增强版(完整)'] = evaluate_attack_performance(
        adj, modified_adj_full.cpu().numpy(), features, labels,
        idx_train, idx_test, surrogate_model, device
    )
    
    # 8. 结果比较和可视化
    print(f"\n=== 攻击效果总结 ===")
    for method, result in results.items():
        print(f"{method}:")
        print(f"  原始准确率: {result['original_acc']:.4f}")
        print(f"  攻击后准确率: {result['modified_acc']:.4f}")
        print(f"  准确率下降: {result['accuracy_drop']:.4f}")
        print(f"  攻击成功率: {result['attack_success_rate']:.2f}%")
        print()
    
    # 可视化结果
    visualize_attack_results(results, 'attack_comparison.png')
    
    # 9. 分析GPF层学到的特征提示
    print("=== GPF层分析 ===")
    print(f"GPF全局提示向量形状: {trained_gpf_full.global_prompt.shape}")
    print(f"GPF提示向量范数: {torch.norm(trained_gpf_full.global_prompt).item():.4f}")
    print(f"GPF提示向量均值: {torch.mean(trained_gpf_full.global_prompt).item():.4f}")
    print(f"GPF提示向量标准差: {torch.std(trained_gpf_full.global_prompt).item():.4f}")
    
    # 保存训练好的GPF层
    torch.save(trained_gpf_full, 'trained_gpf_layer.pth')
    print("训练好的GPF层已保存到: trained_gpf_layer.pth")
    
    print("\n演示完成！")


if __name__ == "__main__":
    main()
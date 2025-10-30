"""
测试不同的样本平衡策略
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import scipy.sparse as sp
from dataconstruction import construct_edge_diff_dataset

def create_test_data():
    """创建测试数据，模拟正样本多于负样本的情况"""
    # 创建一个小的测试图
    n_nodes = 10
    
    # 干净图：较少的边
    clean_edges = [(0,1), (1,2), (2,3), (3,4)]
    clean_adj = sp.csr_matrix((n_nodes, n_nodes))
    for i, j in clean_edges:
        clean_adj[i, j] = 1
        clean_adj[j, i] = 1  # 无向图
    
    # 攻击图：添加更多边，使得翻转边（正样本）多于未翻转边（负样本）
    attack_edges = clean_edges + [(0,5), (1,6), (2,7), (3,8), (4,9), (5,6), (6,7), (7,8)]
    attack_adj = sp.csr_matrix((n_nodes, n_nodes))
    for i, j in attack_edges:
        attack_adj[i, j] = 1
        attack_adj[j, i] = 1
    
    # 创建随机特征
    X_reduced = np.random.randn(n_nodes, 5)
    
    return clean_adj, attack_adj, X_reduced

def test_balance_strategies():
    """测试不同的平衡策略"""
    print("=== 测试样本平衡策略 ===\n")
    
    # 创建测试数据
    clean_adj, attack_adj, X_reduced = create_test_data()
    
    strategies = ['min', 'pos', 'neg']
    
    for strategy in strategies:
        print(f"--- 测试策略: {strategy} ---")
        try:
            edge_pairs, X_pairs, y = construct_edge_diff_dataset(
                clean_adj, attack_adj, X_reduced,
                balance=True,
                balance_strategy=strategy,
                random_state=42
            )
            
            n_pos = np.sum(y)
            n_neg = len(y) - n_pos
            print(f"最终结果: 正样本 {n_pos}, 负样本 {n_neg}")
            print(f"正负比例: {n_pos/n_neg:.2f}:1\n")
            
        except Exception as e:
            print(f"策略 {strategy} 失败: {e}\n")
    
    # 测试不平衡模式
    print("--- 测试不平衡模式 ---")
    edge_pairs, X_pairs, y = construct_edge_diff_dataset(
        clean_adj, attack_adj, X_reduced,
        balance=False,
        random_state=42
    )
    n_pos = np.sum(y)
    n_neg = len(y) - n_pos
    print(f"不平衡结果: 正样本 {n_pos}, 负样本 {n_neg}")
    print(f"正负比例: {n_pos/n_neg:.2f}:1\n")

if __name__ == "__main__":
    test_balance_strategies()
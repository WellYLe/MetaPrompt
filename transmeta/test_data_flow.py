"""
数据流测试代码
验证MetAttack框架中各模块间的数据格式兼容性
"""

import os
import sys
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_data_conversion():
    """测试数据转换函数"""
    print("测试数据转换...")
    
    # 创建测试数据
    num_nodes = 10
    num_features = 5
    
    # 创建稠密邻接矩阵
    adj_dense = np.random.rand(num_nodes, num_nodes)
    adj_dense = (adj_dense + adj_dense.T) / 2  # 对称化
    adj_dense = (adj_dense > 0.5).astype(float)  # 二值化
    np.fill_diagonal(adj_dense, 0)  # 移除自环
    
    # 创建特征矩阵
    features = np.random.rand(num_nodes, num_features)
    
    print(f"原始数据形状:")
    print(f"  邻接矩阵: {adj_dense.shape}")
    print(f"  特征矩阵: {features.shape}")
    
    # 转换为edge_index格式
    edge_index, _ = dense_to_sparse(torch.FloatTensor(adj_dense))
    print(f"  edge_index: {edge_index.shape}")
    
    # 创建PyG Data对象
    graph_data = Data(x=torch.FloatTensor(features), edge_index=edge_index)
    print(f"  PyG Data: x={graph_data.x.shape}, edge_index={graph_data.edge_index.shape}")
    
    return True

def test_gpf_prompt():
    """测试GPF提示模块"""
    print("\n测试GPF提示模块...")
    
    from mettack import GPF
    
    # 创建测试数据
    batch_size = 3
    num_nodes = 10
    num_features = 5
    
    # 初始化GPF
    prompt = GPF(in_channels=num_features)
    
    # 测试add方法
    x = torch.randn(num_nodes, num_features)
    prompted_x = prompt.add(x)
    
    print(f"原始特征: {x.shape}")
    print(f"提示后特征: {prompted_x.shape}")
    print(f"提示参数: {prompt.global_emb.shape}")
    
    assert prompted_x.shape == x.shape, "提示后特征形状不匹配"
    print("✅ GPF提示模块测试通过")
    
    return True

def test_mock_edgeflip():
    """测试模拟EdgeFlipMAE"""
    print("\n测试模拟EdgeFlipMAE...")
    
    # 创建简单的模拟模型
    class MockEdgeFlipMAE:
        def __init__(self, device):
            self.device = device
            
        def predict_all_edges(self, graph_data):
            """模拟预测所有边的翻转概率"""
            num_edges = graph_data.edge_index.shape[1]
            # 返回随机概率
            return np.random.rand(num_edges)
        
        def final_attack(self, prompt, attacker, modified_adj, modified_features, graph_data):
            """模拟最终攻击"""
            # 返回随机分数
            return np.random.rand()
    
    # 创建测试数据
    num_nodes = 10
    num_features = 5
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2
    adj = (adj > 0.5).float()
    torch.fill_diagonal_(adj, 0)
    
    features = torch.randn(num_nodes, num_features)
    edge_index, _ = dense_to_sparse(adj)
    
    graph_data = Data(x=features, edge_index=edge_index)
    
    # 测试模拟模型
    mock_model = MockEdgeFlipMAE('cpu')
    
    # 测试预测
    flip_probs = mock_model.predict_all_edges(graph_data)
    print(f"边翻转概率: {flip_probs.shape}")
    print(f"概率范围: [{flip_probs.min():.3f}, {flip_probs.max():.3f}]")
    
    # 测试最终攻击
    from mettack import GPF
    prompt = GPF(in_channels=num_features)
    score = mock_model.final_attack(prompt, mock_model, adj, features, graph_data)
    print(f"攻击分数: {score:.3f}")
    
    print("✅ 模拟EdgeFlipMAE测试通过")
    return True

def test_linearized_gcn():
    """测试Linearized_GCN"""
    print("\n测试Linearized_GCN...")
    
    try:
        from Linearized_GCN import Linearized_GCN
        
        # 创建测试数据
        num_nodes = 20
        num_features = 10
        num_classes = 3
        
        # 创建模型
        model = Linearized_GCN(
            input_dim=num_features,
            hid_dim=16,
            out_dim=num_classes,
            num_layer=2
        )
        
        # 创建测试输入
        x = torch.randn(num_nodes, num_features)
        adj = torch.rand(num_nodes, num_nodes)
        adj = (adj + adj.T) / 2
        adj = (adj > 0.3).float()
        torch.fill_diagonal_(adj, 1)  # 添加自环
        
        # 归一化邻接矩阵
        D = torch.sum(adj, dim=1)
        D_inv = torch.pow(D, -0.5)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)
        adj_norm = D_mat_inv @ adj @ D_mat_inv
        
        # 前向传播
        output = model(x, adj_norm)
        
        print(f"输入特征: {x.shape}")
        print(f"归一化邻接矩阵: {adj_norm.shape}")
        print(f"输出: {output.shape}")
        
        assert output.shape == (num_nodes, num_classes), "输出形状不匹配"
        print("✅ Linearized_GCN测试通过")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  无法导入Linearized_GCN: {e}")
        return False

def test_partition_graph():
    """测试图分割"""
    print("\n测试图分割...")
    
    try:
        from utils.partition_graph import partition_graph_equal
        
        # 创建测试数据
        num_nodes = 50
        num_features = 10
        num_classes = 3
        
        # 创建稀疏邻接矩阵
        adj = sp.random(num_nodes, num_nodes, density=0.1, format='csr')
        adj = adj + adj.T  # 对称化
        adj.data = np.ones_like(adj.data)  # 二值化
        
        # 创建特征和标签
        features = sp.random(num_nodes, num_features, density=1.0, format='csr')
        labels = np.random.randint(0, num_classes, num_nodes)
        idx_train = np.random.choice(num_nodes, size=num_nodes//2, replace=False)
        
        # 执行图分割
        train_graphs = partition_graph_equal(
            adj, features, labels, idx_train,
            num_partitions=5,
            partition_method='random'
        )
        
        print(f"分割后子图数量: {len(train_graphs)}")
        
        # 检查第一个子图
        if len(train_graphs) > 0:
            subgraph = train_graphs[0]
            print(f"第一个子图属性: {list(subgraph.__dict__.keys()) if hasattr(subgraph, '__dict__') else 'PyG Data对象'}")
            
            if hasattr(subgraph, 'x'):
                print(f"  节点特征: {subgraph.x.shape}")
            if hasattr(subgraph, 'edge_index'):
                print(f"  边索引: {subgraph.edge_index.shape}")
        
        print("✅ 图分割测试通过")
        return True
        
    except ImportError as e:
        print(f"⚠️  无法导入partition_graph: {e}")
        return False
    except Exception as e:
        print(f"❌ 图分割测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("MetAttack 数据流测试")
    print("=" * 50)
    
    tests = [
        ("数据转换", test_data_conversion),
        ("GPF提示模块", test_gpf_prompt),
        ("模拟EdgeFlipMAE", test_mock_edgeflip),
        ("Linearized_GCN", test_linearized_gcn),
        ("图分割", test_partition_graph),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} 测试成功")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {str(e)}")
            results[test_name] = False
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！数据流验证成功！")
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
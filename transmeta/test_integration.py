#!/usr/bin/env python3
"""
测试Mettack数据集与EdgeFlipMAE模型集成的可行性
"""

import os
import sys
import numpy as np
import torch
import json
from pathlib import Path

# 添加路径以导入相关模块
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "DeepRobust" / "examples" / "graph"))

try:
    from mettack_to_edgeflip_integration import (
        load_mettack_dataset,
        extract_node_features_from_pairs,
        create_graph_from_pairs,
        train_edgeflip_mae_with_mettack_data
    )
    from EdgeFlipMAE import EdgeFlipMAE
    print("✅ 成功导入所有必需模块")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)


def test_data_loading():
    """测试数据加载功能"""
    print("\n=== 测试1: 数据加载功能 ===")
    
    # 查找可用的数据文件
    data_dir = Path(__file__).parent.parent / "DeepRobust" / "examples" / "graph"
    npz_files = list(data_dir.glob("*edgeflip_dataset*.npz"))
    
    if not npz_files:
        print("❌ 未找到mettack生成的数据文件")
        print("请先运行 test_mettack.py 生成数据集")
        return False, None
    
    npz_path = npz_files[0]
    print(f"📁 找到数据文件: {npz_path.name}")
    
    try:
        # 测试数据加载
        dataset_dict = load_mettack_dataset(str(npz_path))
        
        print(f"✅ 数据加载成功")
        print(f"   - 样本数量: {len(dataset_dict['y'])}")
        print(f"   - 特征维度: {dataset_dict['X_pairs'].shape[1]}")
        print(f"   - 正样本比例: {np.mean(dataset_dict['y']):.4f}")
        print(f"   - 节点对数量: {len(dataset_dict['pairs'])}")
        
        return True, dataset_dict
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False, None


def test_feature_extraction(dataset_dict):
    """测试特征提取功能"""
    print("\n=== 测试2: 特征提取功能 ===")
    
    try:
        X_pairs = dataset_dict['X_pairs']
        pairs = dataset_dict['pairs']
        
        # 推断节点数量 - 修复索引问题
        n_nodes = int(pairs.max()) + 1  # 使用max()+1
        
        print(f"📊 推断节点数量: {n_nodes}")
        
        # 提取节点特征
        node_features = extract_node_features_from_pairs(X_pairs, pairs, n_nodes)
        
        print(f"✅ 特征提取成功")
        print(f"   - 节点特征形状: {node_features.shape}")
        print(f"   - 特征范围: [{node_features.min():.4f}, {node_features.max():.4f}]")
        print(f"   - 特征均值: {node_features.mean():.4f}")
        
        return True, node_features
        
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        return False, None


def test_graph_creation(pairs, node_features):
    """测试图创建功能"""
    print("\n=== 测试3: 图创建功能 ===")
    
    try:
        # 创建图数据
        graph_data = create_graph_from_pairs(pairs, node_features)
        
        print(f"✅ 图创建成功")
        print(f"   - 节点数量: {graph_data.num_nodes}")
        print(f"   - 边数量: {graph_data.num_edges}")
        print(f"   - 节点特征维度: {graph_data.x.shape[1]}")
        print(f"   - 图连通性: {graph_data.num_edges / (2 * graph_data.num_nodes):.2f}")
        
        return True, graph_data
        
    except Exception as e:
        print(f"❌ 图创建失败: {e}")
        return False, None


def test_model_initialization(node_feat_dim):
    """测试模型初始化"""
    print("\n=== 测试4: 模型初始化 ===")
    
    try:
        # 初始化EdgeFlipMAE模型
        model = EdgeFlipMAE(
            gnn_type='GCN',
            input_dim=node_feat_dim,
            hid_dim=32,  # 使用较小的维度进行测试
            num_layer=2,
            mask_rate=0.15,
            noise_rate=0.1
        )
        
        print(f"✅ 模型初始化成功")
        print(f"   - 模型类型: {model.gnn_type}")
        print(f"   - 输入维度: {model.input_dim}")
        print(f"   - 隐藏维度: {model.hid_dim}")
        print(f"   - 层数: {model.num_layer}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return False, None


def test_data_loading_into_model(model, dataset_dict, graph_data):
    """测试数据加载到模型"""
    print("\n=== 测试5: 数据加载到模型 ===")
    
    try:
        # 准备数据
        edge_pairs = dataset_dict['pairs']
        X_pairs = dataset_dict['X_pairs']
        labels = dataset_dict['y']
        
        print(f"📊 准备加载数据到模型...")
        print(f"   - 边对数量: {len(edge_pairs)}")
        print(f"   - 标签数量: {len(labels)}")
        
        # 加载数据到模型
        model.load_triplet_data(
            edge_pairs=edge_pairs,
            X_pairs=X_pairs,
            labels=labels,
            graph_data=graph_data,
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        print("✅ 数据加载到模型成功")
        print(f"   - 训练集大小: {len(model.train_dataset)}")
        print(f"   - 验证集大小: {len(model.val_dataset)}")
        print(f"   - 测试集大小: {len(model.test_dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载到模型失败: {e}")
        return False


def test_short_training(model):
    """测试短时间训练"""
    print("\n=== 测试6: 短时间训练 ===")
    
    try:
        print("🚀 开始短时间训练测试 (2个epoch)...")
        
        # 临时修改epochs进行短时间训练
        original_epochs = model.epochs
        model.epochs = 50
        
        # 进行短时间训练
        model.pretrain(batch_size=32)  # 使用较小的batch size
        
        # 恢复原始epochs
        model.epochs = original_epochs
        
        print("✅ 短时间训练成功完成")
        
        # 测试评估
        print("📊 测试模型评估...")
        from torch.utils.data import DataLoader
        test_loader = DataLoader(model.test_dataset, batch_size=32, shuffle=False)
        metrics = model.evaluate(test_loader)
        
        print("✅ 模型评估成功")
        print(f"   - 准确率: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"   - F1分数: {metrics.get('f1', 'N/A'):.4f}")
        print(f"   - AUC: {metrics.get('auc', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """运行综合测试"""
    print("🔍 开始Mettack-EdgeFlipMAE集成可行性测试")
    print("=" * 60)
    
    # 测试1: 数据加载
    success, dataset_dict = test_data_loading()
    if not success:
        return False
    
    # 测试2: 特征提取
    success, node_features = test_feature_extraction(dataset_dict)
    if not success:
        return False
    
    # 测试3: 图创建
    success, graph_data = test_graph_creation(dataset_dict['pairs'], node_features)
    if not success:
        return False
    
    # 测试4: 模型初始化
    success, model = test_model_initialization(node_features.shape[1])
    if not success:
        return False
    
    # 测试5: 数据加载到模型
    success = test_data_loading_into_model(model, dataset_dict, graph_data)
    if not success:
        return False
    
    # 测试6: 短时间训练
    success = test_short_training(model)
    if not success:
        return False
    
    return True


def generate_test_report():
    """生成测试报告"""
    print("\n" + "=" * 60)
    print("📋 测试报告生成")
    
    report = {
        "test_time": str(np.datetime64('now')),
        "test_status": "PASSED" if run_comprehensive_test() else "FAILED",
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__
        }
    }
    
    # 保存测试报告
    report_path = Path(__file__).parent / "integration_test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 测试报告已保存到: {report_path}")
    
    return report["test_status"] == "PASSED"


if __name__ == "__main__":
    print("🧪 Mettack-EdgeFlipMAE集成测试")
    print("=" * 60)
    
    try:
        success = generate_test_report()
        
        if success:
            print("\n🎉 所有测试通过！集成方案可行性验证成功！")
            print("\n📚 接下来您可以：")
            print("   1. 查看 usage_guide.md 了解详细使用方法")
            print("   2. 运行 mettack_to_edgeflip_integration.py 进行完整训练")
            print("   3. 根据需要调整模型超参数")
        else:
            print("\n❌ 测试失败，请检查错误信息并修复问题")
            
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试过程中发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
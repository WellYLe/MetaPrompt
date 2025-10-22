"""
Mettack数据集与EdgeFlipMAE模型集成脚本

该脚本提供了将test_mettack.py生成的边翻转数据集
与EdgeFlipMAE模型进行训练的完整解决方案。

主要功能：
1. 加载mettack生成的数据集（.npz格式）
2. 转换数据格式以适配EdgeFlipMAE模型
3. 训练EdgeFlipMAE模型
4. 评估模型性能
"""

import os
import sys
import torch
import numpy as np
import json
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from torch.utils.data import DataLoader

# 导入EdgeFlipMAE模型
from EdgeFlipMAE import EdgeFlipMAE

def load_mettack_dataset(npz_path, meta_path=None):
    """
    加载mettack生成的数据集
    
    Args:
        npz_path: .npz数据文件路径
        meta_path: 可选的元数据JSON文件路径
    
    Returns:
        dict: 包含X_pairs, pairs, y和stats的字典
    """
    print(f"=== 加载Mettack数据集 ===")
    print(f"数据文件: {npz_path}")
    
    # 加载npz数据
    data = np.load(npz_path)
    dataset_dict = {
        'X_pairs': data['X_pairs'],  # (M, 2*F + k) 节点对特征
        'pairs': data['pairs'],      # (M, 2) 边的节点对索引
        'y': data['y']              # (M,) 边翻转标签
    }
    
    # 加载元数据（如果存在）
    stats = None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            stats = json.load(f)
        print(f"元数据文件: {meta_path}")
    
    # 打印数据集信息
    print(f"数据集大小: {len(dataset_dict['y'])}")
    print(f"正样本(翻转): {np.sum(dataset_dict['y'])}")
    print(f"负样本(未翻转): {len(dataset_dict['y']) - np.sum(dataset_dict['y'])}")
    print(f"特征维度: {dataset_dict['X_pairs'].shape[1]}")
    
    if stats:
        print(f"节点数: {stats.get('N_nodes', 'N/A')}")
        print(f"正样本占比: {stats.get('pos_ratio', 'N/A'):.4f}")
    
    dataset_dict['stats'] = stats
    return dataset_dict

def extract_node_features_from_pairs(X_pairs, pairs, n_nodes=None):
    """
    从节点对特征中提取单个节点的特征
    
    Args:
        X_pairs: (M, 2*F + k) 节点对特征矩阵
        pairs: (M, 2) 边的节点对索引
        n_nodes: 图中节点总数，如果为None则自动推断
    
    Returns:
        node_features: (N, F) 节点特征矩阵
    """
    print("=== 从节点对特征中提取单个节点特征 ===")
    
    # 推断节点数量
    if n_nodes is None:
        n_nodes = int(pairs.max()) + 1  # 修复：使用max()+1而不是max()
    
    print(f"节点数量: {n_nodes}")
    print(f"节点对特征形状: {X_pairs.shape}")
    
    # 推断单个节点特征维度
    # 假设结构特征维度为4: [edge_before, deg_u, deg_v, common_neighbors]
    struct_feat_dim = 4
    node_feat_dim = (X_pairs.shape[1] - struct_feat_dim) // 2
    
    print(f"推断的节点特征维度: {node_feat_dim}")
    print(f"结构特征维度: {struct_feat_dim}")
    
    # 初始化节点特征矩阵
    node_features = np.zeros((n_nodes, node_feat_dim))
    node_count = np.zeros(n_nodes)
    
    # 从节点对特征中提取单个节点特征
    for i, (u, v) in enumerate(pairs):
        # 确保索引在有效范围内
        if u >= n_nodes or v >= n_nodes:
            print(f"警告: 节点索引超出范围 u={u}, v={v}, n_nodes={n_nodes}")
            continue
            
        # 提取节点u和v的特征
        feat_u = X_pairs[i, :node_feat_dim]
        feat_v = X_pairs[i, node_feat_dim:2*node_feat_dim]
        
        # 累加特征
        node_features[u] += feat_u
        node_features[v] += feat_v
        node_count[u] += 1
        node_count[v] += 1
    
    # 对每个节点的特征求平均
    for i in range(n_nodes):
        if node_count[i] > 0:
            node_features[i] /= node_count[i]
        else:
            # 未出现的节点使用随机特征
            node_features[i] = np.random.randn(node_feat_dim) * 0.1
    
    print(f"提取的节点特征形状: {node_features.shape}")
    print(f"有效节点数量: {np.sum(node_count > 0)}")
    
    return node_features

def create_graph_from_pairs(pairs, node_features):
    """
    从边对创建PyTorch Geometric图数据
    
    Args:
        pairs: (M, 2) 边的节点对索引
        node_features: (N, F) 节点特征矩阵
    
    Returns:
        Data: PyTorch Geometric数据对象
    """
    # 创建边索引（无向图，需要双向边）
    edges = []
    for u, v in pairs:
        edges.append([u, v])
        edges.append([v, u])  # 添加反向边
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # 去除重复边
    edge_index = torch.unique(edge_index, dim=1)
    
    # 创建节点特征tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 创建PyG数据对象
    graph_data = Data(x=x, edge_index=edge_index)
    
    print(f"图信息: {graph_data.num_nodes} 个节点, {graph_data.num_edges} 条边")
    
    return graph_data

def train_edgeflip_mae_with_mettack_data(dataset_dict, 
                                        gnn_type='GCN',
                                        hid_dim=64,
                                        num_layer=2,
                                        epochs=100,
                                        batch_size=64,
                                        learning_rate=0.001,
                                        device='auto'):
    """
    使用mettack数据集训练EdgeFlipMAE模型
    
    Args:
        dataset_dict: 从load_mettack_dataset返回的数据字典
        其他参数: EdgeFlipMAE模型的超参数
    
    Returns:
        EdgeFlipMAE: 训练好的模型
    """
    print(f"\n=== 准备训练EdgeFlipMAE模型 ===")
    
    # 设备设置
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 提取数据
    X_pairs = dataset_dict['X_pairs']
    pairs = dataset_dict['pairs']
    y = dataset_dict['y']
    stats = dataset_dict.get('stats', {})
    
    # 推断节点数
    n_nodes = stats.get('N_nodes') if stats else pairs.max() + 1
    print(f"图节点数: {n_nodes}")
    
    # 从节点对特征中提取单个节点特征
    node_features = extract_node_features_from_pairs(X_pairs, pairs, n_nodes)
    
    # 创建图数据
    graph_data = create_graph_from_pairs(pairs, node_features)
    graph_data = graph_data.to(device)
    
    # 初始化EdgeFlipMAE模型
    model = EdgeFlipMAE(
        gnn_type=gnn_type,
        dataset_name='Mettack_EdgeFlip',
        input_dim=node_features.shape[1],
        hid_dim=hid_dim,
        num_layer=num_layer,
        device=device,
        mask_rate=0.15,
        noise_rate=0.1,
        learning_rate=learning_rate,
        weight_decay=5e-4,
        epochs=epochs
    )
    
    # 加载训练数据
    print(f"\n=== 加载训练数据 ===")
    model.load_triplet_data(
        edge_pairs=pairs,
        X_pairs=X_pairs,  # 实际上不直接使用，因为我们用GNN编码
        labels=y,
        graph_data=graph_data,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # 训练模型
    print(f"\n=== 开始训练 ===")
    model.pretrain(batch_size=batch_size)
    
    # 保存模型
    print(f"\n=== 保存模型 ===")
    model.save_model()
    
    # 测试模型
    print(f"\n=== 测试模型 ===")
    test_loader = DataLoader(model.test_dataset, batch_size=batch_size, shuffle=False)
    test_metrics = model.evaluate(test_loader)
    
    print("测试集结果:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return model

def main_integration_example():
    """
    完整的集成示例：从mettack数据集到EdgeFlipMAE训练
    """
    print("=== Mettack数据集与EdgeFlipMAE模型集成示例 ===")
    
    # 1. 设置数据路径（根据你的实际文件调整）
    dataset_name = "cora"
    ptb_rate = "050"  # 对应5%的扰动率
    
    npz_path = f"../DeepRobust/examples/graph/{dataset_name}_edgeflip_dataset_ptbrate{ptb_rate}.npz"
    meta_path = f"../DeepRobust/examples/graph/{dataset_name}_edgeflip_dataset_ptbrate{ptb_rate}_meta.json"
    
    # 检查文件是否存在
    if not os.path.exists(npz_path):
        print(f"错误: 数据文件不存在 {npz_path}")
        print("请先运行test_mettack.py生成数据集")
        return
    
    # 2. 加载mettack数据集
    dataset_dict = load_mettack_dataset(npz_path, meta_path)
    
    # 3. 训练EdgeFlipMAE模型
    model = train_edgeflip_mae_with_mettack_data(
        dataset_dict=dataset_dict,
        gnn_type='GCN',
        hid_dim=64,
        num_layer=2,
        epochs=50,  # 为了演示，使用较少的epoch
        batch_size=64,
        learning_rate=0.001
    )
    
    print("\n=== 集成完成 ===")
    print("模型已训练完成并保存到 ./Experiment/pre_trained_model/Mettack_EdgeFlip/")

def load_and_predict_with_mettack_model():
    """
    加载训练好的模型进行预测的示例
    """
    print("=== 加载预训练模型进行预测 ===")
    
    try:
        # 重新加载数据（用于预测）
        dataset_name = "cora"
        ptb_rate = "005"
        npz_path = f"../DeepRobust/examples/graph/{dataset_name}_edgeflip_dataset_ptbrate{ptb_rate}.npz"
        
        dataset_dict = load_mettack_dataset(npz_path)
        
        # 重建图数据
        X_pairs = dataset_dict['X_pairs']
        pairs = dataset_dict['pairs']
        stats = dataset_dict.get('stats', {})
        n_nodes = stats.get('N_nodes') if stats else pairs.max() + 1
        
        node_features = extract_node_features_from_pairs(X_pairs, pairs, n_nodes)
        graph_data = create_graph_from_pairs(pairs, node_features)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph_data = graph_data.to(device)
        
        # 重新创建模型结构
        model = EdgeFlipMAE(
            gnn_type='GCN',
            dataset_name='Mettack_EdgeFlip',
            input_dim=node_features.shape[1],
            hid_dim=64,
            num_layer=2,
            device=device
        )
        
        # 加载预训练权重
        encoder_path = "./Experiment/pre_trained_model/Mettack_EdgeFlip/EdgeFlipMAE.GCN.64hidden_dim.encoder.pth"
        classifier_path = "./Experiment/pre_trained_model/Mettack_EdgeFlip/EdgeFlipMAE.GCN.64hidden_dim.classifier.pth"
        
        model.load_model(encoder_path, classifier_path)
        
        # 预测示例边
        test_pairs = pairs[:100]  # 取前100条边进行测试
        probs = model.predict_edge_flips(test_pairs, graph_data)
        
        print(f"预测了 {len(test_pairs)} 条边")
        print(f"平均翻转概率: {np.mean(probs):.4f}")
        print(f"高风险边数量 (prob > 0.7): {np.sum(probs > 0.7)}")
        
        # 显示前10条边的预测结果
        print("\n前10条边的预测结果:")
        for i in range(min(10, len(test_pairs))):
            edge = test_pairs[i]
            prob = probs[i]
            true_label = dataset_dict['y'][i]
            print(f"边 ({edge[0]}, {edge[1]}): 翻转概率={prob:.3f}, 真实标签={true_label}")
            
    except FileNotFoundError as e:
        print(f"未找到预训练模型文件: {e}")
        print("请先运行main_integration_example()进行训练")

if __name__ == "__main__":
    # 运行完整的集成示例
    main_integration_example()
    
    # 如果需要测试预测功能，取消下面的注释
    # load_and_predict_with_mettack_model()
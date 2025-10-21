import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

# 导入我们的数据构建和模型
from dataconstruction import load_prog_data, reduce_features_svd, construct_edge_diff_dataset
from EdgeFlipMAE import EdgeFlipMAE

def main():
    """使用EdgeFlipMAE进行边翻转检测的完整示例"""
    
    # 1. 加载数据
    print("=== 加载数据 ===")
    clean_path = "../DeepRobust/examples/graph/tmp/cora.npz"
    attacked_path = "../DeepRobust/examples/graph/tmp/cora_modified_095.npz"
    
    # 加载干净图和攻击图
    adj_clean, features, labels, idx_train, idx_val, idx_test = load_prog_data(clean_path)
    adj_attack, _, _, _, _, _ = load_prog_data(attacked_path)
    
    print(f"图节点数: {adj_clean.shape[0]}, 边数: {adj_clean.nnz // 2}")
    print(f"原始特征维度: {features.shape[1]}")
    
    # 2. 特征降维
    print("\n=== 特征降维 ===")
    X_reduced, svd_model = reduce_features_svd(features, n_components=100, random_state=42)
    print(f"降维后特征维度: {X_reduced.shape[1]}")
    
    # 3. 构造三元组数据集
    print("\n=== 构造边翻转数据集 ===")
    edge_pairs, X_pairs, y = construct_edge_diff_dataset(
        adj_clean, adj_attack, X_reduced, balance=True, random_state=42
    )
    print(f"数据集大小: {len(y)}")
    print(f"正样本(翻转): {np.sum(y)}, 负样本(未翻转): {len(y) - np.sum(y)}")
    
    # 4. 创建PyG图数据
    print("\n=== 创建图数据 ===")
    edge_index, _ = from_scipy_sparse_matrix(adj_clean)
    x_tensor = torch.tensor(X_reduced, dtype=torch.float)
    graph_data = Data(x=x_tensor, edge_index=edge_index)
    graph_data = graph_data.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 5. 初始化EdgeFlipMAE模型
    print("\n=== 初始化模型 ===")
    device_id = 0 if torch.cuda.is_available() else 'cpu'
    
    model = EdgeFlipMAE(
        gnn_type='GCN',
        dataset_name='Cora_EdgeFlip',
        input_dim=X_reduced.shape[1],  # 100
        hid_dim=64,
        num_layer=2,
        device=device_id,
        mask_rate=0.15,  # MAE掩码率
        noise_rate=0.1,  # 噪声替换率
        learning_rate=0.001,
        weight_decay=5e-4,
        epochs=100
    )
    
    # 6. 加载数据到模型
    print("\n=== 加载训练数据 ===")
    model.load_triplet_data(
        edge_pairs=edge_pairs,
        X_pairs=X_pairs,  # 这里实际上不直接使用，因为我们用GNN编码
        labels=y,
        graph_data=graph_data,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # 7. 训练模型
    print("\n=== 开始训练 ===")
    model.pretrain(batch_size=64)
    
    # 8. 测试模型
    print("\n=== 测试模型 ===")
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(model.test_dataset, batch_size=64, shuffle=False)
    test_metrics = model.evaluate(test_loader)
    
    print("测试集结果:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # 9. 下游应用示例：预测新图上的边翻转
    print("\n=== 下游应用示例 ===")
    
    # 假设我们有一个新的图和一些待检测的边
    test_edge_pairs = edge_pairs[:100]  # 取前100个边作为示例
    
    # 预测这些边是否被翻转
    flip_probs = model.predict_edge_flips(test_edge_pairs, graph_data)
    
    print(f"预测了 {len(test_edge_pairs)} 条边")
    print(f"平均翻转概率: {np.mean(flip_probs):.4f}")
    print(f"高风险边数量 (prob > 0.7): {np.sum(flip_probs > 0.7)}")
    
    # 显示一些具体的预测结果
    print("\n前10条边的预测结果:")
    for i in range(min(10, len(test_edge_pairs))):
        edge = test_edge_pairs[i]
        prob = flip_probs[i]
        true_label = y[i]
        print(f"边 ({edge[0]}, {edge[1]}): 翻转概率={prob:.3f}, 真实标签={true_label}")

def load_and_predict_example():
    """加载已训练模型进行预测的示例"""
    
    print("=== 加载预训练模型进行预测 ===")
    
    # 重新创建模型结构
    model = EdgeFlipMAE(
        gnn_type='GCN',
        dataset_name='Cora_EdgeFlip',
        input_dim=100,
        hid_dim=64,
        num_layer=2,
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    
    # 加载预训练权重
    encoder_path = "./Experiment/pre_trained_model/Cora_EdgeFlip/EdgeFlipMAE.GCN.64hidden_dim.encoder.pth"
    classifier_path = "./Experiment/pre_trained_model/Cora_EdgeFlip/EdgeFlipMAE.GCN.64hidden_dim.classifier.pth"
    
    try:
        model.load_model(encoder_path, classifier_path)
        
        # 准备测试数据
        # ... (加载图数据的代码)
        
        # 进行预测
        # test_edges = [(0, 1), (1, 2), (2, 3)]  # 示例边
        # probs = model.predict_edge_flips(test_edges, graph_data)
        # print("预测结果:", probs)
        
    except FileNotFoundError:
        print("未找到预训练模型文件，请先运行训练")

if __name__ == "__main__":
    main()
    # load_and_predict_example()  # 如果已有预训练模型，可以运行这个
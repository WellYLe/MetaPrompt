#这个文件是可行的训练mae的文件
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

# 导入我们的数据构建和模型
from dataconstruction import load_prog_data, reduce_features_svd, construct_edge_diff_dataset
from EdgeFlipMAE import EdgeFlipMAE

def ensure_undirected(adj):
    # 将邻接矩阵对称化，去除自环，并二值化为0/1
    adj = adj.tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1
    return adj

def load_adj_npz(path):
    # 兼容 DeepRobust 数据包格式（adj_* 键）和 scipy.sparse.save_npz 标准格式
    try:
        data = np.load(path, allow_pickle=True)
        if all(k in data for k in ['adj_data','adj_indices','adj_indptr','adj_shape']):
            return sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
                                 shape=tuple(data['adj_shape']))
    except Exception:
        pass
    return sp.load_npz(path).tocsr()

def main():
    """使用EdgeFlipMAE进行边翻转检测的完整示例"""
    
    # 1. 加载数据
    print("=== 加载数据 ===")
    clean_path = "../DeepRobust/examples/graph/tmp/cora.npz"  # 保持与数据集来源一致
    # 将 attacked_path 改为 test_mettack.py 实际保存的位置与命名
    attacked_path = "../DeepRobust/examples/graph/tmp/cora_modified_005.npz"  # 例如 5% 对应 005；请按你的实际文件调整

    # 加载干净图和攻击图（兼容两种 npz 存储格式）
    def load_adj_npz(path):
        try:
            # DeepRobust 数据打包格式（含 adj_* 键）
            data = np.load(path, allow_pickle=True)
            if all(k in data for k in ['adj_data','adj_indices','adj_indptr','adj_shape']):
                adj = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
                                    shape=tuple(data['adj_shape']))
                return adj
        except Exception:
            pass
        try:
            # 标准 scipy.sparse.save_npz 格式
            return sp.load_npz(path).tocsr()
        except Exception as e:
            raise RuntimeError(f"无法加载邻接矩阵: {path}, 错误: {e}")

    from dataconstruction import load_prog_data, reduce_features_svd, construct_edge_diff_dataset
    import scipy.sparse as sp

    adj_clean, features = load_prog_data(clean_path)
    adj_attack = load_adj_npz(attacked_path)

    # 一致性检查：形状与节点顺序
    #assert adj_clean.shape == adj_attack.shape, "攻击图与原图节点数不一致，可能加载了不同数据集或不同版本"
    # 可选：确保都是0/1且对称
    assert (adj_clean != adj_clean.T).nnz == 0 and (adj_attack != adj_attack.T).nnz == 0, "图不是无向或已对称化"
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)
    
    # 5. 初始化EdgeFlipMAE模型
    print("\n=== 初始化模型 ===")
    model = EdgeFlipMAE(
        gnn_type='GCN',
        dataset_name='Cora_EdgeFlip',
        input_dim=X_reduced.shape[1],
        hid_dim=64,
        num_layer=2,
        device=device,  # 统一使用 torch.device
        mask_rate=0.15,
        noise_rate=0.1,
        learning_rate=0.001,
        weight_decay=5e-4,
        epochs=100
    )
    
    # 6. 加载训练数据
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
    
    # 7.1 保存训练好的模型权重
    print("\n=== 保存模型权重 ===")
    model.save_model()
    
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
    # print("\n=== 下游应用示例 ===")
    
    # # 假设我们有一个新的图和一些待检测的边
    # test_edge_pairs = edge_pairs[:100]  # 取前100个边作为示例
    
    # # 预测这些边是否被翻转
    # flip_probs = model.predict_edge_flips(test_edge_pairs, graph_data)
    
    # print(f"预测了 {len(test_edge_pairs)} 条边")
    # print(f"平均翻转概率: {np.mean(flip_probs):.4f}")
    # print(f"高风险边数量 (prob > 0.7): {np.sum(flip_probs > 0.7)}")
    
    # # 显示一些具体的预测结果
    # print("\n前10条边的预测结果:")
    # for i in range(min(10, len(test_edge_pairs))):
    #     edge = test_edge_pairs[i]
    #     prob = flip_probs[i]
    #     true_label = y[i]
    #     print(f"边 ({edge[0]}, {edge[1]}): 翻转概率={prob:.3f}, 真实标签={true_label}")

def load_and_predict_example():
    """加载已训练模型进行预测的示例
    
    模块职责：
    - 编码器 encoder：一个 GNN（GCN/GAT/SAGE 等），输入是 graph_data 的节点特征 x 和图结构 edge_index，
      输出每个节点的低维嵌入表示（node_embeddings）。它负责把节点原始特征在图上进行消息传递和编码。
    - 分类器 edge_classifier：一个 MLP，输入是一条边两端节点嵌入的拼接（[h_u, h_v]），
      输出该边是否为"翻转边"的概率（logits）。它负责边级别的二分类。
    """
    
    print("=== 加载预训练模型进行预测 ===")
    
    try:
        # 1) 重新准备图数据（与训练时一致的处理）
        clean_path = "../DeepRobust/examples/graph/tmp/cora.npz"
        attacked_path = "../DeepRobust/examples/graph/tmp/cora_modified_095.npz"
        
        # 加载干净图与特征；加载攻击图只用于构造示例边对（可选）
        adj_clean, features = load_prog_data(clean_path)
        adj_attack, _ = load_prog_data(attacked_path)
        
        # 与训练保持一致的SVD降维
        X_reduced, _ = reduce_features_svd(features, n_components=100, random_state=42)
        
        # 图数据（使用干净图进行消息传递）
        edge_index, _ = from_scipy_sparse_matrix(adj_clean)
        x_tensor = torch.tensor(X_reduced, dtype=torch.float)
        graph_data = Data(x=x_tensor, edge_index=edge_index)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph_data = graph_data.to(device)
        
        # 2) 构造或选择待预测的边对（示例：使用干净图与攻击图的"翻转边"）
        edge_pairs, _, labels = construct_edge_diff_dataset(
            adj_clean, adj_attack, X_reduced, balance=False, random_state=42
        )
        # 选择前若干条边进行演示
        test_edge_pairs = edge_pairs[:100]
        
        # 3) 重新创建模型结构（配置需与训练时一致）
        model = EdgeFlipMAE(
            gnn_type='GCN',
            dataset_name='Cora_EdgeFlip',
            input_dim=100,
            hid_dim=64,
            num_layer=2,
            device=device
        )
        
        # 4) 加载预训练权重（与 save_model 保存的路径一致）
        encoder_path = "./Experiment/pre_trained_model/Cora_EdgeFlip/EdgeFlipMAE.GCN.64hidden_dim.encoder.pth"
        classifier_path = "./Experiment/pre_trained_model/Cora_EdgeFlip/EdgeFlipMAE.GCN.64hidden_dim.classifier.pth"
        model.load_model(encoder_path, classifier_path)
        
        # 5) 预测这些边是否被翻转
        print("=== 进行预测 ===")
        probs = model.predict_edge_flips(test_edge_pairs, graph_data)
        print(f"预测了 {len(test_edge_pairs)} 条边")
        print(f"平均翻转概率: {np.mean(probs):.4f}")
        # 可与真实标签对比（如有）
        if labels is not None:
            print(f"示例标签对比：前10条 -> 预测概率 vs 真实标签")
            for i in range(min(10, len(test_edge_pairs))):
                print(f"Edge {tuple(test_edge_pairs[i])}: prob={probs[i]:.3f}, label={labels[i]}")
        
    except FileNotFoundError:
        print("未找到预训练模型文件，请先运行训练")

if __name__ == "__main__":
    main()
    # load_and_predict_example()  # 如果已有预训练模型，可以运行这个
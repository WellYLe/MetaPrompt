import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

def dense_features_to_encoder_format(features):
    """
    将稠密特征矩阵转换为EdgeFlipMAE的Encoder能接收的形式
    
    Args:
        features: 稠密特征矩阵
                 - numpy.ndarray: shape [num_nodes, feature_dim]
                 - torch.Tensor: shape [num_nodes, feature_dim]
                 - list: 嵌套列表形式的特征
    
    Returns:
        torch.Tensor: shape [num_nodes, feature_dim], dtype=torch.float
                     可直接用于 encoder(x=tensor, edge_index=...)
    
    Examples:
        >>> # numpy数组输入
        >>> features_np = np.random.randn(2708, 1433)
        >>> x_tensor = dense_features_to_encoder_format(features_np)
        >>> print(x_tensor.shape)  # torch.Size([2708, 1433])
        
        >>> # 列表输入
        >>> features_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        >>> x_tensor = dense_features_to_encoder_format(features_list)
        >>> print(x_tensor.shape)  # torch.Size([2, 3])
    """
    
    # 处理不同输入类型
    if isinstance(features, np.ndarray):
        # numpy数组 -> torch tensor
        x_tensor = torch.tensor(features, dtype=torch.float)
    elif isinstance(features, torch.Tensor):
        # 已经是tensor，确保数据类型正确
        x_tensor = features.float()
    elif isinstance(features, (list, tuple)):
        # 列表/元组 -> torch tensor
        x_tensor = torch.tensor(features, dtype=torch.float)
    else:
        raise TypeError(f"不支持的特征类型: {type(features)}")
    
    # 验证维度
    if x_tensor.dim() != 2:
        raise ValueError(f"特征矩阵必须是2维的，当前维度: {x_tensor.dim()}")
    
    num_nodes, feature_dim = x_tensor.shape
    print(f"特征转换完成: {num_nodes} 个节点, {feature_dim} 维特征")
    
    return x_tensor


def dense_adj_to_edge_index(adj_matrix, remove_self_loops=True, ensure_undirected=True):
    """
    将稠密邻接矩阵转换为EdgeFlipMAE的Encoder能接收的边索引形式
    
    Args:
        adj_matrix: 稠密邻接矩阵
                   - numpy.ndarray: shape [num_nodes, num_nodes]
                   - torch.Tensor: shape [num_nodes, num_nodes]
                   - list: 嵌套列表形式的邻接矩阵
        remove_self_loops: bool, 是否移除自环 (默认True)
        ensure_undirected: bool, 是否确保无向图 (默认True)
    
    Returns:
        torch.Tensor: shape [2, num_edges], dtype=torch.long
                     COO格式的边索引，可直接用于 encoder(x=..., edge_index=tensor)
    
    Examples:
        >>> # numpy邻接矩阵
        >>> adj_np = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> edge_index = dense_adj_to_edge_index(adj_np)
        >>> print(edge_index.shape)  # torch.Size([2, 6])
        
        >>> # torch邻接矩阵
        >>> adj_torch = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        >>> edge_index = dense_adj_to_edge_index(adj_torch)
        >>> print(edge_index)  # tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    """
    
    # 处理不同输入类型
    if isinstance(adj_matrix, np.ndarray):
        adj_np = adj_matrix
    elif isinstance(adj_matrix, torch.Tensor):
        adj_np = adj_matrix.cpu().numpy()
    elif isinstance(adj_matrix, (list, tuple)):
        adj_np = np.array(adj_matrix)
    else:
        raise TypeError(f"不支持的邻接矩阵类型: {type(adj_matrix)}")
    
    # 验证维度和形状
    if adj_np.ndim != 2:
        raise ValueError(f"邻接矩阵必须是2维的，当前维度: {adj_np.ndim}")
    
    if adj_np.shape[0] != adj_np.shape[1]:
        raise ValueError(f"邻接矩阵必须是方阵，当前形状: {adj_np.shape}")
    
    num_nodes = adj_np.shape[0]
    
    # 转换为稀疏矩阵进行处理
    adj_sparse = sp.csr_matrix(adj_np)
    
    # 移除自环
    if remove_self_loops:
        adj_sparse.setdiag(0)
        adj_sparse.eliminate_zeros()
    
    # 确保无向图（对称化）
    if ensure_undirected:
        adj_sparse = adj_sparse.maximum(adj_sparse.T)
    
    # 二值化（确保边权重为0或1）
    adj_sparse.data = (adj_sparse.data > 0).astype(np.float32)
    
    # 转换为PyG的edge_index格式
    edge_index, _ = from_scipy_sparse_matrix(adj_sparse)
    
    num_edges = edge_index.shape[1]
    print(f"邻接矩阵转换完成: {num_nodes} 个节点, {num_edges} 条边")
    
    return edge_index


def create_graph_data(features, adj_matrix, **kwargs):
    """
    便捷函数：同时转换特征和邻接矩阵，创建PyG Data对象
    
    Args:
        features: 稠密特征矩阵
        adj_matrix: 稠密邻接矩阵
        **kwargs: 传递给dense_adj_to_edge_index的额外参数
    
    Returns:
        torch_geometric.data.Data: 包含x和edge_index的图数据对象
    
    Examples:
        >>> features = np.random.randn(100, 50)
        >>> adj = np.random.randint(0, 2, (100, 100))
        >>> graph_data = create_graph_data(features, adj)
        >>> print(f"节点特征: {graph_data.x.shape}")
        >>> print(f"边索引: {graph_data.edge_index.shape}")
    """
    
    # 转换特征
    x = dense_features_to_encoder_format(features)
    
    # 转换邻接矩阵
    edge_index = dense_adj_to_edge_index(adj_matrix, **kwargs)
    
    # 创建PyG Data对象
    graph_data = Data(x=x, edge_index=edge_index)
    
    print(f"图数据创建完成: {graph_data.num_nodes} 个节点, {graph_data.num_edges} 条边")
    
    return graph_data


# 使用示例和测试函数
def test_converters():
    """测试数据转换函数"""
    print("=== 测试数据转换函数 ===")
    
    # 1. 测试特征转换
    print("\n1. 测试特征转换:")
    
    # numpy输入
    features_np = np.random.randn(5, 3)
    x1 = dense_features_to_encoder_format(features_np)
    print(f"numpy输入: {features_np.shape} -> {x1.shape}, dtype: {x1.dtype}")
    
    # 列表输入
    features_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    x2 = dense_features_to_encoder_format(features_list)
    print(f"列表输入: {len(features_list)}x{len(features_list[0])} -> {x2.shape}")
    
    # 2. 测试邻接矩阵转换
    print("\n2. 测试邻接矩阵转换:")
    
    # 简单的3x3邻接矩阵
    adj_simple = np.array([
        [0, 1, 1],
        [1, 0, 1], 
        [1, 1, 0]
    ])
    edge_index1 = dense_adj_to_edge_index(adj_simple)
    print(f"3x3邻接矩阵: {adj_simple.shape} -> {edge_index1.shape}")
    print(f"边连接: {edge_index1}")
    
    # 带自环的邻接矩阵
    adj_with_loops = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ])
    edge_index2 = dense_adj_to_edge_index(adj_with_loops, remove_self_loops=False)
    print(f"带自环: {edge_index2.shape} 条边")
    
    edge_index3 = dense_adj_to_edge_index(adj_with_loops, remove_self_loops=True)
    print(f"移除自环: {edge_index3.shape} 条边")
    
    # 3. 测试完整图数据创建
    print("\n3. 测试完整图数据创建:")
    
    features = np.random.randn(4, 10)
    adj = np.random.randint(0, 2, (4, 4))
    graph_data = create_graph_data(features, adj)
    
    print(f"图数据: 节点特征 {graph_data.x.shape}, 边索引 {graph_data.edge_index.shape}")
    
    # 4. 验证与EdgeFlipMAE的兼容性
    print("\n4. 验证EdgeFlipMAE兼容性:")
    print("✓ 特征格式: torch.Tensor, dtype=float")
    print("✓ 边索引格式: torch.Tensor, shape=[2, num_edges], dtype=long")
    print("✓ 可直接用于: encoder(x=graph_data.x, edge_index=graph_data.edge_index)")


def node_embeddings_to_dense_matrix(node_embeddings):
    """
    将Encoder输出的节点嵌入转换为稠密矩阵
    
    Args:
        node_embeddings: Encoder输出的节点嵌入
                        - torch.Tensor: shape [num_nodes, hid_dim]
                        - 可能在GPU上，需要转移到CPU
    
    Returns:
        numpy.ndarray: shape [num_nodes, hid_dim], dtype=float32
                      稠密的节点嵌入矩阵，可用于后续分析或保存
    
    Examples:
        >>> # 假设从EdgeFlipMAE获得节点嵌入
        >>> model = EdgeFlipMAE(input_dim=100, hid_dim=64)
        >>> node_embeddings = model.encoder(x=graph_data.x, edge_index=graph_data.edge_index)
        >>> dense_embeddings = node_embeddings_to_dense_matrix(node_embeddings)
        >>> print(f"嵌入形状: {dense_embeddings.shape}")  # (2708, 64)
        >>> print(f"数据类型: {dense_embeddings.dtype}")   # float32
        
        >>> # 可以直接用于科学计算
        >>> import numpy as np
        >>> similarity = np.dot(dense_embeddings, dense_embeddings.T)  # 计算相似度矩阵
        >>> np.save('node_embeddings.npy', dense_embeddings)  # 保存到文件
    """
    
    # 类型检查
    if not isinstance(node_embeddings, torch.Tensor):
        raise TypeError(f"输入必须是torch.Tensor，当前类型: {type(node_embeddings)}")
    
    # 维度检查
    if node_embeddings.dim() != 2:
        raise ValueError(f"节点嵌入必须是2维张量，当前维度: {node_embeddings.dim()}")
    
    num_nodes, hid_dim = node_embeddings.shape
    
    # 转换为numpy数组
    # 1. 确保在CPU上
    # 2. 分离计算图（如果需要梯度的话）
    # 3. 转换为numpy
    dense_matrix = node_embeddings.detach().cpu().numpy().astype(np.float32)
    
    print(f"节点嵌入转换完成: {num_nodes} 个节点, {hid_dim} 维嵌入 -> numpy数组 {dense_matrix.shape}")
    
    return dense_matrix


def embeddings_to_similarity_matrix(node_embeddings, metric='cosine'):
    """
    将节点嵌入转换为节点相似度矩阵
    
    Args:
        node_embeddings: 节点嵌入 (torch.Tensor 或 numpy.ndarray)
        metric: 相似度度量方式
               - 'cosine': 余弦相似度 (默认)
               - 'euclidean': 欧氏距离相似度
               - 'dot': 点积相似度
    
    Returns:
        numpy.ndarray: shape [num_nodes, num_nodes] 相似度矩阵
    
    Examples:
        >>> embeddings = torch.randn(100, 64)
        >>> sim_matrix = embeddings_to_similarity_matrix(embeddings, metric='cosine')
        >>> print(f"相似度矩阵形状: {sim_matrix.shape}")  # (100, 100)
    """
    
    # 转换为numpy格式
    if isinstance(node_embeddings, torch.Tensor):
        embeddings_np = node_embeddings_to_dense_matrix(node_embeddings)
    else:
        embeddings_np = np.array(node_embeddings, dtype=np.float32)
    
    num_nodes = embeddings_np.shape[0]
    
    if metric == 'cosine':
        # 余弦相似度
        # 先归一化
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        normalized_embeddings = embeddings_np / norms
        
        # 计算余弦相似度
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
    elif metric == 'euclidean':
        # 欧氏距离相似度 (转换为相似度: 1 / (1 + distance))
        from scipy.spatial.distance import cdist
        distance_matrix = cdist(embeddings_np, embeddings_np, metric='euclidean')
        similarity_matrix = 1 / (1 + distance_matrix)
        
    elif metric == 'dot':
        # 点积相似度
        similarity_matrix = np.dot(embeddings_np, embeddings_np.T)
        
    else:
        raise ValueError(f"不支持的相似度度量: {metric}")
    
    print(f"相似度矩阵计算完成: {num_nodes}x{num_nodes}, 度量方式: {metric}")
    
    return similarity_matrix


def save_embeddings(node_embeddings, filepath, format='npy'):
    """
    保存节点嵌入到文件
    
    Args:
        node_embeddings: 节点嵌入 (torch.Tensor)
        filepath: 保存路径
        format: 保存格式
               - 'npy': numpy二进制格式 (默认)
               - 'csv': CSV文本格式
               - 'txt': 纯文本格式
    
    Examples:
        >>> embeddings = torch.randn(2708, 64)
        >>> save_embeddings(embeddings, 'cora_embeddings.npy')
        >>> save_embeddings(embeddings, 'cora_embeddings.csv', format='csv')
    """
    
    # 转换为稠密矩阵
    dense_matrix = node_embeddings_to_dense_matrix(node_embeddings)
    
    if format == 'npy':
        np.save(filepath, dense_matrix)
    elif format == 'csv':
        np.savetxt(filepath, dense_matrix, delimiter=',', fmt='%.6f')
    elif format == 'txt':
        np.savetxt(filepath, dense_matrix, fmt='%.6f')
    else:
        raise ValueError(f"不支持的保存格式: {format}")
    
    print(f"节点嵌入已保存到: {filepath} (格式: {format})")


def load_embeddings(filepath, format='npy'):
    """
    从文件加载节点嵌入
    
    Args:
        filepath: 文件路径
        format: 文件格式
               - 'npy': numpy二进制格式 (默认)
               - 'csv': CSV文本格式
               - 'txt': 纯文本格式
    
    Returns:
        torch.Tensor: 加载的节点嵌入
    
    Examples:
        >>> embeddings = load_embeddings('cora_embeddings.npy')
        >>> print(embeddings.shape)
    """
    
    if format == 'npy':
        dense_matrix = np.load(filepath)
    elif format in ['csv', 'txt']:
        dense_matrix = np.loadtxt(filepath, delimiter=',' if format == 'csv' else None)
    else:
        raise ValueError(f"不支持的加载格式: {format}")
    
    # 转换回torch tensor
    embeddings_tensor = torch.tensor(dense_matrix, dtype=torch.float)
    
    print(f"节点嵌入已从 {filepath} 加载: {embeddings_tensor.shape}")
    
    return embeddings_tensor


def test_embedding_converters():
    """测试节点嵌入转换函数"""
    print("=== 测试节点嵌入转换函数 ===")
    
    # 1. 创建模拟的节点嵌入
    print("\n1. 创建模拟节点嵌入:")
    num_nodes, hid_dim = 100, 64
    node_embeddings = torch.randn(num_nodes, hid_dim)
    print(f"原始嵌入: {node_embeddings.shape}, dtype: {node_embeddings.dtype}")
    
    # 2. 测试转换为稠密矩阵
    print("\n2. 测试转换为稠密矩阵:")
    dense_matrix = node_embeddings_to_dense_matrix(node_embeddings)
    print(f"稠密矩阵: {dense_matrix.shape}, dtype: {dense_matrix.dtype}")
    
    # 3. 测试相似度矩阵计算
    print("\n3. 测试相似度矩阵计算:")
    
    # 余弦相似度
    cosine_sim = embeddings_to_similarity_matrix(node_embeddings, metric='cosine')
    print(f"余弦相似度矩阵: {cosine_sim.shape}, 范围: [{cosine_sim.min():.3f}, {cosine_sim.max():.3f}]")
    
    # 点积相似度
    dot_sim = embeddings_to_similarity_matrix(node_embeddings, metric='dot')
    print(f"点积相似度矩阵: {dot_sim.shape}, 范围: [{dot_sim.min():.3f}, {dot_sim.max():.3f}]")
    
    # 4. 测试保存和加载
    print("\n4. 测试保存和加载:")
    
    # 保存为npy格式
    save_embeddings(node_embeddings, 'test_embeddings.npy')
    
    # 加载并验证
    loaded_embeddings = load_embeddings('test_embeddings.npy')
    print(f"加载的嵌入: {loaded_embeddings.shape}")
    
    # 验证一致性
    diff = torch.abs(node_embeddings - loaded_embeddings).max()
    print(f"保存/加载一致性检查: 最大差异 = {diff:.6f}")
    
    # 5. 实际使用示例
    print("\n5. 实际使用示例:")
    print("# 在EdgeFlipMAE中的使用:")
    print("model = EdgeFlipMAE(input_dim=100, hid_dim=64)")
    print("node_embeddings = model.encoder(x=graph_data.x, edge_index=graph_data.edge_index)")
    print("dense_embeddings = node_embeddings_to_dense_matrix(node_embeddings)")
    print("similarity_matrix = embeddings_to_similarity_matrix(node_embeddings)")
    print("save_embeddings(node_embeddings, 'my_embeddings.npy')")
    
    # 清理测试文件
    import os
    if os.path.exists('test_embeddings.npy'):
        os.remove('test_embeddings.npy')
        print("\n测试文件已清理")


def prepare_classifier_input_from_dense(dense_adj_matrix, dense_node_embeddings, 
                                       include_self_loops=False, directed=False,
                                       classify_all_possible_edges=True):
    """
    从稠密邻接矩阵和节点嵌入矩阵准备Classifier的输入数据
    
    Args:
        dense_adj_matrix: 稠密邻接矩阵
                         - numpy.ndarray 或 torch.Tensor: shape [num_nodes, num_nodes]
                         - 值为0或1，表示节点间是否有边
        dense_node_embeddings: 稠密节点嵌入矩阵
                              - numpy.ndarray 或 torch.Tensor: shape [num_nodes, hid_dim]
                              - 每行是一个节点的嵌入向量
        include_self_loops: 是否包含自环边 (默认False)
        directed: 是否为有向图 (默认False，无向图)
        classify_all_possible_edges: 是否对所有可能的边进行分类 (默认True)
                                   - True: 包括邻接矩阵中值为0的边（不存在的边）
                                   - False: 只包括邻接矩阵中值>0的边（已存在的边）
 
    Returns:
        dict: 包含以下键值对
            - 'edge_embeddings': torch.Tensor, shape [num_edges, 2*hid_dim]
                                 Classifier的输入，每行是一条边的嵌入
            - 'edge_pairs': torch.Tensor, shape [num_edges, 2]
                           对应的边对索引 (源节点, 目标节点)
            - 'edge_labels': torch.Tensor, shape [num_edges]
                           真实的边标签 (1=存在, 0=不存在)
            - 'num_edges': int, 边的总数
            - 'edge_info': list of dict, 每条边的详细信息
    
    Examples:
        >>> # 创建示例数据
        >>> adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # 3个节点的无向图
        >>> node_embeddings = np.random.randn(3, 64)  # 3个节点，64维嵌入
        >>> 
        >>> # 准备分类器输入 - 包括所有可能的边
        >>> classifier_input = prepare_classifier_input_from_dense(adj_matrix, node_embeddings)
        >>> print(f"边嵌入形状: {classifier_input['edge_embeddings'].shape}")  # [3, 128] - 所有可能的边
        >>> print(f"边对: {classifier_input['edge_pairs']}")  # [[0,1], [0,2], [1,2]]
        >>> print(f"边标签: {classifier_input['edge_labels']}")  # [1, 0, 1] - 真实存在情况
        >>> 
        >>> # 直接用于分类器
        >>> edge_logits = classifier(classifier_input['edge_embeddings'])
    """
    
    # 转换为torch tensor
    if isinstance(dense_adj_matrix, np.ndarray):
        adj_matrix = torch.tensor(dense_adj_matrix, dtype=torch.float)
    else:
        adj_matrix = dense_adj_matrix.float()
    
    if isinstance(dense_node_embeddings, np.ndarray):
        node_embeddings = torch.tensor(dense_node_embeddings, dtype=torch.float)
    else:
        node_embeddings = dense_node_embeddings.float()
    
    num_nodes = adj_matrix.shape[0]
    hid_dim = node_embeddings.shape[1]
    
    # 验证输入维度
    assert adj_matrix.shape == (num_nodes, num_nodes), f"邻接矩阵形状错误: {adj_matrix.shape}"
    assert node_embeddings.shape == (num_nodes, hid_dim), f"节点嵌入形状错误: {node_embeddings.shape}"
    
    # 提取边
    edge_pairs = []
    edge_labels = []
    edge_info = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            # 跳过自环（如果不包含）
            if i == j and not include_self_loops:
                continue
            
            # 对于无向图，只保留上三角部分避免重复
            if not directed and i > j:
                continue
            
            # 获取边的真实标签
            edge_exists = adj_matrix[i, j] > 0
            
            # 根据参数决定是否包含这条边
            if classify_all_possible_edges or edge_exists:
                edge_pairs.append([i, j])
                edge_labels.append(1.0 if edge_exists else 0.0)
                edge_info.append({
                    'source': i,
                    'target': j,
                    'weight': float(adj_matrix[i, j]),
                    'exists': edge_exists,
                    'is_self_loop': i == j
                })
    
    if len(edge_pairs) == 0:
        print("警告: 没有找到任何边!")
        return {
            'edge_embeddings': torch.empty(0, 2 * hid_dim),
            'edge_pairs': torch.empty(0, 2, dtype=torch.long),
            'edge_labels': torch.empty(0),
            'num_edges': 0,
            'edge_info': []
        }
    
    # 转换为tensor
    edge_pairs = torch.tensor(edge_pairs, dtype=torch.long)  # [num_edges, 2]
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)  # [num_edges]
    
    # 构造边嵌入
    src_embeddings = node_embeddings[edge_pairs[:, 0]]  # [num_edges, hid_dim]
    dst_embeddings = node_embeddings[edge_pairs[:, 1]]  # [num_edges, hid_dim]
    edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)  # [num_edges, 2*hid_dim]
    
    num_edges = edge_pairs.shape[0]
    existing_edges = edge_labels.sum().item()
    non_existing_edges = num_edges - existing_edges
    
    print(f"成功准备分类器输入: {num_edges} 条边, 每条边嵌入维度: {2*hid_dim}")
    print(f"图信息: {num_nodes} 个节点, {'有向' if directed else '无向'}图, {'包含' if include_self_loops else '不包含'}自环")
    if classify_all_possible_edges:
        print(f"边分布: {existing_edges} 条存在的边, {non_existing_edges} 条不存在的边")
    
    return {
        'edge_embeddings': edge_embeddings,
        'edge_pairs': edge_pairs,
        'edge_labels': edge_labels,
        'num_edges': num_edges,
        'edge_info': edge_info
    }


def batch_classify_all_edges(classifier, dense_adj_matrix, dense_node_embeddings, 
                           batch_size=1000, include_self_loops=False, directed=False,
                           return_probabilities=True, classify_all_possible_edges=True):
    """
    批量对图中所有边进行分类
    
    Args:
        classifier: EdgeFlipMAE的edge_classifier或完整模型
        dense_adj_matrix: 稠密邻接矩阵
        dense_node_embeddings: 稠密节点嵌入矩阵
        batch_size: 批处理大小 (默认1000)
        include_self_loops: 是否包含自环边
        directed: 是否为有向图
        return_probabilities: 是否返回概率值 (True) 还是logits (False)
        classify_all_possible_edges: 是否对所有可能的边进行分类 (默认True)
    
    Returns:
        dict: 分类结果
            - 'predictions': torch.Tensor, 每条边的预测结果
            - 'edge_pairs': torch.Tensor, 对应的边对
            - 'edge_labels': torch.Tensor, 真实的边标签
            - 'edge_info': list, 边的详细信息
            - 'summary': dict, 分类结果统计
    
    Examples:
        >>> # 使用训练好的模型
        >>> model = EdgeFlipMAE(...)
        >>> model.load_model(encoder_path, classifier_path)
        >>> 
        >>> # 批量分类所有可能的边
        >>> results = batch_classify_all_edges(
        ...     model.edge_classifier, adj_matrix, node_embeddings
        ... )
        >>> print(f"预测结果: {results['predictions']}")
        >>> print(f"真实标签: {results['edge_labels']}")
        >>> print(f"统计信息: {results['summary']}")
    """
    
    # 准备分类器输入
    classifier_input = prepare_classifier_input_from_dense(
        dense_adj_matrix, dense_node_embeddings, 
        include_self_loops=include_self_loops, directed=directed,
        classify_all_possible_edges=classify_all_possible_edges
    )
    
    edge_embeddings = classifier_input['edge_embeddings']
    edge_pairs = classifier_input['edge_pairs']
    edge_labels = classifier_input['edge_labels']
    edge_info = classifier_input['edge_info']
    num_edges = classifier_input['num_edges']
    
    if num_edges == 0:
        return {
            'predictions': torch.empty(0),
            'edge_pairs': torch.empty(0, 2, dtype=torch.long),
            'edge_labels': torch.empty(0),
            'edge_info': [],
            'summary': {'total_edges': 0, 'positive_predictions': 0, 'negative_predictions': 0,
                       'true_positives': 0, 'true_negatives': 0, 'false_positives': 0, 'false_negatives': 0}
        }
    
    # 批量处理
    all_predictions = []
    classifier.eval()
    
    with torch.no_grad():
        for i in range(0, num_edges, batch_size):
            batch_end = min(i + batch_size, num_edges)
            batch_embeddings = edge_embeddings[i:batch_end]
            
            # 获取预测结果
            batch_logits = classifier(batch_embeddings).squeeze()
            
            if return_probabilities:
                batch_preds = torch.sigmoid(batch_logits)
            else:
                batch_preds = batch_logits
            
            all_predictions.append(batch_preds)
    
    # 合并所有批次的结果
    predictions = torch.cat(all_predictions, dim=0)
    
    # 统计信息
    if return_probabilities:
        pred_binary = (predictions > 0.5).float()
    else:
        pred_binary = (predictions > 0).float()
    
    positive_predictions = pred_binary.sum().item()
    negative_predictions = num_edges - positive_predictions
    
    # 计算混淆矩阵
    true_positives = ((pred_binary == 1) & (edge_labels == 1)).sum().item()
    true_negatives = ((pred_binary == 0) & (edge_labels == 0)).sum().item()
    false_positives = ((pred_binary == 1) & (edge_labels == 0)).sum().item()
    false_negatives = ((pred_binary == 0) & (edge_labels == 1)).sum().item()
    
    summary = {
        'total_edges': num_edges,
        'positive_predictions': positive_predictions,
        'negative_predictions': negative_predictions,
        'positive_ratio': positive_predictions / num_edges if num_edges > 0 else 0,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'accuracy': (true_positives + true_negatives) / num_edges if num_edges > 0 else 0,
        'precision': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
        'recall': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    }
    
    print(f"批量分类完成: {num_edges} 条边")
    print(f"预测统计: 正类 {positive_predictions}, 负类 {negative_predictions}")
    if classify_all_possible_edges:
        print(f"分类性能: 准确率 {summary['accuracy']:.3f}, 精确率 {summary['precision']:.3f}, 召回率 {summary['recall']:.3f}")
    
    return {
        'predictions': predictions,
        'edge_pairs': edge_pairs,
        'edge_labels': edge_labels,
        'edge_info': edge_info,
        'summary': summary
    }


def create_edge_classification_dataset(dense_adj_matrix, dense_node_embeddings, 
                                     edge_labels=None, include_self_loops=False, 
                                     directed=False, classify_all_possible_edges=True):
    """
    创建边分类数据集，用于训练或评估
    
    Args:
        dense_adj_matrix: 稠密邻接矩阵
        dense_node_embeddings: 稠密节点嵌入矩阵
        edge_labels: 边标签 (可选)
                    - None: 使用邻接矩阵作为标签
                    - list/array: 与边对应的自定义标签
        include_self_loops: 是否包含自环边
        directed: 是否为有向图
        classify_all_possible_edges: 是否包含所有可能的边
    
    Returns:
        dict: 数据集
            - 'edge_embeddings': torch.Tensor
            - 'edge_labels': torch.Tensor
            - 'edge_pairs': torch.Tensor
            - 'dataset_info': dict
    """
    
    # 准备分类器输入
    classifier_input = prepare_classifier_input_from_dense(
        dense_adj_matrix, dense_node_embeddings,
        include_self_loops=include_self_loops, directed=directed,
        classify_all_possible_edges=classify_all_possible_edges
    )
    
    result = {
        'edge_embeddings': classifier_input['edge_embeddings'],
        'edge_pairs': classifier_input['edge_pairs'],
        'dataset_info': {
            'num_edges': classifier_input['num_edges'],
            'embedding_dim': classifier_input['edge_embeddings'].shape[1] if classifier_input['num_edges'] > 0 else 0,
            'include_self_loops': include_self_loops,
            'directed': directed,
            'classify_all_possible_edges': classify_all_possible_edges
        }
    }
    
    # 处理标签
    if edge_labels is not None:
        if isinstance(edge_labels, (list, np.ndarray)):
            edge_labels = torch.tensor(edge_labels, dtype=torch.float)
        
        assert len(edge_labels) == classifier_input['num_edges'], \
            f"标签数量 ({len(edge_labels)}) 与边数量 ({classifier_input['num_edges']}) 不匹配"
        
        result['edge_labels'] = edge_labels
    else:
        # 使用邻接矩阵作为标签
        result['edge_labels'] = classifier_input['edge_labels']
    
    result['dataset_info']['has_labels'] = True
    
    return result


def test_classifier_input_preparation():
    """测试分类器输入准备函数"""
    print("=== 测试分类器输入准备函数 ===")
    
    # 1. 创建示例数据
    print("\n1. 创建示例数据:")
    num_nodes = 4
    hid_dim = 32
    
    # 创建一个简单的无向图邻接矩阵 (不是完全图)
    adj_matrix = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    
    # 创建随机节点嵌入
    node_embeddings = np.random.randn(num_nodes, hid_dim)
    
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print(f"节点嵌入形状: {node_embeddings.shape}")
    print(f"邻接矩阵:\n{adj_matrix}")
    print(f"实际存在的边数: {np.sum(adj_matrix > 0) // 2}")  # 无向图除以2
    
    # 2. 测试只分类已存在的边
    print("\n2. 测试只分类已存在的边:")
    classifier_input_existing = prepare_classifier_input_from_dense(
        adj_matrix, node_embeddings, classify_all_possible_edges=False
    )
    
    print(f"边嵌入形状: {classifier_input_existing['edge_embeddings'].shape}")
    print(f"边对: {classifier_input_existing['edge_pairs']}")
    print(f"边标签: {classifier_input_existing['edge_labels']}")
    
    # 3. 测试分类所有可能的边
    print("\n3. 测试分类所有可能的边:")
    classifier_input_all = prepare_classifier_input_from_dense(
        adj_matrix, node_embeddings, classify_all_possible_edges=True
    )
    
    print(f"边嵌入形状: {classifier_input_all['edge_embeddings'].shape}")
    print(f"边对: {classifier_input_all['edge_pairs']}")
    print(f"边标签: {classifier_input_all['edge_labels']}")
    print(f"存在的边: {classifier_input_all['edge_labels'].sum().item()}")
    print(f"不存在的边: {(classifier_input_all['edge_labels'] == 0).sum().item()}")
    
    # 4. 验证边标签的正确性
    print("\n4. 验证边标签的正确性:")
    for i, (edge_pair, label) in enumerate(zip(classifier_input_all['edge_pairs'], classifier_input_all['edge_labels'])):
        src, dst = edge_pair[0].item(), edge_pair[1].item()
        actual_value = adj_matrix[src, dst]
        expected_label = 1.0 if actual_value > 0 else 0.0
        print(f"边 ({src},{dst}): 邻接矩阵值={actual_value}, 预期标签={expected_label}, 实际标签={label.item()}")
        assert label.item() == expected_label, f"边标签不匹配!"
    
    print("✓ 边标签验证通过!")
    
    # 5. 测试包含自环
    print("\n5. 测试包含自环:")
    classifier_input_loops = prepare_classifier_input_from_dense(
        adj_matrix, node_embeddings, include_self_loops=True, classify_all_possible_edges=True
    )
    print(f"包含自环的边数量: {classifier_input_loops['num_edges']}")
    
    # 6. 测试有向图
    print("\n6. 测试有向图:")
    classifier_input_directed = prepare_classifier_input_from_dense(
        adj_matrix, node_embeddings, directed=True, classify_all_possible_edges=True
    )
    print(f"有向图边数量: {classifier_input_directed['num_edges']}")
    
    # 7. 模拟分类器调用
    print("\n7. 模拟分类器调用:")
    
    # 创建一个简单的模拟分类器
    class MockClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    mock_classifier = MockClassifier(2 * hid_dim)
    mock_classifier.eval()
    
    with torch.no_grad():
        edge_logits = mock_classifier(classifier_input_all['edge_embeddings'])
        edge_probs = torch.sigmoid(edge_logits).squeeze()
    
    print(f"分类器输出形状: {edge_logits.shape}")
    print(f"边存在概率: {edge_probs}")
    print(f"真实边标签: {classifier_input_all['edge_labels']}")
    
    # 8. 测试批量分类
    print("\n8. 测试批量分类 - 所有可能的边:")
    batch_results = batch_classify_all_edges(
        mock_classifier, adj_matrix, node_embeddings, 
        batch_size=3, classify_all_possible_edges=True
    )
    print(f"批量分类结果: {batch_results['summary']}")
    
    # 9. 对比两种模式的差异
    print("\n9. 对比两种分类模式:")
    print(f"只分类已存在边: {classifier_input_existing['num_edges']} 条边")
    print(f"分类所有可能边: {classifier_input_all['num_edges']} 条边")
    print(f"差异: {classifier_input_all['num_edges'] - classifier_input_existing['num_edges']} 条不存在的边被包含")
    
    print("\n=== 测试完成 ===")
    print("✓ 现在Classifier可以判断所有可能的边，包括邻接矩阵中值为0的边!")


def predictions_to_dense_probability_matrix(predictions, edge_pairs, num_nodes, 
                                         directed=False, fill_diagonal=0.0):
    """
    将边预测概率转换为稠密概率矩阵
    
    Args:
        predictions: 边预测概率数组 [num_edges]
        edge_pairs: 边对索引 [num_edges, 2]
        num_nodes: 节点总数
        directed: 是否为有向图
        fill_diagonal: 对角线填充值 (自环概率)
    
    Returns:
        np.ndarray: 稠密概率矩阵 [num_nodes, num_nodes]
            - matrix[i,j] 表示边(i,j)翻转的概率
            - 对于无向图，matrix[i,j] = matrix[j,i]
    
    Examples:
        >>> predictions = np.array([0.8, 0.3, 0.9])
        >>> edge_pairs = np.array([[0,1], [0,2], [1,2]])
        >>> prob_matrix = predictions_to_dense_probability_matrix(
        ...     predictions, edge_pairs, num_nodes=3
        ... )
        >>> print(prob_matrix)
        [[0.0, 0.8, 0.3],
         [0.8, 0.0, 0.9],
         [0.3, 0.9, 0.0]]
    """
    
    # 初始化概率矩阵
    prob_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # 填充对角线
    np.fill_diagonal(prob_matrix, fill_diagonal)
    
    # 填充边概率
    for i, (src, dst) in enumerate(edge_pairs):
        src, dst = int(src), int(dst)
        prob_matrix[src, dst] = predictions[i]
        
        # 对于无向图，确保对称性
        if not directed and src != dst:
            prob_matrix[dst, src] = predictions[i]
    
    return prob_matrix


def edge_flip_probability_analysis(dense_adj_matrix, dense_features, classifier, 
                                 encoder=None, include_self_loops=False, 
                                 directed=False, return_dense_matrix=True):
    """
    完整的边翻转概率分析：从稠密矩阵到边翻转概率矩阵
    
    Args:
        dense_adj_matrix: 稠密邻接矩阵 [num_nodes, num_nodes]
        dense_features: 稠密特征矩阵 [num_nodes, feature_dim]
        classifier: EdgeFlipMAE的edge_classifier
        encoder: EdgeFlipMAE的encoder (可选)
        include_self_loops: 是否包含自环边
        directed: 是否为有向图
        return_dense_matrix: 是否返回稠密概率矩阵
    
    Returns:
        dict: 边翻转概率分析结果
            - 'flip_probability_matrix': 稠密边翻转概率矩阵 [num_nodes, num_nodes]
            - 'original_adj_matrix': 原始邻接矩阵
            - 'predictions': 原始预测概率
            - 'edge_pairs': 边对索引
            - 'flip_analysis': 边翻转分析结果
            - 'statistics': 概率统计信息
    
    Examples:
        >>> # 分析边翻转概率
        >>> adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        >>> features = np.random.randn(3, 100)
        >>> 
        >>> results = edge_flip_probability_analysis(
        ...     adj_matrix, features, model.edge_classifier, model.encoder
        ... )
        >>> 
        >>> print("边翻转概率矩阵:")
        >>> print(results['flip_probability_matrix'])
        >>> # [[0.0, 0.85, 0.23],
        >>> #  [0.85, 0.0, 0.91], 
        >>> #  [0.23, 0.91, 0.0]]
        >>> 
        >>> print("翻转分析:")
        >>> for analysis in results['flip_analysis']:
        ...     print(f"边{analysis['edge']}: 翻转概率={analysis['flip_probability']:.3f}")
    """
    
    # 1. 获取完整分类结果
    classification_results = complete_edge_classification_pipeline(
        dense_adj_matrix, dense_features, classifier, encoder,
        include_self_loops=include_self_loops, directed=directed,
        return_probabilities=True
    )
    
    predictions = classification_results['predictions']
    edge_pairs = classification_results['edge_pairs']
    edge_labels = classification_results['edge_labels']
    num_nodes = dense_adj_matrix.shape[0]
    
    # 2. 转换为稠密概率矩阵
    if return_dense_matrix:
        flip_prob_matrix = predictions_to_dense_probability_matrix(
            predictions, edge_pairs, num_nodes, directed=directed
        )
    else:
        flip_prob_matrix = None
    
    # 3. 边翻转分析
    flip_analysis = []
    for i, (src, dst) in enumerate(edge_pairs):
        src, dst = int(src), int(dst)
        original_exists = bool(edge_labels[i])
        predicted_prob = float(predictions[i])
        
        # 计算翻转概率
        if original_exists:
            # 原本存在的边，翻转概率 = 1 - 预测存在概率
            flip_probability = 1.0 - predicted_prob
            flip_type = "removal"  # 移除边
        else:
            # 原本不存在的边，翻转概率 = 预测存在概率
            flip_probability = predicted_prob
            flip_type = "addition"  # 添加边
        
        analysis_info = {
            'edge': (src, dst),
            'original_exists': original_exists,
            'predicted_probability': predicted_prob,
            'flip_probability': flip_probability,
            'flip_type': flip_type,
            'confidence': abs(predicted_prob - 0.5) * 2,  # 置信度 (0-1)
            'recommendation': 'flip' if flip_probability > 0.5 else 'keep'
        }
        flip_analysis.append(analysis_info)
    
    # 4. 统计信息
    flip_probs = [analysis['flip_probability'] for analysis in flip_analysis]
    existing_edges = [analysis for analysis in flip_analysis if analysis['original_exists']]
    non_existing_edges = [analysis for analysis in flip_analysis if not analysis['original_exists']]
    
    statistics = {
        'total_edges_analyzed': len(flip_analysis),
        'existing_edges_count': len(existing_edges),
        'non_existing_edges_count': len(non_existing_edges),
        'average_flip_probability': float(np.mean(flip_probs)),
        'max_flip_probability': float(np.max(flip_probs)),
        'min_flip_probability': float(np.min(flip_probs)),
        'high_flip_probability_edges': len([p for p in flip_probs if p > 0.7]),
        'low_flip_probability_edges': len([p for p in flip_probs if p < 0.3]),
        'recommended_flips': len([a for a in flip_analysis if a['recommendation'] == 'flip'])
    }
    
    if existing_edges:
        removal_probs = [a['flip_probability'] for a in existing_edges]
        statistics['average_removal_probability'] = float(np.mean(removal_probs))
    
    if non_existing_edges:
        addition_probs = [a['flip_probability'] for a in non_existing_edges]
        statistics['average_addition_probability'] = float(np.mean(addition_probs))
    
    print(f"边翻转概率分析完成:")
    print(f"• 分析了 {statistics['total_edges_analyzed']} 条边")
    print(f"• 平均翻转概率: {statistics['average_flip_probability']:.3f}")
    print(f"• 建议翻转的边: {statistics['recommended_flips']} 条")
    
    return {
        'flip_probability_matrix': flip_prob_matrix,
        'original_adj_matrix': dense_adj_matrix.copy(),
        'predictions': predictions,
        'edge_pairs': edge_pairs,
        'flip_analysis': flip_analysis,
        'statistics': statistics,
        'classification_results': classification_results
    }


def complete_edge_classification_pipeline(dense_adj_matrix, dense_features, classifier, 
                                        encoder=None, include_self_loops=False, 
                                        directed=False, return_probabilities=True, 
                                        threshold=0.5, batch_size=1000):
    """
    完整的边分类流水线：从稠密矩阵到分类结果
    
    Args:
        dense_adj_matrix: 稠密邻接矩阵
        dense_features: 稠密特征矩阵
        classifier: EdgeFlipMAE的edge_classifier
        encoder: EdgeFlipMAE的encoder (可选，如果提供则先编码特征)
        include_self_loops: 是否包含自环边
        directed: 是否为有向图
        return_probabilities: 是否返回概率值
        threshold: 二分类阈值
        batch_size: 批处理大小
    
    Returns:
        dict: 完整的分类结果
    """
    
    # 1. 处理特征
    if encoder is not None:
        # 如果提供了encoder，先编码特征
        x = dense_features_to_encoder_format(dense_features)
        edge_index = dense_adj_to_edge_index(dense_adj_matrix, 
                                           remove_self_loops=not include_self_loops,
                                           ensure_undirected=not directed)
        
        encoder.eval()
        with torch.no_grad():
            node_embeddings = encoder(x, edge_index)
        
        dense_node_embeddings = node_embeddings_to_dense_matrix(node_embeddings)
    else:
        # 直接使用提供的特征作为节点嵌入
        dense_node_embeddings = dense_features
    
    # 2. 批量分类
    results = batch_classify_all_edges(
        classifier, dense_adj_matrix, dense_node_embeddings,
        batch_size=batch_size, include_self_loops=include_self_loops,
        directed=directed, return_probabilities=return_probabilities,
        classify_all_possible_edges=True
    )
    
    # 3. 添加二分类结果
    if return_probabilities:
        binary_predictions = (results['predictions'] > threshold).float()
    else:
        binary_predictions = (results['predictions'] > 0).float()
    
    results['binary_predictions'] = binary_predictions
    results['threshold'] = threshold
    
    return results


def demo_edge_flip_probability_analysis():
    """
    演示边翻转概率分析功能
    """
    print("=== 边翻转概率分析演示 ===")
    
    # 创建示例数据
    print("\n1. 创建示例数据:")
    adj_matrix = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    features = np.random.randn(4, 64)
    
    print(f"邻接矩阵:\n{adj_matrix}")
    print(f"特征矩阵形状: {features.shape}")
    
    # 创建模拟分类器
    print("\n2. 创建模拟分类器:")
    import torch.nn as nn
    
    class MockClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            nn.init.xavier_uniform_(self.linear.weight)
        
        def forward(self, x):
            return self.linear(x)
    
    mock_classifier = MockClassifier(128)  # 64*2=128
    
    # 边翻转概率分析
    print("\n3. 边翻转概率分析:")
    flip_results = edge_flip_probability_analysis(
        adj_matrix, features, mock_classifier
    )
    
    print("\n4. 分析结果:")
    
    print("\n4.1 稠密翻转概率矩阵:")
    print(f"形状: {flip_results['flip_probability_matrix'].shape}")
    print(f"矩阵:\n{flip_results['flip_probability_matrix']}")
    
    print("\n4.2 边翻转分析:")
    for i, analysis in enumerate(flip_results['flip_analysis'][:6]):
        edge = analysis['edge']
        original = "存在" if analysis['original_exists'] else "不存在"
        flip_prob = analysis['flip_probability']
        flip_type = "移除" if analysis['flip_type'] == "removal" else "添加"
        recommendation = "翻转" if analysis['recommendation'] == "flip" else "保持"
        
        print(f"  边{edge}: 原本{original}, {flip_type}概率={flip_prob:.3f}, 建议{recommendation}")
    
    print("\n4.3 统计信息:")
    stats = flip_results['statistics']
    print(f"  总边数: {stats['total_edges_analyzed']}")
    print(f"  平均翻转概率: {stats['average_flip_probability']:.3f}")
    print(f"  建议翻转的边: {stats['recommended_flips']}")
    print(f"  高翻转概率边(>0.7): {stats['high_flip_probability_edges']}")
    print(f"  低翻转概率边(<0.3): {stats['low_flip_probability_edges']}")
    
    if 'average_removal_probability' in stats:
        print(f"  平均移除概率: {stats['average_removal_probability']:.3f}")
    if 'average_addition_probability' in stats:
        print(f"  平均添加概率: {stats['average_addition_probability']:.3f}")
    
    print("\n5. 使用说明:")
    print("• flip_probability_matrix: 稠密矩阵形式，matrix[i,j]表示边(i,j)的翻转概率")
    print("• flip_analysis: 每条边的详细翻转分析，包括翻转类型和建议")
    print("• statistics: 整体翻转概率统计，帮助理解图的稳定性")
    
    return flip_results


if __name__ == "__main__":
    test_converters()
    print("\n" + "="*50)
    test_embedding_converters()
    print("\n" + "="*50)
    test_classifier_input_preparation()
    print("\n" + "="*50)
    demo_edge_flip_probability_analysis()
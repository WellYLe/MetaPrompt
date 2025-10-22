# Mettack数据集与EdgeFlipMAE模型集成使用指南

## 概述

本指南详细说明如何使用`test_mettack.py`生成的边翻转数据集来训练`EdgeFlipMAE`模型。

## 数据流程图

```
Mettack攻击 → 边翻转数据集 → EdgeFlipMAE训练 → 边翻转检测模型
```

## 1. 数据集格式分析

### Mettack数据集输出格式

`test_mettack.py`中的`build_edge_change_dataset`函数生成的数据集包含：

- **X_pairs**: `(M, 2*F + k)` - 节点对特征矩阵
  - 前`2*F`维：两个节点特征的拼接 `[feat_u, feat_v]`
  - 后`k`维：结构特征 `[edge_before, deg_u, deg_v, common_neighbors]`
- **pairs**: `(M, 2)` - 边的节点对索引 `[(u1,v1), (u2,v2), ...]`
- **y**: `(M,)` - 边翻转标签 (1=翻转, 0=未翻转)

### EdgeFlipMAE模型输入要求

EdgeFlipMAE模型需要：
- **graph_data**: PyTorch Geometric `Data`对象，包含节点特征和边索引
- **edge_pairs**: 待分类的边对列表
- **labels**: 边翻转标签

## 2. 关键适配点

### 2.1 特征维度提取

从`X_pairs`中分离出单个节点的特征：

```python
# 假设结构特征维度为4
struct_feat_dim = 4
node_feat_dim = (X_pairs.shape[1] - struct_feat_dim) // 2

# 提取节点u和v的特征
feat_u = X_pairs[i, :node_feat_dim]
feat_v = X_pairs[i, node_feat_dim:2*node_feat_dim]
```

### 2.2 图结构重建

从边对重建完整的图结构：

```python
# 创建无向图的双向边
edges = []
for u, v in pairs:
    edges.append([u, v])
    edges.append([v, u])  # 添加反向边

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_index = torch.unique(edge_index, dim=1)  # 去重
```

### 2.3 节点特征聚合

由于同一节点可能出现在多个边对中，需要聚合特征：

```python
# 对每个节点的特征求平均
for i in range(n_nodes):
    if node_count[i] > 0:
        node_features[i] /= node_count[i]
    else:
        # 未出现的节点使用随机特征
        node_features[i] = np.random.randn(node_feat_dim) * 0.1
```

## 3. 使用步骤

### 步骤1：生成Mettack数据集

```bash
cd DeepRobust/examples/graph
python test_mettack.py --dataset cora --ptb_rate 0.05
```

这将生成：
- `cora_edgeflip_dataset_ptbrate005.npz` - 数据文件
- `cora_edgeflip_dataset_ptbrate005_meta.json` - 元数据文件

### 步骤2：训练EdgeFlipMAE模型

```python
from mettack_to_edgeflip_integration import main_integration_example

# 运行完整的集成训练
main_integration_example()
```

### 步骤3：使用训练好的模型

```python
from mettack_to_edgeflip_integration import load_and_predict_with_mettack_model

# 加载模型并进行预测
load_and_predict_with_mettack_model()
```

## 4. 核心函数说明

### 4.1 `load_mettack_dataset(npz_path, meta_path=None)`

加载mettack生成的数据集文件。

**参数：**
- `npz_path`: .npz数据文件路径
- `meta_path`: 可选的元数据JSON文件路径

**返回：**
- 包含X_pairs, pairs, y和stats的字典

### 4.2 `extract_node_features_from_pairs(X_pairs, pairs, n_nodes)`

从节点对特征中提取单个节点的特征。

**参数：**
- `X_pairs`: 节点对特征矩阵
- `pairs`: 边的节点对索引
- `n_nodes`: 图中节点总数

**返回：**
- `(N, F)` 节点特征矩阵

### 4.3 `create_graph_from_pairs(pairs, node_features)`

从边对创建PyTorch Geometric图数据。

**参数：**
- `pairs`: 边的节点对索引
- `node_features`: 节点特征矩阵

**返回：**
- PyTorch Geometric `Data`对象

### 4.4 `train_edgeflip_mae_with_mettack_data(dataset_dict, **kwargs)`

使用mettack数据集训练EdgeFlipMAE模型。

**主要参数：**
- `dataset_dict`: 数据集字典
- `gnn_type`: GNN类型 ('GCN', 'GAT', 'GraphSAGE'等)
- `hid_dim`: 隐藏层维度
- `epochs`: 训练轮数
- `batch_size`: 批次大小

## 5. 配置建议

### 5.1 模型超参数

```python
# 推荐配置
config = {
    'gnn_type': 'GCN',           # 或 'GAT', 'GraphSAGE'
    'hid_dim': 64,               # 隐藏层维度
    'num_layer': 2,              # GNN层数
    'epochs': 100,               # 训练轮数
    'batch_size': 64,            # 批次大小
    'learning_rate': 0.001,      # 学习率
    'mask_rate': 0.15,           # 节点掩码率
    'noise_rate': 0.1            # 噪声率
}
```

### 5.2 数据集参数

```python
# build_edge_change_dataset参数
dataset_config = {
    'max_neg_ratio': 3,          # 负样本比例
    'keep_all': False,           # 是否保留所有负样本
    'add_structural_features': True  # 是否添加结构特征
}
```

## 6. 性能评估

模型训练完成后，会输出以下指标：

- **Accuracy**: 分类准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1分数
- **AUC**: ROC曲线下面积

## 7. 故障排除

### 7.1 常见错误

1. **文件不存在错误**
   ```
   错误: 数据文件不存在 xxx.npz
   解决: 先运行test_mettack.py生成数据集
   ```

2. **维度不匹配错误**
   ```
   错误: 特征维度推断错误
   解决: 检查结构特征维度设置
   ```

3. **内存不足错误**
   ```
   错误: CUDA out of memory
   解决: 减小batch_size或使用CPU训练
   ```

### 7.2 调试技巧

1. **检查数据集统计信息**
   ```python
   dataset_dict = load_mettack_dataset(npz_path, meta_path)
   print(f"正负样本比例: {np.mean(dataset_dict['y']):.4f}")
   ```

2. **验证图结构**
   ```python
   graph_data = create_graph_from_pairs(pairs, node_features)
   print(f"图连通性: {graph_data.num_edges / (2 * graph_data.num_nodes):.2f}")
   ```

3. **监控训练过程**
   ```python
   # EdgeFlipMAE会自动打印训练进度
   # 注意观察验证集F1分数的变化
   ```

## 8. 扩展应用

### 8.1 多数据集训练

可以合并多个不同扰动率的数据集进行训练：

```python
# 加载多个数据集
datasets = []
for ptb_rate in [0.01, 0.05, 0.1]:
    npz_path = f"cora_edgeflip_dataset_ptbrate{int(ptb_rate*100):03d}.npz"
    datasets.append(load_mettack_dataset(npz_path))

# 合并数据集
combined_dataset = merge_datasets(datasets)
```

### 8.2 跨数据集评估

在一个数据集上训练，在另一个数据集上测试：

```python
# 在Cora上训练
model = train_edgeflip_mae_with_mettack_data(cora_dataset)

# 在CiteSeer上测试
citeseer_dataset = load_mettack_dataset("citeseer_edgeflip_dataset.npz")
evaluate_cross_dataset(model, citeseer_dataset)
```

## 9. 总结

通过本集成方案，您可以：

1. ✅ 无缝连接Mettack攻击数据与EdgeFlipMAE模型
2. ✅ 自动处理数据格式转换和适配
3. ✅ 获得高质量的边翻转检测模型
4. ✅ 支持多种GNN架构和超参数配置
5. ✅ 提供完整的训练、评估和预测流程

这个集成方案为图对抗攻击检测研究提供了一个强大而灵活的工具链。
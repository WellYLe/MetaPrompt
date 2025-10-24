# NPZ数据加载器使用说明

## 概述

`NPZDataLoader` 是一个专门设计的数据加载器，用于从NPZ文件加载图数据并生成与现有 `node_task.py` 兼容的数据加载器。该加载器支持从修改后的图数据（如通过DeepRobust生成的对抗样本）创建induced graphs，并提供与原有代码完全兼容的接口。

## 主要功能

1. **NPZ文件加载**: 支持从NPZ格式文件加载图数据
2. **子图生成**: 使用与 `induced_graph.py` 相同的算法生成induced graphs
3. **数据分割**: 自动根据训练/测试索引分割数据
4. **兼容接口**: 提供与现有 `GraphDataset` 和 `DataLoader` 完全兼容的接口
5. **设备支持**: 支持CPU和GPU设备

## 安装要求

```python
torch
torch_geometric
numpy
scipy
pickle
```

## 使用方法

### 基本使用

```python
from npz_dataloader import NPZDataLoader

# 创建数据加载器
loader = NPZDataLoader(
    dataset_name="cora",
    npz_file_path="path/to/cora_modified_001.npz",
    device='cpu',  # 或 'cuda'
    batch_size=32,
    smallest_size=10,  # 子图最小节点数
    largest_size=300   # 子图最大节点数
)

# 获取训练和测试数据加载器
train_loader, test_loader = loader.get_data_loaders()

# 使用数据加载器（与现有代码完全兼容）
for batch in train_loader:
    # batch.x: 节点特征
    # batch.edge_index: 边索引
    # batch.y: 标签
    # batch.index: 原始节点索引
    pass
```

### 便捷函数

```python
from npz_dataloader import create_npz_dataloader

# 使用便捷函数创建
loader = create_npz_dataloader(
    dataset_name="cora",
    npz_file_path="path/to/cora_modified_001.npz",
    device='cpu',
    batch_size=32
)

train_loader, test_loader = loader.get_data_loaders()
```

### 获取原始数据

```python
# 获取原始图数据（PyTorch Geometric格式）
original_data = loader.get_original_data()
print(f"节点数: {original_data.x.shape[0]}")
print(f"特征维度: {original_data.x.shape[1]}")
print(f"边数: {original_data.edge_index.shape[1]}")
```

### 保存Induced Graphs

```python
# 保存生成的induced graphs到文件
loader.save_induced_graphs("./induced_graphs_cora.pkl")
```

## 参数说明

### NPZDataLoader参数

- `dataset_name`: 数据集名称（字符串）
- `npz_file_path`: NPZ文件的完整路径
- `device`: 计算设备，'cpu' 或 'cuda'
- `batch_size`: 批处理大小，默认32
- `smallest_size`: 子图最小节点数，默认10
- `largest_size`: 子图最大节点数，默认300

## 与现有代码的兼容性

### 与node_task.py的兼容性

该数据加载器生成的 `train_loader` 和 `test_loader` 与 `node_task.py` 中第249-267行的数据加载器完全兼容：

```python
# 现有代码中的使用方式
if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
    # 使用NPZDataLoader替代原有逻辑
    npz_loader = NPZDataLoader(dataset_name, npz_path, device, batch_size)
    train_loader, test_loader = npz_loader.get_data_loaders()
    
    # 后续代码无需修改
    # 训练和测试逻辑保持不变
```

### 与GraphDataset的兼容性

生成的数据加载器使用相同的 `GraphDataset` 类，确保完全兼容：

```python
# 数据结构完全一致
for batch in train_loader:
    assert hasattr(batch, 'x')          # 节点特征
    assert hasattr(batch, 'edge_index') # 边索引
    assert hasattr(batch, 'y')          # 标签
    assert hasattr(batch, 'index')      # 原始节点索引
```

## NPZ文件格式要求

NPZ文件应包含以下键值：

- `adj_data`: 邻接矩阵数据（CSR格式）
- `adj_indices`: 邻接矩阵索引
- `adj_indptr`: 邻接矩阵指针
- `adj_shape`: 邻接矩阵形状
- `attr_data`: 节点特征数据（CSR格式）
- `attr_indices`: 特征矩阵索引
- `attr_indptr`: 特征矩阵指针
- `attr_shape`: 特征矩阵形状
- `labels`: 节点标签
- `idx_train`: 训练节点索引
- `idx_val`: 验证节点索引
- `idx_test`: 测试节点索引

## 性能优化

1. **设备选择**: 如果有GPU，建议使用 `device='cuda'` 以加速计算
2. **批处理大小**: 根据内存大小调整 `batch_size`
3. **子图大小**: 根据数据集特点调整 `smallest_size` 和 `largest_size`

## 示例代码

完整的使用示例请参考 `usage_example.py` 文件。

## 注意事项

1. 确保NPZ文件路径正确且文件存在
2. 如果遇到OpenMP冲突，设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE`
3. 大型数据集可能需要较长时间生成induced graphs
4. 生成的induced graphs会自动缓存，避免重复计算

## 错误处理

常见错误及解决方案：

1. **ImportError**: 检查依赖包是否正确安装
2. **FileNotFoundError**: 检查NPZ文件路径是否正确
3. **CUDA错误**: 检查GPU内存是否足够，或改用CPU
4. **内存错误**: 减小批处理大小或子图大小

## 更新日志

- v1.0: 初始版本，支持基本的NPZ文件加载和induced graph生成
- 完全兼容现有的node_task.py和GraphDataset接口
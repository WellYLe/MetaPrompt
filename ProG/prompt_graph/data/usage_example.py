"""
NPZ数据加载器使用示例
演示如何使用NPZDataLoader来加载和处理图数据
"""

import torch
from npz_dataloader import NPZDataLoader, create_npz_dataloader

def example_usage():
    """
    使用示例：展示如何使用NPZDataLoader
    """
    # 设置参数
    dataset_name = "cora"
    npz_file_path = r"c:\Users\11326\Desktop\MetaPrompt\DeepRobust\examples\graph\tmp\cora_modified_001.npz"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    
    print(f"Using device: {device}")
    
    # 方法1: 直接创建NPZDataLoader实例
    print("=== 方法1: 直接创建NPZDataLoader ===")
    dataloader = NPZDataLoader(
        dataset_name=dataset_name,
        npz_file_path=npz_file_path,
        device=device,
        batch_size=batch_size,
        smallest_size=10,
        largest_size=300
    )
    
    # 获取数据加载器
    train_loader, test_loader = dataloader.get_data_loaders()
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 查看第一个批次的数据
    for batch in train_loader:
        print(f"训练批次形状: {batch}")
        print(f"节点特征形状: {batch.x.shape}")
        print(f"边索引形状: {batch.edge_index.shape}")
        print(f"标签形状: {batch.y.shape}")
        break
    
    # 方法2: 使用便捷函数
    print("\n=== 方法2: 使用便捷函数 ===")
    dataloader2 = create_npz_dataloader(
        dataset_name=dataset_name,
        npz_file_path=npz_file_path,
        device=device,
        batch_size=batch_size
    )
    
    train_loader2, test_loader2 = dataloader2.get_data_loaders()
    print(f"训练集批次数: {len(train_loader2)}")
    print(f"测试集批次数: {len(test_loader2)}")
    
    # 获取原始数据
    original_data = dataloader.get_original_data()
    print(f"\n原始数据信息:")
    print(f"节点数: {original_data.x.shape[0]}")
    print(f"特征维度: {original_data.x.shape[1]}")
    print(f"边数: {original_data.edge_index.shape[1]}")
    print(f"类别数: {len(torch.unique(original_data.y))}")
    print(f"训练节点数: {len(original_data.idx_train)}")
    print(f"测试节点数: {len(original_data.idx_test)}")
    
    # 保存induced graphs（可选）
    save_path = "./induced_graphs_cora.pkl"
    dataloader.save_induced_graphs(save_path)
    print(f"\nInduced graphs已保存到: {save_path}")


def compatibility_test():
    """
    兼容性测试：验证与现有代码的兼容性
    """
    print("=== 兼容性测试 ===")
    
    # 模拟node_task.py中的使用方式
    dataset_name = "cora"
    npz_file_path = r"c:\Users\11326\Desktop\MetaPrompt\DeepRobust\examples\graph\tmp\cora_modified_001.npz"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建数据加载器
    dataloader = NPZDataLoader(
        dataset_name=dataset_name,
        npz_file_path=npz_file_path,
        device=device,
        batch_size=32
    )
    
    # 获取数据加载器（类似node_task.py中的方式）
    train_loader, test_loader = dataloader.get_data_loaders()
    
    # 验证数据结构
    print("验证训练数据加载器:")
    for i, batch in enumerate(train_loader):
        print(f"批次 {i+1}:")
        print(f"  - 图数量: {batch.num_graphs}")
        print(f"  - 节点特征: {batch.x.shape}")
        print(f"  - 边索引: {batch.edge_index.shape}")
        print(f"  - 标签: {batch.y.shape}")
        print(f"  - 索引: {batch.index.shape if hasattr(batch, 'index') else 'N/A'}")
        
        if i >= 2:  # 只显示前3个批次
            break
    
    print("\n验证测试数据加载器:")
    for i, batch in enumerate(test_loader):
        print(f"批次 {i+1}:")
        print(f"  - 图数量: {batch.num_graphs}")
        print(f"  - 节点特征: {batch.x.shape}")
        print(f"  - 边索引: {batch.edge_index.shape}")
        print(f"  - 标签: {batch.y.shape}")
        
        if i >= 2:  # 只显示前3个批次
            break
    
    print("\n兼容性测试完成！")


if __name__ == "__main__":
    try:
        example_usage()
        print("\n" + "="*50)
        compatibility_test()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
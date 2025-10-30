import numpy as np

# 加载 npz 文件，允许加载 pickle 对象
npz_file = np.load(r'c:/Users/11326/Desktop/MetaPrompt/DeepRobust/examples/graph/tmp/pubmed.npz', allow_pickle=True)

print("NPZ 文件中的键:")
print("=" * 30)

# 显示所有键
for key in npz_file.files:
    print(f"键名: {key}")
    
    # 获取对应数组的信息
    array = npz_file[key]
    print(f"  - 形状: {array.shape}")
    print(f"  - 数据类型: {array.dtype}")
    print(f"  - 维度: {array.ndim}")
    
    # 如果数组不太大，显示前几个元素
    if array.size <= 20:
        print(f"  - 数据: {array}")
    else:
        print(f"  - 数据预览: {array.flat[:5]}... (共 {array.size} 个元素)")
    print()

# 关闭文件
npz_file.close()

print(f"总共有 {len(npz_file.files)} 个键")
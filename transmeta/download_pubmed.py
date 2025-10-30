"""
下载和准备PubMed数据集的脚本
类似于cora.npz的格式，生成适合EdgeFlipMAE使用的pubmed.npz文件
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import networkx as nx
import urllib.request
from sklearn.preprocessing import StandardScaler

# 添加DeepRobust路径
sys.path.append('../DeepRobust')
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def download_pubmed_raw_files(root_dir):
    """
    下载PubMed原始数据文件
    """
    os.makedirs(root_dir, exist_ok=True)
    
    base_url = 'https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/'
    files = [
        'ind.pubmed.x', 'ind.pubmed.y', 'ind.pubmed.tx', 'ind.pubmed.ty',
        'ind.pubmed.allx', 'ind.pubmed.ally', 'ind.pubmed.graph', 'ind.pubmed.test.index'
    ]
    
    print("下载PubMed原始数据文件...")
    for filename in files:
        file_path = os.path.join(root_dir, filename)
        if not os.path.exists(file_path):
            url = base_url + filename
            print(f"下载: {url}")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"完成: {filename}")
            except Exception as e:
                print(f"下载失败 {filename}: {e}")
                return False
    
    print("所有文件下载完成!")
    return True

def load_pubmed_raw_data(root_dir):
    """
    加载PubMed原始数据并转换为标准格式
    """
    print("加载PubMed原始数据...")
    
    dataset = 'pubmed'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    
    for name in names:
        filename = f"ind.{dataset}.{name}"
        file_path = os.path.join(root_dir, filename)
        
        with open(file_path, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    
    # 加载测试索引
    test_idx_file = os.path.join(root_dir, "ind.pubmed.test.index")
    test_idx_reorder = parse_index_file(test_idx_file)
    test_idx_range = np.sort(test_idx_reorder)
    
    # 构建特征矩阵
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    # 构建邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    # 构建标签
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.where(labels)[1]
    
    print(f"数据加载完成:")
    print(f"  节点数: {adj.shape[0]}")
    print(f"  边数: {adj.nnz // 2}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  类别数: {len(np.unique(labels))}")
    
    return adj, features, labels

def save_pubmed_npz(adj, features, labels, save_path):
    """
    保存PubMed数据为npz格式，兼容EdgeFlipMAE的加载函数
    """
    print(f"保存数据到: {save_path}")
    
    # 确保邻接矩阵是CSR格式
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()
    
    # 确保特征矩阵是CSR格式
    if not sp.isspmatrix_csr(features):
        features = features.tocsr()
    
    # 生成训练/验证/测试划分
    idx_train, idx_val, idx_test = get_train_val_test(
        nnodes=adj.shape[0], 
        val_size=0.1, 
        test_size=0.8, 
        stratify=labels,
        seed=42
    )
    
    print(f"数据划分:")
    print(f"  训练集: {len(idx_train)} 节点")
    print(f"  验证集: {len(idx_val)} 节点") 
    print(f"  测试集: {len(idx_test)} 节点")
    
    # 保存为npz格式，使用与cora.npz相同的结构
    np.savez(
        save_path,
        # 邻接矩阵
        adj_data=adj.data,
        adj_indices=adj.indices,
        adj_indptr=adj.indptr,
        adj_shape=adj.shape,
        # 特征矩阵
        attr_data=features.data,
        attr_indices=features.indices,
        attr_indptr=features.indptr,
        attr_shape=features.shape,
        # 标签和索引
        labels=labels,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test
    )
    
    print(f"数据保存完成: {save_path}")
    return True

def verify_pubmed_npz(npz_path):
    """
    验证生成的npz文件是否正确
    """
    print(f"验证文件: {npz_path}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        print("文件包含的键:")
        for key in data.files:
            print(f"  {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")
        
        # 重构邻接矩阵和特征矩阵
        adj = sp.csr_matrix(
            (data['adj_data'], data['adj_indices'], data['adj_indptr']),
            shape=tuple(data['adj_shape'])
        )
        
        features = sp.csr_matrix(
            (data['attr_data'], data['attr_indices'], data['attr_indptr']),
            shape=tuple(data['attr_shape'])
        )
        
        labels = data['labels']
        idx_train = data['idx_train']
        idx_val = data['idx_val']
        idx_test = data['idx_test']
        
        print(f"验证结果:")
        print(f"  邻接矩阵: {adj.shape}, 边数: {adj.nnz // 2}")
        print(f"  特征矩阵: {features.shape}")
        print(f"  标签: {labels.shape}, 类别数: {len(np.unique(labels))}")
        print(f"  训练集: {len(idx_train)}")
        print(f"  验证集: {len(idx_val)}")
        print(f"  测试集: {len(idx_test)}")
        
        data.close()
        print("验证通过!")
        return True
        
    except Exception as e:
        print(f"验证失败: {e}")
        return False

def main():
    """
    主函数：下载和准备PubMed数据集
    """
    print("=== PubMed数据集下载和准备工具 ===")
    
    # 设置路径
    raw_data_dir = "../PubMed/raw"
    output_dir = "../DeepRobust/examples/graph/tmp"
    output_file = os.path.join(output_dir, "pubmed.npz")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 下载原始数据文件
    if not download_pubmed_raw_files(raw_data_dir):
        print("下载失败，退出程序")
        return False
    
    # 步骤2: 加载和处理数据
    try:
        adj, features, labels = load_pubmed_raw_data(raw_data_dir)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False
    
    # 步骤3: 保存为npz格式
    try:
        save_pubmed_npz(adj, features, labels, output_file)
    except Exception as e:
        print(f"数据保存失败: {e}")
        return False
    
    # 步骤4: 验证生成的文件
    if verify_pubmed_npz(output_file):
        print(f"\n✅ PubMed数据集准备完成!")
        print(f"文件位置: {output_file}")
        print(f"现在可以在EdgeFlipMAE中使用此文件了。")
        return True
    else:
        print("❌ 文件验证失败")
        return False

if __name__ == "__main__":
    main()
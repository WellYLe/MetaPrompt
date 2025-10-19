
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append('c:/Users/11326/Desktop/MetaPrompt/transmeta')
import torch
from ..ProG.prompt_graph.tasker.LinkTask import LinkTask
from torch_geometric.data import Data

def load_edge_samples(data_path):
    """加载边样本数据"""
    edge_samples_path = os.path.join(data_path, 'edge_samples.pt')
    edge_labels_path = os.path.join(data_path, 'edge_labels.pt')
    
    if os.path.exists(edge_samples_path) and os.path.exists(edge_labels_path):
        edge_samples = torch.load(edge_samples_path)
        edge_labels = torch.load(edge_labels_path)
        print(f"Loaded edge samples: {edge_samples.shape}")
        print(f"Loaded edge labels: {edge_labels.shape}")
        return edge_samples, edge_labels
    else:
        raise FileNotFoundError(f"Edge data files not found in {data_path}")

def generate_edge_induced_graphs_for_cora():
    """为Cora数据集生成边诱导子图"""
    
    # 数据路径
    data_path = 'c:/Users/11326/Desktop/MetaPrompt/transmeta/Experiment/sample_data/Node/Cora/1_shot'
    
    # 加载边样本
    try:
        edge_samples, edge_labels = load_edge_samples(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 创建LinkTask实例（需要先加载完整的Cora图数据）
    # 这里假设你有完整的Cora数据集
    try:
        link_task = LinkTask(dataset_name='Cora', num_shot=5)
        
        # 使用新方法生成边诱导子图
        edge_graphs = link_task.load_edge_induced_graph(
            edge_samples=edge_samples,
            num_hops=5,           # 5跳邻域
            smallest_size=10,     # 最小子图大小
            largest_size=300      # 最大子图大小
        )
        
        print(f"Generated {len(edge_graphs)} edge-induced subgraphs")
        
        # 检查生成的子图
        for i, graph in enumerate(edge_graphs[:3]):  # 查看前3个子图
            print(f"\nEdge {i}:")
            print(f"  Nodes: {graph.x.shape[0]}")
            print(f"  Edges: {graph.edge_index.shape[1]}")
            print(f"  Original edge: {graph.src_node} -> {graph.dst_node}")
            print(f"  Edge index in dataset: {graph.edge_idx}")
        
        # 保存生成的边诱导子图
        output_path = os.path.join(data_path, 'edge_induced_graphs.pt')
        torch.save(edge_graphs, output_path)
        print(f"\nSaved edge-induced graphs to: {output_path}")
        
        return edge_graphs
        
    except Exception as e:
        print(f"Error creating LinkTask or generating graphs: {e}")
        return None

def analyze_edge_data_structure():
    """分析边数据的结构"""
    data_path = 'c:/Users/11326/Desktop/MetaPrompt/transmeta/Experiment/sample_data/Link/Cora/5_shot'
    
    try:
        edge_samples, edge_labels = load_edge_samples(data_path)
        
        print("=== Edge Data Analysis ===")
        print(f"Edge samples shape: {edge_samples.shape}")
        print(f"Edge labels shape: {edge_labels.shape}")
        print(f"Edge samples dtype: {edge_samples.dtype}")
        print(f"Edge labels dtype: {edge_labels.dtype}")
        
        if len(edge_samples.shape) == 2 and edge_samples.shape[0] == 2:
            print(f"Number of edges: {edge_samples.shape[1]}")
            print(f"Sample edges (first 5):")
            for i in range(min(5, edge_samples.shape[1])):
                src, dst = edge_samples[0, i].item(), edge_samples[1, i].item()
                label = edge_labels[i].item() if len(edge_labels.shape) == 1 else edge_labels[i]
                print(f"  Edge {i}: {src} -> {dst}, Label: {label}")
        
        print(f"Unique labels: {torch.unique(edge_labels)}")
        print(f"Label distribution: {torch.bincount(edge_labels.long())}")
        
    except Exception as e:
        print(f"Error analyzing edge data: {e}")

if __name__ == "__main__":
    print("=== Analyzing Edge Data Structure ===")
    analyze_edge_data_structure()
    
    print("\n=== Generating Edge-Induced Graphs ===")
    edge_graphs = generate_edge_induced_graphs_for_cora()
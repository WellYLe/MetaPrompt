
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append('c:/Users/11326/Desktop/MetaPrompt/transmeta')
import torch
from LinkTask import LinkTask
from torch_geometric.data import Data
from prompt_graph.data import load4node
from torch_geometric.utils import k_hop_subgraph, subgraph

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

def build_edge_induced_graphs(data, edge_samples, edge_labels=None, num_hops=1, smallest_size=10, largest_size=300, device='cpu'):
    edge_graphs_list = []
    num_edges = edge_samples.shape[1]

    for edge_idx in range(num_edges):
        src_node = edge_samples[0, edge_idx].item()
        dst_node = edge_samples[1, edge_idx].item()

        # 各取 num_hops 跳邻域
        src_subset, _, _, _ = k_hop_subgraph(node_idx=src_node, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=False)
        dst_subset, _, _, _ = k_hop_subgraph(node_idx=dst_node, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=False)

        # 合并邻域并确保包含原始边端点
        combined_subset = torch.unique(torch.cat([src_subset, dst_subset, torch.tensor([src_node, dst_node], device=device)]))

        # 太小时，增加跳数到 7
        if len(combined_subset) < smallest_size:
            extended_hops = num_hops + 1
            while len(combined_subset) < smallest_size and extended_hops <= 2:
                src_subset_ext, _, _, _ = k_hop_subgraph(node_idx=src_node, num_hops=extended_hops, edge_index=data.edge_index, relabel_nodes=False)
                dst_subset_ext, _, _, _ = k_hop_subgraph(node_idx=dst_node, num_hops=extended_hops, edge_index=data.edge_index, relabel_nodes=False)
                combined_subset = torch.unique(torch.cat([src_subset_ext, dst_subset_ext, torch.tensor([src_node, dst_node], device=device)]))
                extended_hops += 1

        # 太大时，保留原始端点并随机采样其他节点
        if len(combined_subset) > largest_size:
            mask = ~torch.isin(combined_subset, torch.tensor([src_node, dst_node], device=device))
            other_nodes = combined_subset[mask]
            sampled_other = other_nodes[torch.randperm(len(other_nodes))][:largest_size - 2]
            combined_subset = torch.cat([torch.tensor([src_node, dst_node], device=device), sampled_other])

        # 构造诱导子图
        sub_edge_index, _ = subgraph(combined_subset, data.edge_index, relabel_nodes=True)
        x = data.x[combined_subset]

        graph_kwargs = dict(x=x, edge_index=sub_edge_index, edge_idx=edge_idx, src_node=src_node, dst_node=dst_node)
        if edge_labels is not None:
            graph_kwargs['y'] = edge_labels[edge_idx].item() if edge_labels.dim() == 1 else edge_labels[edge_idx]
        edge_graph = Data(**graph_kwargs)

        edge_graphs_list.append(edge_graph)

    return edge_graphs_list

def generate_edge_induced_graphs_for_cora():
    """为Cora数据集生成边诱导子图"""

    # 建议使用 Link 路径（边样本/边标签）
    data_path = 'c:/Users/11326/Desktop/MetaPrompt/transmeta/Experiment/sample_data/Link/Cora/5_shot/1'

    # 加载边样本
    try:
        edge_samples, edge_labels = load_edge_samples(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 加载完整图数据（用于取 k-hop 邻域）
    data, input_dim, output_dim = load4node('Cora')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)

    # 在本文件中生成边诱导子图
    edge_graphs = build_edge_induced_graphs(
        data=data,
        edge_samples=edge_samples.to(device),
        edge_labels=edge_labels.to(device),
        num_hops=1,           # 5跳邻域
        smallest_size=10,     # 最小子图大小
        largest_size=100,     # 最大子图大小
        device=device
    )

    print(f"Generated {len(edge_graphs)} edge-induced subgraphs")

    # 可选：将生成的子图交给 LinkTask 使用（作为 graphs_list）
    # link_task = LinkTask(
    #     data=data,
    #     input_dim=input_dim,
    #     output_dim=output_dim,
    #     graphs_list=edge_graphs,
    #     dataset_name='Cora',
    #     shot_num=5
    # )

    # 保存生成的边诱导子图
    output_path = os.path.join(data_path, 'edge_induced_graphs.pt')
    torch.save(edge_graphs, output_path)
    print(f"\nSaved edge-induced graphs to: {output_path}")

    return edge_graphs

def analyze_edge_data_structure():
    """分析边数据的结构"""
    data_path = 'c:/Users/11326/Desktop/MetaPrompt/transmeta/Experiment/sample_data/Link/Cora/5_shot/1'
    
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
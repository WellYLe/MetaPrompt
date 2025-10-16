import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F


class GPF(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(GPF, self).__init__()
        self.global_emb = torch.nn.Parameter(torch.Tensor(1,in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        return x + self.global_emb

class GPF_plus(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int, original_dim=None):
        super(GPF_plus, self).__init__()
        self.in_channels = in_channels
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        
        # 添加投影层，用于处理不同维度的输入
        self.projection = None
        if original_dim is not None and original_dim != in_channels:
            self.projection = torch.nn.Linear(original_dim, in_channels)
            print(f"Created projection layer: {original_dim} -> {in_channels}")
            
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()
        if self.projection is not None:
            self.projection.reset_parameters()

    def add(self, x: torch.Tensor):
        # 如果输入维度与预期不符，且有投影层，则进行投影
        if self.projection is not None and x.shape[1] != self.in_channels:
            x = self.projection(x)
            
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p


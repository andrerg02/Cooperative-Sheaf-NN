import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class SumGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') 
        self.W_s = nn.Linear(in_channels, out_channels, bias=False)
        self.W_n = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out = self.propagate(edge_index, x=x)

        return F.gelu(self.W_s(x) + self.W_n(out))

    def message(self, x_j):
        return x_j
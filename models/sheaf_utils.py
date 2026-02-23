from torch_geometric.nn import SGConv, SAGEConv, NNConv, GCNConv, GATConv, GPSConv, GraphSAGE
from models.sum_gnn import SumGNN
import inspect
import torch.nn.functional as F
from torch.nn import Linear, ReLU

import torch.nn as nn
import torch

class GPS(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias=False):
        super().__init__()

        self.conv = GPSConv(channels=in_channels, conv=GCNConv(in_channels, in_channels), heads=4, dropout=0.2, attn_dropout=0.2)

        self.mlp = nn.Sequential(
            Linear(in_channels, in_channels // 2),
            ReLU(),
            Linear(in_channels // 2, in_channels // 4),
            ReLU(),
            Linear(in_channels // 4, out_channels),
        )

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None):
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        
        return self.mlp(x)

def get_layer_type(layer_type):
    if layer_type == 'GCN':
        model_cls = GCNConv
    elif layer_type == 'GAT':
        model_cls = GATConv
    elif layer_type == 'SAGE':
        model_cls = SAGEConv
    elif layer_type == 'SGC':
        model_cls = SGConv
    elif layer_type == 'GPS':
        model_cls = GPSConv
    elif layer_type == 'NNConv':
        model_cls = NNConv
    elif layer_type == 'SumGNN':
        model_cls = SumGNN
    else:
        raise ValueError(f"Unsupported GNN layer type: {layer_type}")
    return model_cls
    
def gnn_builder(gnn_type, in_channels, out_channels, num_layers, hidden_channels=None, aggr='add'):
    gnn = get_layer_type(gnn_type)
    layers = nn.ModuleList()
    if hidden_channels is None or num_layers == 1:
        if gnn_type == 'GPS':
            for i in range(num_layers):
                layers.append(gnn(in_channels, conv=GCNConv(in_channels, in_channels), heads=2, dropout=0.2))
        elif gnn_type == 'NNConv':
            edge_net = nn.LazyLinear(in_channels*out_channels)
            for i in range(num_layers):
                layers.append(gnn(in_channels, out_channels, nn=edge_net, aggr=aggr))
        elif gnn_type == 'SGC':
            layers = gnn(in_channels, out_channels, K=num_layers)
        elif gnn_type in ['GCN', 'GAT', 'SAGE', 'SumGNN']:
            for i in range(num_layers):
                layers.append(gnn(in_channels, out_channels))
        else:
            raise ValueError(f"Unsupported GNN layer type: {gnn_type}")
    else:
        if gnn_type == 'GPS':
            layers.append(nn.Linear(in_channels, hidden_channels))
            for i in range(num_layers-2):
                layers.append(gnn(hidden_channels, hidden_channels, conv=GCNConv(hidden_channels, hidden_channels), heads=4, dropout=0.2))
            layers.append(nn.Linear(hidden_channels, out_channels))
        elif gnn_type == 'NNConv':
            edge_net = nn.LazyLinear(in_channels*hidden_channels)
            layers.append(gnn(in_channels, hidden_channels, nn=edge_net, aggr=aggr))
            edge_net = nn.LazyLinear(hidden_channels**2)
            for i in range(num_layers-2):
                layers.append(gnn(hidden_channels, hidden_channels, nn=edge_net, aggr=aggr))
            edge_net = nn.LazyLinear(hidden_channels*out_channels)
            layers.append(NNConv(hidden_channels, out_channels, nn=edge_net, aggr=aggr))
        elif gnn_type == 'SGC':
            layers.append(SGConv(in_channels, hidden_channels, K=1))
            layers.append(SGConv(hidden_channels, hidden_channels, K=num_layers-2))
            layers.append(SGConv(hidden_channels, out_channels, K=1))
        elif gnn_type in ['GCN', 'GAT', 'SAGE', 'SumGNN']:
            layers.append(gnn(in_channels, hidden_channels))
            for i in range(num_layers-2):
                layers.append(gnn(hidden_channels, hidden_channels))
            layers.append(gnn(hidden_channels, out_channels))
        else:
            raise ValueError(f"Unsupported GNN layer type: {gnn_type}")
    return layers

def custom_forward(self, maps, batch):
    if self.gnn_layers > 0:
            sig = inspect.signature(self.gnn.forward)
            if self.gnn_default:
                maps = self.phi(maps, batch.edge_index)
            elif self.linear_emb:
                maps = self.emb1(maps)
                maps = F.gelu(maps)
                if self.layer_type != 'SGC':
                    for layer in range(self.gnn_layers):
                        prev = maps
                        if batch.edge_attr is not None and 'edge_attr' in sig.parameters:
                            maps = self.phi[layer](maps, batch.edge_index, edge_attr=batch.edge_attr)
                        else:
                            maps = self.phi[layer](maps, batch.edge_index)
                        maps = F.gelu(maps)
                        if self.residual:
                            maps = maps + prev
                else:
                    maps = self.phi(maps, batch.edge_index)
                    maps = F.gelu(maps)
                maps = self.emb2(maps)
            else:
                if self.layer_type != 'SGC':
                    for layer in range(self.gnn_layers):
                        prev = maps
                        if batch.edge_attr is not None and 'edge_attr' in sig.parameters:
                            maps = self.phi[layer](maps, batch.edge_index, edge_attr=batch.edge_attr)
                        else:
                            maps = self.phi[layer](maps, batch.edge_index)
                        maps = F.gelu(maps) if (self.gnn_layers != 1 and layer != self.gnn_layers - 1) else maps
                        if self.residual and layer not in [self.gnn_layers - 1, 0]:
                            maps = maps + prev
                else:
                    if self.gnn_layers == 1:
                        maps = self.phi(maps, batch.edge_index)
                    else:
                        for layer in self.phi:
                            maps = layer(maps, batch.edge_index)
                            maps = F.gelu(maps) if layer != self.phi[-1] else maps
    else:
        maps = self.phi(maps)
    
    return maps

def batched_sym_matrix_pow(matrices: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrices: A batch of matrices.
        p: Power.
        positive_definite: If positive definite
    Returns:
        Power of each matrix in the batch.
    """
    # vals, vecs = torch.linalg.eigh(matrices)
    # SVD is much faster than  vals, vecs = torch.linalg.eigh(matrices) for large batches.
    vecs, vals, _ = torch.linalg.svd(matrices)
    good = vals > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
    vals = vals.pow(p).where(good, torch.zeros((), device=matrices.device, dtype=matrices.dtype))
    matrix_power = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
    return matrix_power
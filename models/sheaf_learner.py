import torch
import inspect

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn.pool import TopKPooling
from models.sheaf_utils import get_layer_type, gnn_builder, custom_forward


class ConformalSheafLearner(nn.Module):
    """Learns a conformal sheaf passing node features through a GNN or MLP + activation."""

    def __init__(self, d: int,  hidden_channels: int, out_shape: Tuple[int], linear_emb: bool,
                 gnn_type: str, gnn_layers: int, gnn_hidden: int, gnn_default: bool, gnn_residual: bool,
                 pe_size: int, conformal: bool, sheaf_act="tanh"):
        super(ConformalSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        assert (gnn_type, gnn_residual) != ('SGC', True), "SGC does not support residual connections."
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.gnn_layers = gnn_layers
        self.conformal = conformal
        self.residual = gnn_residual
        self.gnn_default = gnn_default
        self.linear_emb = linear_emb
        self.gnn_hidden = gnn_hidden
        self.layer_type = gnn_type
        out_channels = int(np.prod(out_shape) + 1) if conformal else int(np.prod(out_shape))

        if gnn_layers > 0:
            self.gnn = get_layer_type(gnn_type)
            if gnn_default == 1:
                self.phi = GraphSAGE(
                    hidden_channels * d + pe_size,
                    gnn_hidden,
                    num_layers=gnn_layers,
                    out_channels=out_channels,
                    norm='layer',
                    )
            elif gnn_default == 2:
                self.phi = GraphSAGE(
                    hidden_channels * d + pe_size,
                    gnn_hidden,
                    num_layers=gnn_layers,
                    out_channels=out_channels,
                    dropout=0.2,
                    act='gelu',
                    project=True,
                    bias=True,
                    )
            else:
                if linear_emb:
                    self.emb1 = nn.Linear((hidden_channels + pe_size) * d, gnn_hidden)
                    self.phi = gnn_builder(gnn_type, gnn_hidden, gnn_hidden, gnn_layers)
                    self.emb2 = nn.Linear(gnn_hidden, out_channels)
                else:
                    self.phi = gnn_builder(gnn_type, (hidden_channels + pe_size) * d, out_channels, gnn_layers, gnn_hidden)

        else:
            self.phi = nn.Sequential(
                nn.Linear((hidden_channels + pe_size) * d, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, out_channels)
            )

    def forward(self, x, batch):
        pe = batch.pe if hasattr(batch, 'pe') else torch.empty(x.size(0), 0, device=batch.x.device)
        maps = torch.cat([x, pe], -1)

        maps = custom_forward(self, maps, batch)

        if self.conformal:
            return torch.tanh(maps[:, :-1]), torch.exp(maps[:, -1].clamp_max(np.log(10)))
        else:
            return torch.tanh(maps), torch.ones(maps.size(0), device=x.device)
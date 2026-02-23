
from torch import nn
from torch_geometric.nn import global_mean_pool

from models.sheaf_base import SheafDiffusion
from models.CSNNconv import CSNNConv
from models.DenseCSNN import DenseCSNNConv

import torch.nn.functional as F

class CSNN(SheafDiffusion):
    def __init__(self, args):
        super(CSNN, self).__init__(args)
        assert args['d'] > 1
        assert args['num_heads'] > 0

        self.lin1 = nn.Linear(self.input_dim, self.hidden_channels * self.d)

        mlp_layers = []

        if self.num_heads <= 1:
            mlp_layers.append(nn.Linear(self.hidden_channels * self.d, self.output_dim))
        else:
            for i in range(self.num_heads-1):
                mlp_layers.append(nn.Linear(self.hidden_channels * self.d, self.hidden_channels * self.d))
                mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Linear(self.hidden_channels * self.d, self.output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)

        self.coop_layers = nn.ModuleList(
            [CSNNConv(args) for _ in range(self.layers)]
        )

        if self.layer_norm:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm([self.hidden_channels * self.d]) for _ in range(self.layers)]
            )

        if self.batch_norm:
            self.batch_norm = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_channels * self.d, affine=True) for _ in range(self.layers)]
                )

    def forward(self, batch):
        batch.x = batch.x.float()
        batch.edge_attr = batch.edge_attr.float() if batch.edge_attr is not None else None
        batch.x = F.dropout(batch.x, p=self.input_dropout, training=self.training)
        batch.x = self.lin1(batch.x)

        if self.use_act:
            batch.x = F.gelu(batch.x)

        for layer in range(self.layers):
            batch.x = self.coop_layers[layer](batch)
            if self.layer_norm:
                batch.x = self.layer_norms[layer](batch.x)
            if self.batch_norm:
                batch.x = self.batch_norm[layer](batch.x)
        
        if self.graph_level:
            batch.x = global_mean_pool(batch.x, batch.batch)

        batch.x = self.mlp(batch.x)

        return batch.x
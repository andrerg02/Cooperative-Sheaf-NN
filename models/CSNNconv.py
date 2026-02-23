import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from models.orthogonal import Orthogonal
from torch_scatter import scatter_add
import torch.nn.functional as F
import time

from models.sheaf_base import SheafDiffusion
from models.sheaf_learner import ConformalSheafLearner

class CSNNConv(SheafDiffusion, MessagePassing):
    def __init__(self,
                 args,
                 bias=False):
        SheafDiffusion.__init__(self, args)
        MessagePassing.__init__(self, aggr='add', flow='target_to_source', node_dim=0)

        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=bias)
            nn.init.orthogonal_(self.lin_right_weights.weight.data)
        else:
            self.lin_right_weights = nn.Identity()
        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=bias)
            nn.init.eye_(self.lin_left_weights.weight.data)
        else:
            self.lin_left_weights = nn.Identity()

        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=self.orth_trans)

        self.in_maps_learner = ConformalSheafLearner(
                self.d,
                self.hidden_channels,
                out_shape=(self.get_param_size(),),
                linear_emb=self.linear_emb,
                gnn_type=self.gnn_type,
                gnn_layers=self.gnn_layers,
                gnn_hidden=self.gnn_hidden,
                gnn_default=self.gnn_default,
                gnn_residual=self.gnn_residual,
                pe_size=self.pe_size,
                conformal=self.conformal,
                sheaf_act=self.sheaf_act)

        self.out_maps_learner = ConformalSheafLearner(
                self.d,
                self.hidden_channels,
                out_shape=(self.get_param_size(),),
                linear_emb=self.linear_emb,
                gnn_type=self.gnn_type,
                gnn_layers=self.gnn_layers,
                gnn_hidden=self.gnn_hidden,
                gnn_default=self.gnn_default,
                gnn_residual=self.gnn_residual,
                pe_size=self.pe_size,
                conformal=self.conformal,
                sheaf_act=self.sheaf_act)

        self.epsilons = nn.Parameter(torch.zeros((self.d, 1)))
    
    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2
    
    def restriction_maps_builder(self, edge_index, T, S, c_T, c_S):
        row, col = edge_index
        
        T = self.orth_transform(T) * c_T[:, None, None]
        S = self.orth_transform(S) * c_S[:, None, None]

        in_diag_maps = scatter_add(c_S[col] ** 2, col, dim=0, dim_size=self.graph_size)[:, None]
        out_diag_maps = scatter_add(c_T[row] ** 2, row, dim=0, dim_size=self.graph_size)[:, None]

        out_diag_sqrt_inv = (out_diag_maps + 1).pow(-0.5)
        in_diag_sqrt_inv = (in_diag_maps + 1).pow(-0.5)

        norm_T_out = T * out_diag_sqrt_inv.view(-1,1,1)
        norm_T_in = T * in_diag_sqrt_inv.view(-1,1,1)
        norm_S_out = S * out_diag_sqrt_inv.view(-1,1,1)
        norm_S_in = S * in_diag_sqrt_inv.view(-1,1,1)

        return (in_diag_maps, norm_T_in, norm_S_in), (out_diag_maps, norm_T_out, norm_S_out)

    def left_right_linear(self, x, left, right):
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, self.graph_size * self.d).t()

        x = right(x)
        return x

    def forward(self, batch):
        self.graph_size = batch.x.size(0)

        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
        x_maps = batch.x.reshape(self.graph_size, self.hidden_channels * self.d)

        to_be_T_maps, c_T = self.out_maps_learner(x_maps, batch)
        to_be_S_maps, c_S = self.in_maps_learner(x_maps, batch)

        batch.x = batch.x.view(self.graph_size * self.d, -1)
        x0 = batch.x

        L_in_comps, L_out_comps = self.restriction_maps_builder(batch.edge_index,
                                                                to_be_T_maps,
                                                                to_be_S_maps,
                                                                c_T,
                                                                c_S)
        D_out, T_out, S_out = L_out_comps
        D_in, T_in, S_in = L_in_comps
        
        c_T_norm = c_T[:, None]**2 * (D_out + 1).pow(-1)
        c_S_norm = c_S[:, None]**2 * (D_in + 1).pow(-1)

        batch.x = self.left_right_linear(batch.x, self.lin_left_weights, self.lin_right_weights)
        batch.x = batch.x.reshape(self.graph_size, self.d, self.hidden_channels)

        Sx_out = S_out @ batch.x
        TtTx = c_T_norm[..., None] * batch.x
        batch.x = self.propagate(batch.edge_index, x=TtTx, y=Sx_out, T=T_out.transpose(-2,-1))

        Sx_in = S_in @ batch.x
        StSx = c_S_norm[..., None] * batch.x
        batch.x = self.propagate(batch.edge_index, x=StSx, y=Sx_in, T=T_in.transpose(-2,-1))
        
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.use_act:
            batch.x = F.gelu(batch.x)   

        batch.x = batch.x.reshape(self.graph_size * self.d, -1)
        x0 = (1 + torch.tanh(self.epsilons).tile(self.graph_size, 1)) * x0 - batch.x
        batch.x = x0

        return batch.x.view(self.graph_size, -1)

    def message(self, x_i, y_j, T_i):
        msg = T_i @ y_j

        return x_i - msg
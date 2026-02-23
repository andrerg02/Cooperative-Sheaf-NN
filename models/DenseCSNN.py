import torch
import torch_sparse

from torch import nn
from torch_scatter import scatter_add

from models.orthogonal import Orthogonal
from models.sheaf_base import SheafDiffusion
from models.sheaf_learner import ConformalSheafLearner

class DenseCSNNConv(SheafDiffusion):

    def __init__(self, args):
        super(DenseCSNNConv, self).__init__(args)
        assert args['d'] > 1

        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(self.lin_right_weights.weight.data)
        else:
            self.lin_right_weights = nn.Identity()
        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
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

        self.weights = nn.LazyLinear(1)

        self.epsilons = nn.Parameter(torch.zeros((self.d, 1)))
    
    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2
    
    def normalise(self, diag, tril, row, col):
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        diag_sqrt_inv = (diag + 1).pow(-0.5)
        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if tril.dim() > 2 else diag_sqrt_inv.view(-1, d)
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if diag.dim() > 2 else diag_sqrt_inv.view(-1, d)
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps
    
    def get_laplacian_indices(self, edge_index):
        row, col = edge_index

        row_expand = row.view(-1, 1, 1) * self.d + torch.arange(self.d, device=self.device).view(1, 1, -1)
        row_expand = row_expand.expand(-1, self.d, -1)

        col_expand = col.view(-1, 1, 1) * self.d + torch.arange(self.d, device=self.device).view(1, -1, 1)
        col_expand = col_expand.expand(-1, -1, self.d)

        row_indices = row_expand.reshape(-1)
        col_indices = col_expand.reshape(-1)
        off_diag_indices = torch.stack([row_indices, col_indices], dim=0)

        arange_d = torch.arange(self.d, device=self.device)
        base = torch.arange(self.graph_size, device=self.device) * self.d

        diag_i = base.view(-1, 1, 1) + arange_d.view(1, 1, self.d).expand(self.graph_size, self.d, self.d)
        diag_j = base.view(-1, 1, 1) + arange_d.view(1, self.d, 1).expand(self.graph_size, self.d, self.d)
        diag_i = diag_i.reshape(-1)
        diag_j = diag_j.reshape(-1)

        diag_indices = torch.stack([diag_i, diag_j], dim=0)

        return diag_indices, off_diag_indices

    def laplacian_builder(self, edge_index, S, T, c_S, c_T):
        row, col = edge_index
        
        S = self.orth_transform(S) * c_S[:, None, None]
        T = self.orth_transform(T) * c_T[:, None, None]
        S_maps, T_maps = S[row], T[col]

        off_diag_maps = -torch.bmm(T_maps.transpose(-2,-1), S_maps)
        in_diag_maps = scatter_add(c_T[col] ** 2, col, dim=0, dim_size=self.graph_size)[:, None]
        out_diag_maps = scatter_add(c_S[row] ** 2, row, dim=0, dim_size=self.graph_size)[:, None]

        in_diag_maps, in_off_diag_maps = self.normalise(in_diag_maps, off_diag_maps, col, row)
        out_diag_maps, out_off_diag_maps = self.normalise(out_diag_maps, off_diag_maps, row, col)

        eye = torch.eye(self.d, device=self.device).unsqueeze(0)
        in_diag_maps = (in_diag_maps.expand(-1, self.d).unsqueeze(-1) * eye).view(-1)
        in_off_diag_maps = in_off_diag_maps.view(-1)
        out_diag_maps = (out_diag_maps.expand(-1, self.d).unsqueeze(-1) * eye).view(-1)
        out_off_diag_maps = out_off_diag_maps.view(-1)

        diag_indices, off_diag_indices = self.get_laplacian_indices(edge_index)

        L_in_values = torch.cat([in_off_diag_maps, in_diag_maps], dim=0)
        L_out_values = torch.cat([out_off_diag_maps, out_diag_maps], dim=0)
        L_idx = torch.cat([off_diag_indices, diag_indices], dim=1)

        return (L_in_values, L_out_values, L_idx)

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

        to_be_S_maps, c_S = self.out_maps_learner(x_maps, batch)
        to_be_T_maps, c_T = self.in_maps_learner(x_maps, batch)

        L_in, L_out, idx = self.laplacian_builder(batch.edge_index, to_be_S_maps, to_be_T_maps, c_S, c_T)

        batch.x = batch.x.view(self.graph_size * self.d, -1)
        x0 = batch.x

        batch.x = self.left_right_linear(batch.x, self.lin_left_weights, self.lin_right_weights)
        batch.x = torch_sparse.spmm(idx, L_out, batch.x.size(0), batch.x.size(0), batch.x)
        batch.x = torch_sparse.spmm(idx, L_in, batch.x.size(0), batch.x.size(0), batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.use_act:
            batch.x = F.gelu(batch.x)

        batch.x = batch.x.reshape(self.graph_size * self.d, -1)
        x0 = (1 + torch.tanh(self.epsilons).tile(self.graph_size, 1)) * x0 - batch.x
        batch.x = x0

        return batch.x.view(self.graph_size, -1)
import math
import inspect

from typing import Optional, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_scatter import scatter_add
from torch_sparse import SparseTensor

from torch_householder import torch_householder_orgqr

from torch_geometric.nn import (
    MessagePassing,
    SGConv,
    SAGEConv,
    NNConv,
    GCNConv,
    GATConv,
    GPSConv,
    GraphSAGE,
    global_mean_pool,
    global_add_pool,
    global_max_pool)

TensorTriplet = Tuple[Tensor, Tensor, Tensor]
Linear = nn.Linear
Identity = nn.Identity

class CSNN(nn.Module):
    """ The Sheaf Neural Network from `"Cooperative Sheaf Neural Networks" <https://arxiv.org/abs/2507.00647>`_ paper,
    using the :class:`CSNNConv` operator for message passing.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int or List[int]): Number of hidden channels or a list of hidden channels
        num_layers (int): Number of layers in the model.
        out_channels (int): Number of output channels.
        use_eps (bool, optional): If True, uses the adjusted residual connection. (default :obj:`True`)
        stalk_dimension (int, optional): Dimension of the sheaf stalks. (default :obj:`2`)
        input_dropout (float, optional): Dropout rate for the input features. (default :obj:`0.0`)
        dropout (float, optional): Dropout rate for the features in the model. (default :obj:`0.0`)
        act (str, optional): Activation function to use in the model. Options are :obj:`'tanh'`, :obj:`'relu'`, :obj:`'gelu'`, :obj:`'sigmoid'`,
            :obj:`'elu'`, or :obj:`'id'`. (default :obj:`'gelu'`)
        pe_size (int, optional): Size of the positional encoding to use in the model. (default :obj:`0`)
        num_heads (int, optional): Number of heads to use in the final MLP layer. If set to 1, uses a linear readout (default :obj:`1`)
        norm (str, optional): Normalization to use in the model. Options are :obj:`'none'`, :obj:`'LayerNorm'`, or :obj:`'BatchNorm'`. (default :obj:`'none'`)
        graph_level (bool, optional): If True, applies a global pooling operation at the end of the model. (default :obj:`False`)
        pooling (str, optional): Pooling operation to use at the end of the model on graph-level tasks. Options are :obj:`'mean'`, :obj:`'add'`, or :obj:`'max'`. (default :obj:`'mean'`)
    """

    def __init__(self, in_channels: int,
                 hidden_channels: int | List[int],
                 num_layers: int,
                 out_channels: int,
                 use_eps: Optional[bool] = True,
                 stalk_dimension: Optional[int] = 2,
                 input_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.0,
                 act: Optional[str] ='gelu',
                 pe_size: Optional[int] = 0,
                 num_heads: int = 1,
                 norm: Optional[str] = 'none',
                 graph_level: Optional[bool] = False,
                 pooling: Optional[str] = 'mean',
                 **kwargs):
        super().__init__()
        assert stalk_dimension > 1
        assert num_heads > 0
        if (isinstance(hidden_channels, List) and use_eps):
            raise ValueError('hidden_channels must be the same between layers to use '
                             '(ajusted) residual connection. If they are all the same, '
                             'please pass hidden_channels as a single integer')

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.use_eps = use_eps
        self.d = stalk_dimension
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.act = act
        self.pe_size = pe_size
        self.use_norm = norm != 'none'
        self.graph_level = graph_level

        if act == 'tanh':
            self.act = torch.tanh
        elif act == 'relu':
            self.act = F.relu
        elif act == 'gelu':
            self.act = F.gelu
        elif act == 'sigmoid':
            self.act = torch.sigmoid
        elif act == 'elu':
            self.act = F.elu
        elif act == 'id':
            self.act = lambda x: x
        else:
            raise ValueError(f"Unsupported act {act}")
        
        if pooling == 'mean':
            self.pooling_func = global_mean_pool
        elif pooling == 'add' or pooling == 'sum':
            self.pooling_func = global_add_pool
        elif pooling == 'max':
            self.pooling_func = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling {pooling}")

        if isinstance(hidden_channels, int):
            channel_list = [hidden_channels] * (num_layers+1)
        else:
            channel_list = hidden_channels
        
        assert len(channel_list) - 1 == num_layers
            
        self.lin1 = nn.Linear(in_channels, channel_list[0] * self.d)
        
        mlp_hidden = channel_list[-1]
        mlp_layers = []
        if num_heads <= 1:
            mlp_layers.append(nn.Linear(mlp_hidden * self.d, out_channels))
        else:
            for _ in range(num_heads-1):
                mlp_layers.append(nn.Linear(mlp_hidden * self.d, mlp_hidden * self.d))
                mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Linear(mlp_hidden * self.d, out_channels))
        
        self.mlp = nn.Sequential(*mlp_layers)

        self.coop_layers = nn.ModuleList(
            [CSNNConv(in_channels=channel_list[layer],
                     out_channels=channel_list[layer+1],
                     stalk_dimension=stalk_dimension,
                     **kwargs) for layer in range(num_layers)]
        )

        if self.use_norm:
            if norm == 'LayerNorm':
                self.norms = nn.ModuleList(
                    [nn.LayerNorm([channel_list[layer] * self.d]) for layer in range(1,num_layers+1)]
                )
            elif norm == 'BatchNorm':
                self.norms = nn.ModuleList(
                    [nn.BatchNorm1d(channel_list[layer] * self.d) for layer in range(1,num_layers+1)]
                    )

        if use_eps:
            self.epsilons = nn.ParameterList(
                [nn.Parameter(torch.zeros((self.d, 1))) for _ in range(num_layers)]
            )

    def forward(self, x: Tensor,
                edge_index: Tensor | SparseTensor,
                edge_attr: Tensor | None = None,
                batch: Optional[Tensor] = None,
                batch_size: Optional[int] = None,
                pe: Optional[Tensor] = None):
        
        if pe is None:
            pe = torch.zeros((x.size(0), 0), device=x.device, dtype=x.dtype)
        
        assert pe.size(1) == self.pe_size, \
            f'Expected positional encoding size {self.pe_size}, got {pe.size(1)}'
        
        graph_size = x.size(0)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)

        x = self.act(x)

        for layer in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x0 = x.view(graph_size * self.d, -1)
            x = self.coop_layers[layer](x, edge_index, edge_attr, pe)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.act(x)

            if self.use_eps:
                x = (1 + torch.tanh(self.epsilons[layer]).tile(graph_size, 1)) * x0 - x
            
            x = x.view(graph_size, -1)

            if self.use_norm:
                x = self.norms[layer](x)
        
        if self.graph_level:
            x = self.pooling_func(x, batch, batch_size)

        x = self.mlp(x)

        return x
    
class CSNNConv(MessagePassing):
    r"""The conformal sheaf convolutional operator from the `"Cooperative Sheaf 
    Neural Networks" <https://arxiv.org/abs/2507.00647>`_ paper.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stalk_dimension (int, optional): Dimension of the sheaf stalks. (default :obj:`2`)
        left_weights (bool, optional): If True, applies left weights to the features. (default :obj:`True`)
        right_weights (bool, optional): If True, applies right weights to the features. (default :obj:`True`)
        use_bias (bool, optional): Add bias in the weights. (default :obj:`False`)
        sheaf_act (str, optional): Activation function applied on the sheaf maps. (default :obj:`'tanh'`)
        orth_trans (str, optional): Method to learn orthogonal maps. Options are
            :obj:`'householder'`, :obj:`'matrix_exp'`, :obj:`'cayley'`, or
            :obj:`'euler'`. The  :obj:`'euler'` method can only be used if stalk_dimension is 2 or 3. (default :obj:`'householder'`)
        linear_emb (bool, optional): Use a linear+act embedding/readout when learning the sheaf. (default :obj:`True`)
        gnn_type (str, optional): Type of GNN to use for learning the sheaf. Options are
            :obj:`'SAGE'`, :obj:`'GCN'`, :obj:`'GAT'`, :obj:`'NNConv'`, :obj:`'SGC'`, or :obj:`'SumGNN'`. (default :obj:`'SAGE'`)
        gnn_layers (int, optional): Number of GNN layers to use for learning the sheaf. (default :obj:`1`)
        gnn_hidden (int, optional): Number of hidden channels in the GNN layers. (default :obj:`32`)
        gnn_default (int, optional): Set this to 0 to use a custom GNN to learn the restriction maps.
            To reproduce the experiments in the paper, use either 1 or 2. (default :obj:`False`)
        gnn_residual (bool, optional): Use residual connections in the GNN layers. (default :obj:`False`)
        pe_size (int, optional): Size of the positional encoding to use in the GNN layers. (default :obj:`0`)
        conformal (bool, optional): Whether to learn conformal restriction maps. If set to
            :obj:`False`, the model learns in and out flat bundles (default :obj:`True`)
        print_params (bool, optional): Print the parameters of the network. (default :obj:`False`)
    """

    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 stalk_dimension: Optional[int]  = 2,
                 left_weights:    Optional[bool] = True,
                 right_weights:   Optional[bool] = True,
                 use_bias:        Optional[bool] = False,
                 sheaf_act:       Optional[str]  = 'tanh',
                 orth_trans:      Optional[str]  = 'householder',
                 linear_emb:      Optional[bool] = True,
                 gnn_type:        Optional[str]  = 'SAGE',
                 gnn_layers:      Optional[int]  = 1,
                 gnn_hidden:      Optional[int]  = 32,
                 gnn_default:     Optional[bool] = False,
                 gnn_residual:    Optional[bool] = False,
                 pe_size:         Optional[int]  = 0,
                 conformal:       Optional[bool] = True,
                 print_params:    Optional[bool] = False
                 ):
        MessagePassing.__init__(self, aggr='add',
                                flow='target_to_source',
                                node_dim=0)

        self.d = stalk_dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.right_weights = right_weights
        self.left_weights = left_weights
        self.orth_trans = orth_trans

        if print_params:
            print('------------------------------------------------')
            print('Running CSNNConv with the following parameters:')
            args = inspect.getfullargspec(self.__init__).args
            values = inspect.getargvalues(inspect.currentframe())
            for arg,value in zip(args, values.locals.values()):
                if arg != 'self':
                    print(f"{arg}: {value}")
            print('------------------------------------------------')

        if in_channels != out_channels:
            assert right_weights, \
            f'The right_weights changes from in_channels to out_channels \
            Either set right_weights=True or ensure in_channels == out_channels.'

        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.in_channels,
                                               self.out_channels,
                                               bias=use_bias)
            nn.init.orthogonal_(self.lin_right_weights.weight.data)
        else:
            self.lin_right_weights = nn.Identity()

        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d,
                                              self.d,
                                              bias=use_bias)
            nn.init.eye_(self.lin_left_weights.weight.data)
        else:
            self.lin_left_weights = nn.Identity()

        self.orth_transform = Orthogonal(d=self.d,
                                         orthogonal_map=orth_trans)

        self.in_maps_learner = ConformalSheafLearner(
                self.d,
                self.in_channels,
                out_shape = (self.get_param_size(),),
                linear_emb = linear_emb,
                gnn_type = gnn_type,
                gnn_layers = gnn_layers,
                gnn_hidden = gnn_hidden,
                gnn_default = gnn_default,
                gnn_residual = gnn_residual,
                pe_size = pe_size,
                conformal = conformal,
                sheaf_act = sheaf_act)

        self.out_maps_learner = ConformalSheafLearner(
                self.d,
                self.in_channels,
                out_shape = (self.get_param_size(),),
                linear_emb = linear_emb,
                gnn_type = gnn_type,
                gnn_layers = gnn_layers,
                gnn_hidden = gnn_hidden,
                gnn_default = gnn_default,
                gnn_residual = gnn_residual,
                pe_size = pe_size,
                conformal = conformal,
                sheaf_act = sheaf_act)
    
    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2
    
    def restriction_maps_builder(self, edge_index: Tensor | SparseTensor,
                                 T:   Tensor, S:   Tensor,
                                 c_T: Tensor, c_S: Tensor) -> Tuple[TensorTriplet, TensorTriplet]:
        
        if isinstance(edge_index, SparseTensor):
            assert edge_index.size(0) == edge_index.size(1)
            row, col, _ = edge_index.coo()
        else:
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
    
    def left_right_linear(self, x: Tensor, left: Linear | Identity,
                          right: Linear | Identity) -> Tensor:
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, self.graph_size * self.d).t()

        x = right(x)
        return x

    def forward(self, x: Tensor, edge_index: Tensor | SparseTensor,
                edge_attr: Optional[Tensor], pe: Optional[Tensor] = None) -> Tensor:
        
        self.graph_size = x.size(0)

        assert x.view(self.graph_size, -1).size(1) == self.in_channels * self.d, \
            f'Expected input size {self.in_channels * self.d}, got {x.view(self.graph_size, -1).size(1)}. \
            Are you embedding graph features into sheaf features?'

        x_maps = x.reshape(self.graph_size, self.in_channels * self.d)

        to_be_T_maps, c_T = self.out_maps_learner(x_maps, edge_index, edge_attr, pe)
        to_be_S_maps, c_S = self.in_maps_learner(x_maps, edge_index, edge_attr, pe)

        x = x.view(self.graph_size * self.d, -1)

        L_in_comps, L_out_comps = self.restriction_maps_builder(edge_index, to_be_T_maps,
                                                                to_be_S_maps, c_T, c_S)
        D_out, T_out, S_out = L_out_comps
        D_in, T_in, S_in = L_in_comps
        
        c_T_norm = c_T[:, None]**2 * (D_out + 1).pow(-1)
        c_S_norm = c_S[:, None]**2 * (D_in + 1).pow(-1)

        x = self.left_right_linear(x, self.lin_left_weights, self.lin_right_weights)
        x = x.reshape(self.graph_size, self.d, self.out_channels)

        Sx_out = S_out @ x
        TtTx = c_T_norm[..., None] * x
        x = self.propagate(edge_index, x=TtTx, y=Sx_out, T=T_out.transpose(-2,-1))

        Sx_in = S_in @ x
        StSx = c_S_norm[..., None] * x
        x = self.propagate(edge_index, x=StSx, y=Sx_in, T=T_in.transpose(-2,-1))

        return x.reshape(self.graph_size * self.d, -1)

    def message(self, x_i, y_j, T_i):
        msg = T_i @ y_j

        return x_i - msg
  
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

class Orthogonal(nn.Module):
    """Based on https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#orthogonal"""
    def __init__(self, d, orthogonal_map):
        super().__init__()
        assert orthogonal_map in ["matrix_exp", "cayley", "householder", "euler"]
        self.d = d
        self.orthogonal_map = orthogonal_map

    def get_2d_rotation(self, params, det=1):
        # assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 1
        sin = torch.sin(params * 2 * math.pi)
        cos = torch.cos(params * 2 * math.pi)
        if det == 1:
            return torch.cat([cos, -sin,
                            sin, cos], dim=1).view(-1, 2, 2)
        if det == -1:
            return torch.cat([-cos, sin,
                            sin, cos], dim=1).view(-1, params.size(1), 2, 2)

    def get_3d_rotation(self, params):
        assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 3

        alpha = params[:, 0].view(-1, 1) * 2 * math.pi
        beta = params[:, 1].view(-1, 1) * 2 * math.pi
        gamma = params[:, 2].view(-1, 1) * 2 * math.pi

        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta),  torch.cos(beta)
        sin_g, cos_g = torch.sin(gamma), torch.cos(gamma)

        return torch.cat(
            [cos_a*cos_b, cos_a*sin_b*sin_g - sin_a*cos_g, cos_a*sin_b*cos_g + sin_a*sin_g,
             sin_a*cos_b, sin_a*sin_b*sin_g + cos_a*cos_g, sin_a*sin_b*cos_g - cos_a*sin_g,
             -sin_b, cos_b*sin_g, cos_b*cos_g], dim=1).view(-1, 3, 3)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.orthogonal_map != "euler":
            # Construct a lower diagonal matrix where to place the parameters.
            offset = -1 if self.orthogonal_map == 'householder' else 0
            tril_indices = torch.tril_indices(row=self.d, col=self.d, offset=offset, device=params.device)
            new_params = torch.zeros(
                (params.size(0), self.d, self.d), dtype=params.dtype, device=params.device)
            new_params[:, tril_indices[0], tril_indices[1]] = params
            params = new_params

        if self.orthogonal_map == "matrix_exp" or self.orthogonal_map == "cayley":
            # We just need n x k - k(k-1)/2 parameters
            params = params.tril()
            A = params - params.transpose(-2, -1)
            # A is skew-symmetric (or skew-hermitian)
            if self.orthogonal_map == "matrix_exp":
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == "cayley":
                # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
                Id = torch.eye(self.d, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
        elif self.orthogonal_map == 'householder':
            eye = torch.eye(self.d, device=params.device).unsqueeze(0).repeat(params.size(0), 1, 1)
            A = params.tril(diagonal=-1) + eye
            Q = torch_householder_orgqr(A)
        elif self.orthogonal_map == 'euler':
            assert 2 <= self.d <= 3
            if self.d == 2:
                Q = self.get_2d_rotation(params)
            else:
                Q = self.get_3d_rotation(params)
        else:
            raise ValueError(f"Unsupported transformations {self.orthogonal_map}")
        return Q

class ConformalSheafLearner(nn.Module):
    """Learns a conformal sheaf passing node features through a GNN or MLP + activation."""

    def __init__(self, d:         int,
                 hidden_channels: int,
                 out_shape:       Tuple[int],
                 linear_emb:      bool,
                 gnn_type:        str,
                 gnn_layers:      int,
                 gnn_hidden:      int,
                 gnn_default:     bool,
                 gnn_residual:    bool,
                 pe_size:         int,
                 conformal:       bool,
                 sheaf_act:       str = 'tanh'):
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
        self.sheaf_act = sheaf_act
        out_channels = int(np.prod(out_shape) + 1) if conformal else int(np.prod(out_shape))

        if sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'relu':
            self.act = F.relu
        elif sheaf_act == 'gelu':
            self.act = F.gelu
        elif sheaf_act == 'sigmoid':
            self.act = torch.sigmoid
        elif sheaf_act == 'elu':
            self.act = F.elu
        elif sheaf_act == 'id':
            self.act = lambda x: x
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

        if gnn_layers > 0:
            self.gnn = self.get_layer_type(gnn_type)
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
                    self.phi = self.gnn_builder(gnn_type, gnn_hidden, gnn_hidden, gnn_layers)
                    self.emb2 = nn.Linear(gnn_hidden, out_channels)
                else:
                    self.phi = self.gnn_builder(gnn_type, (hidden_channels + pe_size) * d, out_channels, gnn_layers, gnn_hidden)

        else:
            self.phi = nn.Sequential(
                nn.Linear((hidden_channels + pe_size) * d, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, out_channels)
            )

    def get_layer_type(self, layer_type):
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
    
    def gnn_builder(self, gnn_type, in_channels, out_channels, num_layers, hidden_channels=None):
        gnn = self.get_layer_type(gnn_type)
        layers = nn.ModuleList()
        if hidden_channels is None or num_layers == 1:
            if gnn_type == 'GPS':
                raise NotImplementedError("Lacking GPSConv setup.")
            elif gnn_type == 'NNConv':
                edge_net = nn.LazyLinear(in_channels*out_channels)
                for i in range(num_layers):
                    layers.append(gnn(in_channels, out_channels, nn=edge_net, aggr='add'))
            elif gnn_type == 'SGC':
                layers = gnn(in_channels, out_channels, K=num_layers)
            elif gnn_type in ['GCN', 'GAT', 'SAGE', 'SumGNN']:
                for i in range(num_layers):
                    layers.append(gnn(in_channels, out_channels))
            else:
                raise ValueError(f"Unsupported GNN layer type: {gnn_type}")
        else:
            if gnn_type == 'GPS':
                raise NotImplementedError("GPSConv is not implemented.")
            elif gnn_type == 'NNConv':
                edge_net = nn.LazyLinear(in_channels*hidden_channels)
                layers.append(gnn(in_channels, hidden_channels, nn=edge_net, aggr='add'))
                edge_net = nn.LazyLinear(hidden_channels**2)
                for i in range(num_layers-2):
                    layers.append(gnn(hidden_channels, hidden_channels, nn=edge_net, aggr='add'))
                edge_net = nn.LazyLinear(hidden_channels*out_channels)
                layers.append(NNConv(hidden_channels, out_channels, nn=edge_net, aggr='add'))
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

    def forward(self, x, edge_index, edge_attr=None, pe=None):
        pe = pe if pe is not None else torch.empty(x.size(0), 0, device=x.device)
        maps = torch.cat([x, pe], -1)
        
        if self.gnn_layers > 0:
            sig = inspect.signature(self.gnn.forward)

        if self.gnn_layers > 0:
            sig = inspect.signature(self.gnn.forward)
            if self.gnn_default:
                maps = self.phi(maps, edge_index)
            elif self.linear_emb:
                maps = self.emb1(maps)
                maps = F.gelu(maps)
                if self.layer_type != 'SGC':
                    for layer in range(self.gnn_layers):
                        prev = maps
                        if edge_attr is not None and 'edge_attr' in sig.parameters:
                            maps = self.phi[layer](maps, edge_index, edge_attr=edge_attr)
                        else:
                            maps = self.phi[layer](maps, edge_index)
                        maps = F.gelu(maps)
                        if self.residual:
                            maps = maps + prev
                else:
                    maps = self.phi(maps, edge_index)
                    maps = F.gelu(maps)
                maps = self.emb2(maps)
            else:
                if self.layer_type != 'SGC':
                    for layer in range(self.gnn_layers):
                        prev = maps
                        if edge_attr is not None and 'edge_attr' in sig.parameters:
                            maps = self.phi[layer](maps, edge_index, edge_attr=edge_attr)
                        else:
                            maps = self.phi[layer](maps, edge_index)
                        maps = F.gelu(maps) if (self.gnn_layers != 1 and layer != self.gnn_layers - 1) else maps
                        if self.residual and layer not in [self.gnn_layers - 1, 0]:
                            maps = maps + prev
                else:
                    if self.gnn_layers == 1:
                        maps = self.phi(maps, edge_index)
                    else:
                        for layer in self.phi:
                            maps = layer(maps, edge_index)
                            maps = F.gelu(maps) if layer != self.phi[-1] else maps
        else:
            maps = self.phi(maps)

        if self.conformal:
            return self.act(maps[:, :-1]), torch.exp(maps[:, -1].clamp_max(np.log(10)))
        else:
            return self.act(maps), torch.ones(maps.size(0), device=x.device)
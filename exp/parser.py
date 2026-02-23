# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from distutils.util import strtobool
import argparse


def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')


def get_parser():
    parser = argparse.ArgumentParser()
    # Optimisation params
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw'], default='adam')
    parser.add_argument('--scheduler', type=str, choices=['cosine_with_warmup', 'reduce_on_plateau', 'step_lr', 'none'], default='none')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--lr_decay_patience', type=int, default=20)
    parser.add_argument('--clip_grad', type=float, default=0)
    parser.add_argument('--min_acc', type=float, default=0.0,
                        help="Minimum test acc on the first fold to continue training.")
    parser.add_argument('--stop_strategy', type=str, choices=['loss', 'acc'], default='acc')

    # Model configuration
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=20)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--use_act', dest='use_act', type=str2bool, default=True)
    parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        default='householder', help="Parametrisation to use for the orthogonal group.")
    parser.add_argument('--sheaf_act', type=str, default="tanh", help="Activation to use in sheaf learner.")
    parser.add_argument('--edge_weights', dest='edge_weights', type=str2bool, default=False,
                        help="Learn edge weights for connection Laplacian")
    parser.add_argument('--num_heads', type=int, default=1)
    
    # Sheaf learner parameters
    parser.add_argument('--linear_emb', type=str2bool, default=False)
    parser.add_argument('--gnn_type', type=str, choices=['GCN', 'GAT', 'SAGE', 'SGC', 'NNConv', 'SumGNN', 'GPS'], default='SAGE')
    parser.add_argument("--gnn_layers", type=int, default=0)
    parser.add_argument("--gnn_hidden", type=int, default=16)
    parser.add_argument("--gnn_default", type=int, default=1,
                        help="Use default settings to reproduce results in the paper." \
                        "1 for standard SAGE; 2 for SAGE with GELU, project and dropout=0.2" \
                        "0 for False, i.e. custom setup using linear_emb/gnn_type")
    parser.add_argument('--gnn_residual', type=str2bool, default=False)
    parser.add_argument("--pe_size", type=int, default=0)
    parser.add_argument('--pe_type', type=str, choices=['RWSE', 'LapPE'], default='RWSE',
                        help="Type of positional encoding to use. 'RWSE' for random walk, 'LapPE' for Laplacian eigenvectors.")
    parser.add_argument('--layer_norm', type=str2bool, default=False)
    parser.add_argument('--batch_norm', type=str2bool, default=False)
    parser.add_argument('--conformal', type=str2bool, default=True)

    # Experiment parameters
    parser.add_argument('--dataset', default='amazon-ratings')
    parser.add_argument('--task', type=str, choices=['node_level', 'graph_level'], default='node_level')
    parser.add_argument('--task_type', type=str, choices=['multiclass', 'multilabel', 'binary'], default='multiclass')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--accum_grad', type=int, default=1)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--model', type=str, default='CoopSheaf') #choices=['CoopSheaf', "gcn", "gat", "sage"], default='CoopSheaf')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=10)
    
    return parser

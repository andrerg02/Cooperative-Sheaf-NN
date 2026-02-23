"""Data utils functions for pre-processing and data loading."""
import os
import torch
import json
import numpy as np
import torch_geometric.transforms as T

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import download_url

def order_edge_index(edge_index):
    row1, row2 = edge_index[0], edge_index[1]
    mask = row1 > row2
    edge_index[0, mask], edge_index[1, mask] = edge_index[1, mask], edge_index[0, mask]
    
    return edge_index

def load_heterophilic_data(dataset_str):
    path = 'utils/heterophilic-data'
    if not os.path.exists(path):
        os.makedirs(path)

    if dataset_str == 'squirrel':
            dataset_str = 'squirrel_filtered_directed'
    elif dataset_str == 'chameleon':
        dataset_str = 'chameleon_filtered_directed'
    
    if not os.path.isfile(path+dataset_str+'.npz'):
        url = 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data'
        download_url(f'{url}/{dataset_str}.npz', os.path.join(ROOT_DIR, 'heterophilic-data'))

    if dataset_str == 'squirrel':
        dataset_str = 'squirrel_filtered_directed'
    elif dataset_str == 'chameleon':
        dataset_str = 'chameleon_filtered_directed'
    data = np.load(os.path.join('utils/heterophilic-data', f'{dataset_str.replace("-", "_")}.npz'))
    features = torch.tensor(data['node_features'])
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges']).t()
    full_edges = torch.unique(torch.cat([edges, edges.flip(0)], dim=1), dim=1)
    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])

    dataset = Data(x=features,
                   edge_index=full_edges if dataset_str not in ['roman_empire',
                                                                'squirrel_filtered_directed',
                                                                'chameleon_filtered_directed'] else edges,
                   y=labels,
                   train_mask=train_masks,
                   val_mask=val_masks,
                   test_mask=test_masks)
    
    if dataset_str in ['actor', 'wisconsin', 'texas', 'cornell']:
        dataset = T.NormalizeFeatures()(dataset)
    
    loader = DataLoader([dataset], batch_size=1, shuffle=False)

    return [loader]

def load_tud_data(dataset_str, batch_size, fold, split='train'):
    dataset_str = dataset_str.upper()
    assert dataset_str in ['ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'NCI1', 'PROTEINS', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    
    data_name = dataset_str.replace('-', '_').upper()
    dataset = torch.load(os.path.join('tud-data', f'{data_name}.pt'), weights_only=False)
    original_fold_dict = json.load(open(f'folds/{data_name}_splits.json', "r"))[fold]
    model_selection_dict = original_fold_dict['model_selection'][0]
    if split in ['train', 'validation']:
        data_list = [dataset[idx] for idx in model_selection_dict[split]]
    else:
        data_list = [dataset[idx] for idx in original_fold_dict[split]]
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    return loader


def get_data(args):
    if args.task == 'graph_level':
        loaders = []
        for fold in range(args.folds):
            loader_by_fold = []
            for split in ['train', 'validation', 'test']:
                loader = load_tud_data(args.dataset, args.batch_size, fold, split)
                loader_by_fold.append(loader)
            loaders.append(loader_by_fold)
        return loaders
    elif args.task == 'node_level':
        dataset = load_heterophilic_data(args.dataset)
        return dataset
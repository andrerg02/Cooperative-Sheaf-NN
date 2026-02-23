import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score

from utils.cosine_scheduler import cosine_with_warmup_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

def custom_train(data, model, loss_fn, fold, dataset, epoch):
    data.pe = None
    if hasattr(data, 'laplacian_eigenvector_pe'):
        data.pe = data.laplacian_eigenvector_pe
    elif hasattr(data, 'random_walk_pe'):
        data.pe = data.random_walk_pe
    if fold is None:
        out = model(data)
        loss = loss_fn(out, data.y)
    else:
        mask = data.train_mask[fold]
        out = model(data)
        loss = loss_fn(out[mask], data.y[mask])
    
    del out
    return loss

def custom_test(data, model):
    data.pe = None
    if hasattr(data, 'laplacian_eigenvector_pe'):
        data.pe = data.laplacian_eigenvector_pe
    elif hasattr(data, 'random_walk_pe'):
        data.pe = data.random_walk_pe

    out = model(data)
    return out, data.y

def get_scores(label, pred, task_type):
    if task_type == 'binary':
        return roc_auc_score(label.cpu().numpy(), pred.sigmoid().squeeze(1).cpu().numpy())
    elif task_type == 'multiclass':
        return pred.max(1)[1].eq(label).sum().item() / label.size(0)
    elif task_type == 'multilabel':
        return average_precision_score(label.cpu().numpy(), pred.sigmoid().cpu().numpy())
    else:
        raise ValueError(f"Unknown task: {task_type}")

def get_optimizer(model, optimizer, lr, weight_decay):
        if optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        return optimizer

def get_scheduler(optimizer, scheduler_name, max_epoch, patience):
        if scheduler_name == 'cosine_with_warmup':
            print("Using cosine with warmup scheduler")
            scheduler = cosine_with_warmup_scheduler(optimizer, num_warmup_epochs=5, max_epoch=max_epoch)
        elif scheduler_name == 'reduce_on_plateau':
            print("Using ReduceLROnPlateau scheduler")
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-5, patience=patience)
        elif scheduler_name == 'step_lr':
            print("Using StepLR scheduler")
            scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return scheduler

def get_loss_fn(task, task_type):
    if task == 'graph_level':
        if task_type == 'multilabel':
            loss_fn = F.binary_cross_entropy_with_logits
        elif task_type == 'multiclass':
            loss_fn = F.cross_entropy
            #raise ValueError(f"Unknown task type for graph level: {self.args.task_type}")
    elif task == 'node_level':
        if task_type == 'binary':
            loss_fn = lambda x,y: F.binary_cross_entropy(x.sigmoid().squeeze(1), y.float())
        elif task_type == 'multiclass':
            loss_fn = F.cross_entropy
    else:
        raise ValueError(f"Unknown task type for node level: {task_type}")
    return loss_fn
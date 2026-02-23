#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm
import time

from models.coopshv_model import CSNN

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from utils.data_utils import get_data
from torch.nn.utils import clip_grad_norm_
from utils.exp_utils import (
    custom_train,
    custom_test,
    get_scores,
    get_optimizer,
    get_scheduler,
    get_loss_fn
    )


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

PEAK = True

class Experiment(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
        self.set_seeds()
        self.data = get_data(args)
        first_batch = next(iter(self.data[0])) if self.args.task == 'node_level' else next(iter(self.data[0][0]))
        print(first_batch, "Example data from train loader")
        self.add_extra_args(first_batch)

        self.num_ensemble = 3

        print(f'Experimenting with {args.model} model')

    def set_seeds(self):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    def add_extra_args(self, first_batch):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        self.args.sha = sha
        self.args.input_dim = first_batch.x.size(1)
        classes = first_batch.y.unique().size(0)
        self.args.output_dim = 1 if classes == 2 else classes
        self.args.device = self.device

    def train(self, model, optimizer, train_loader, loss_fn, epoch, clip_grad=0, fold=None):
        model.train()
        optimizer.zero_grad()

        global PEAK

        for i, data in enumerate(train_loader):
            data = data.to(self.device)
            loss = custom_train(data, model, loss_fn, fold, self.args.dataset, epoch)

            loss = loss / self.args.accum_grad
            loss.backward()

            if (i + 1) % self.args.accum_grad == 0 or (i + 1) == len(train_loader):
                if clip_grad > 0:
                    clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()
        
        if PEAK:
            ##get peak memory
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 1024**2
            print("----------------------------------")
            print(f"Peak memory usage: {mem:.2f} MB")
            print("----------------------------------")
            PEAK = False

    def test(self, model, data_loader, loss_fn, fold=None):
        model.eval()
        if fold is not None:
            accs, losses = [[],[],[]], [[],[],[]]
        else:
            accs, losses = [[]], [[]]
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                pred, true = custom_test(data, model)
                c = 0
                if fold is not None:
                    for _, full_mask in data('train_mask', 'val_mask', 'test_mask'):
                        mask = full_mask[fold]
                        masked_pred = pred[mask]
                        masked_true = true[mask]
                        loss = loss_fn(masked_pred, masked_true)
                        accs[c].append(get_scores(masked_true, masked_pred, self.args.task_type))
                        losses[c].append(loss.item())
                        c += 1
                else:
                    loss = loss_fn(pred, true)
                    accs[0].append(get_scores(true, pred, self.args.task_type))
                    losses[0].append(loss.item())

            accs = np.mean(accs, axis=1)
            losses = np.mean(losses, axis=1)

            if len(accs) > 1:
                return accs, losses
            else:
                return accs[0], losses[0]

    def get_model_cls(self):
        if args.model == "CoopSheaf":
            model_cls = CSNN
        else:
            raise ValueError(f'Unknown model {args.model}')
        return model_cls

    def run_exp(self, cfg, fold):
        model_cls = self.get_model_cls()
        model = model_cls(cfg)
        model = model.to(self.device)
        #model = model.to(torch.double)

        optimizer = get_optimizer(model, self.args.optimizer, self.args.lr, self.args.weight_decay)
        if self.args.scheduler != 'none':
            scheduler = get_scheduler(optimizer, self.args.scheduler, self.args.epochs, self.args.lr_decay_patience)
        loss_fn = get_loss_fn(self.args.task, self.args.task_type)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Total number of parameters in {self.args.model}: {params}")

        epoch = 0
        best_val_acc = test_acc = 0
        best_val_loss = float('inf')
        best_epoch = 0
        bad_counter = 0

        times = []
        for epoch in range(self.args.epochs):
            start = time.perf_counter()
            if self.args.task == 'graph_level':
                fold_data = self.data[fold]
                train_loader = fold_data[0]
                val_loader = fold_data[1]
                test_loader = fold_data[2]

                self.train(model, optimizer, train_loader, loss_fn, epoch, self.args.clip_grad)

                train_acc, train_loss = self.test(model, train_loader, loss_fn)
                val_acc, val_loss = self.test(model, val_loader, loss_fn)
                tmp_test_acc, tmp_test_loss = self.test(model, test_loader, loss_fn)

            elif self.args.task == 'node_level':
                self.train(model, optimizer, self.data[0], loss_fn, epoch, self.args.clip_grad, fold)

                [train_acc, val_acc, tmp_test_acc], [train_loss, val_loss, tmp_test_loss] = \
                    self.test(model, self.data[0], loss_fn, fold)

            if fold == 0:
                res_dict = {
                    f'fold{fold}_train_acc': train_acc,
                    f'fold{fold}_train_loss': train_loss,
                    f'fold{fold}_val_acc': val_acc,
                    f'fold{fold}_val_loss': val_loss,
                    f'fold{fold}_tmp_test_acc': tmp_test_acc,
                    f'fold{fold}_tmp_test_loss': tmp_test_loss,
                }
                wandb.log(res_dict, step=epoch)

            if self.args.scheduler == 'reduce_on_plateau':
                scheduler.step(best_val_acc)
            elif self.args.scheduler == 'cosine_with_warmup':
                scheduler.step()

            new_best_trigger = val_acc > best_val_acc if self.args.stop_strategy == 'acc' else val_loss < best_val_loss
            if new_best_trigger:
                best_val_acc = val_acc
                best_val_loss = val_loss
                test_acc = tmp_test_acc
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1
            
            end = time.perf_counter()
            times.append(end - start)
            if epoch % args.print_freq == 0 or epoch == self.args.epochs - 1:
                print(f"Epoch {epoch} | Train acc: {train_acc:.4f} | Train loss: {train_loss:.4f} | ")
                print(f"Val acc: {val_acc:.4f} | Val loss: {val_loss:.4f} | Test acc: {tmp_test_acc:.4f} | Test loss: {tmp_test_loss:.4f}")
                print(f"Best epoch so far: {best_epoch} | Best val acc: {best_val_acc:.4f} | Best test acc: {test_acc:.4f}")
                print(f"Average time/epoch: {np.mean(times):.2f} seconds in {len(times)} epochs. Total of {np.sum(times):.2f} seconds")
                times = []  # Reset times for the next epoch

            if bad_counter == self.args.early_stopping:
                break

        print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
        print(f"Test acc: {test_acc:.4f}")
        print(f"Best val acc: {best_val_acc:.4f}")

        wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})

        return test_acc, best_val_acc

    def run(self):
        results = []
        print(f"Running with wandb account: {self.args.entity}")
        wandb.init(project="cooperative_sheaf_homophilic", config=vars(self.args), entity=self.args.entity)
        wandb.config['sha'] = self.args.sha
        wandb.config['input_dim'] = self.args.input_dim
        wandb.config['output_dim'] = self.args.output_dim
        print(self.args)
        for fold in tqdm(range(self.args.folds)):
            test_acc, best_val_acc = self.run_exp(wandb.config, fold)
            results.append([test_acc, best_val_acc])

        test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

        wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
        wandb.log(wandb_results)
        wandb.finish()

        model_name = self.args.model #if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
        print(f'{model_name} on {self.args.dataset} | SHA: {self.args.sha}')
        print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    Experiment(args).run()

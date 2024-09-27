# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:25:58 2022

@author: Fanding Xu
"""

import time
import os
import numpy as np
import torch
from IPython.display import display, Image
from rdkit import Chem
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, mean_squared_error, mean_absolute_error, f1_score, average_precision_score
from collections import defaultdict
from databuild import GenerateData, comps_visualize
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from IPython.display import display, Image
from databuild import TaskConfig, GenerateDataset, MolDatasetInMemory, RandomSplit, set_seed, RandomScaffoldSplit, LabelNormalizer

class Trainer():
    def __init__(self, args, model, device):
        self.args = args
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=args.lr_reduce_rate,
                                                                    patience=args.lr_reduce_patience, min_lr=1e-6)
        self.device = device
        if args.dataset in ['bbbp', 'bace', 'clintox', 'hiv', 'tox21', 'muv', 'sider', 'covid', 'muta']:
            self.fit_and_test = self._train_test_cls
            self.loss_function = torch.nn.BCEWithLogitsLoss() # reduction = "none"
        else:
            self.fit_and_test = self._train_test_reg
            if args.dataset in ['esol', 'freesolv', 'lipo']:
                self.loss_function = torch.nn.MSELoss()   
            else:
                self.loss_function = torch.nn.L1Loss()


    def _train_test_cls(self, loader_tr, loader_va, loader_te, log_train_results=False,
                   load_path=None, save_path='/content/buffer/fintune_temp.pt'):
        if load_path is not None:
            if os.path.exists(load_path):
                self.model.GNN.load_state_dict(torch.load(load_path), strict=False)
                print("********** Pre-train Model Loaded **********")   
        self.losses_tr = []
        self.losses_va = []
        self.metric_tr = []
        self.metric_va = []
        best = None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for epoch in range(1, self.args.epochs+1):
            tic = time.time()
            if epoch % 10 == 0 or epoch == 1 or epoch == self.args.epochs:
                print("Epoch: {:d}/{:d}".format(epoch, self.args.epochs), end='')
            self.model.train()
            epoch_loss = 0   
            for data in loader_tr:
                data = data.to(self.device)
                pred = self.model(data)
                y = data.y
                if self.args.pooling not in ['AsymCheegerCut', 'Diff', 'MinCut' ,'DMoN', 'Hosc', 'just_balance']:
                    if pred.size(0) != y.size(0):
                        min_size = min(pred.size(0), y.size(0))
                        pred = pred[:min_size]
                        y = y[:min_size]
                if isinstance(pred, tuple):
                    pred, pool_loss = pred
                    pred = pred[y>=0]
                    y = y[y>=0]
                    loss = self.loss_function(pred, y) + pool_loss
                else:
                    pred = pred[y>=0]
                    y = y[y>=0]
                    loss = self.loss_function(pred, y)
                epoch_loss += loss.item()
                loss.backward() # Derive gradients.
                self.optimizer.step() # Update parameters based on gradients.
                self.optimizer.zero_grad() # Clear gradients.      
            if log_train_results:
                #print(" Training set: ", end='')
                train_loss, train_metric = self._eval_cls(loader_tr)
                self.losses_tr.append(train_loss)
                self.metric_tr.append(train_metric)
            #print("  Validation set: ", end='')
            val_loss, val_metric = self._eval_cls(loader_va)

            self.losses_va.append(val_loss)
            self.metric_va.append(val_metric)
            #print('\n', end='')
            self.scheduler.step(val_loss)
            # lr_cur = self.optimizer.param_groups[0]['lr']
            # print(lr_cur)
            if epoch == self.args.min_epoch:
                torch.save(self.model.state_dict(), save_path)
            if best is None or (best - val_loss) > 1e-6:
                best = val_loss
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), save_path)
                times = 0
            else:
                times += 1
                #print("not improved for {:d} times, current best is epoch {:d}: {:.4f}".format(times, best_epoch, best))
            toc = time.time()
            #print("time costs: {:2f}s\n".format(toc-tic))
            
            # self.draw()
            
            if times >= self.args.patience:
                break
        self.stop_epoch = epoch
        self.model.load_state_dict(torch.load(save_path), strict=False)
        te_loss, te_metric = self._eval_cls(loader_te)
        return te_loss, te_metric
    

    def _train_test_reg(self, loader_tr, loader_va, loader_te, log_train_results=False,
                        load_path=None, save_path='/content/buffer/fintune_temp.pt'):
        if load_path is not None:
            if os.path.exists(load_path):
                self.model.GNN.load_state_dict(torch.load(load_path), strict=False)
                print("********** Pre-train Model Loaded **********")   
        self.losses_tr = []
        self.losses_va = []
        self.metric_tr = []
        self.metric_va = []
        best = None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for epoch in range(1, self.args.epochs+1):
            tic = time.time()
            if epoch % 10 == 0 or epoch == 1 or epoch == self.args.epochs:
                print("Epoch: {:d}/{:d}".format(epoch, self.args.epochs), end='')
            self.model.train()
            epoch_loss = 0   
            for data in loader_tr:
                data = data.to(self.device)
                pred = self.model(data)
                y = data.y
                #if self.args.dataset == 'qm7':
                    #y = self.args.normalizer.norm(y)
                if self.args.pooling not in ['AsymCheegerCut', 'Diff', 'MinCut' ,'DMoN', 'Hosc', 'just_balance']:
                    if pred.size(0) != y.size(0):
                        min_size = min(pred.size(0), y.size(0))
                        pred = pred[:min_size]
                        y = y[:min_size]
                if isinstance(pred, tuple):
                    pred, pool_loss = pred
                    loss = self.loss_function(pred, y) + pool_loss
                else:
                    loss = self.loss_function(pred, y)
                
                epoch_loss += loss.item()
                loss.backward() # Derive gradients.
                self.optimizer.step() # Update parameters based on gradients.
                self.optimizer.zero_grad() # Clear gradients.   
            if log_train_results:
                #print(" Training set: ", end='')
                train_loss, train_metric = self._eval_cls(loader_tr)
                self.losses_tr.append(train_loss)
                self.metric_tr.append(train_metric)
            #print("  Validation set: ", end='')
            #mask = ~torch.isnan(pred) & ~torch.isnan(y)
            #pred = pred[mask]
            #y = y[mask]   
            val_loss, val_metric = self._eval_reg(loader_va)
            self.losses_va.append(val_loss)
            self.metric_va.append(val_metric)
            #print('\n', end='') 
            self.scheduler.step(val_loss)
            if epoch == self.args.min_epoch:
                torch.save(self.model.state_dict(), save_path)
            if best is None or (best - val_loss) > 1e-6:
                best = val_loss
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), save_path)
                times = 0
            else:
                times += 1
                #print("not improved for {:d} times, current best is epoch {:d}: {:.4f}".format(times, best_epoch, best))
            toc = time.time()
            #print("time costs: {:2f}s\n".format(toc-tic))
            if times >= self.args.patience:
                break    
        self.stop_epoch = epoch
        self.model.load_state_dict(torch.load(save_path), strict=False)
        #mask = ~torch.isnan(pred) & ~torch.isnan(y)
        #pred = pred[mask]
        #y = y[mask]   
        te_loss, te_metric = self._eval_reg(loader_te)
        return te_loss, te_metric
    

    @torch.no_grad()
    def _eval_cls(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
            y = batch.y
            if self.args.pooling not in ['AsymCheegerCut', 'Diff', 'MinCut' ,'DMoN', 'Hosc', 'just_balance']:
                if pred.size(0) != y.size(0):
                    min_size = min(pred.size(0), y.size(0))
                    pred = pred[:min_size]
                    y = y[:min_size]
            if isinstance(pred, tuple):
                pred, pool_loss = pred
                loss = self.loss_function(pred[y>=0], y[y>=0]) + pool_loss
            else:
                loss = self.loss_function(pred[y>=0], y[y>=0])
            loss_val += loss.item()
            y_true.append(y)
            y_scores.append(pred.sigmoid())
        loss_val /= len(loader)
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
        roc_list = []
        prc_list = []
        f1_list = []
        ap_list = []
        # pr_list = []
        # rc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_valid = y_true[:,i] >= 0
                cur_roc = roc_auc_score(y_true[is_valid,i], y_scores[is_valid,i])
                pred = (y_scores[is_valid, i] >= 0.5).astype(int)
                precision, recall, thresholds = precision_recall_curve(y_true[is_valid,i], y_scores[is_valid,i])
                cur_prc = auc(recall, precision)
                cur_f1 = f1_score(y_true[is_valid,i], pred)
                cur_ap = average_precision_score(y_true[is_valid,i], pred)
                roc_list.append(cur_roc)
                prc_list.append(cur_prc)
                f1_list.append(cur_f1)
                ap_list.append(cur_ap)
                # pr_list.append(precision)
                # rc_list.append(recall)
                
        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))
            
            
        def aver_list(x):
            return sum(x) / len(x)
        
        roc = aver_list(roc_list)
        prc = aver_list(prc_list)
        f1 = aver_list(f1_list)
        ap = aver_list(ap_list)
        # pr = aver_list(pr_list)
        # rc = aver_list(rc_list)
        results = [roc, prc, f1, ap]
        #print("loss={:.4f} ".format(loss_val), end='')
        #print("auroc={:.4f}  auprc={:.4f} ".format(roc, prc), end='')
        return loss_val, results
    
    
    
    @torch.no_grad()
    def _eval_cls2(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
            y = batch.y
            if isinstance(pred, tuple):
                pred, pool_loss = pred
                loss = self.loss_function(pred[y>=0], y[y>=0]) + pool_loss
            else:
                loss = self.loss_function(pred[y>=0], y[y>=0])
            loss_val += loss.item()
            y_true.append(y)
            y_scores.append(pred)
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
        return y_scores
    
    @torch.no_grad()
    def _eval_reg(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
            y = batch.y

            #if self.args.dataset == 'qm7':
                #y = LabelNormalizer(y, device)
            #print(f"Prediction size: {pred.size()}, Target size: {y.size()}")
            if self.args.pooling not in ['AsymCheegerCut', 'Diff', 'MinCut' ,'DMoN', 'Hosc', 'just_balance']:
                if pred.size(0) != y.size(0):
                    min_size = min(pred.size(0), y.size(0))
                    pred = pred[:min_size]
                    y = y[:min_size]
            #print(f"Prediction size: {pred.size()}, Target size: {y.size()}")
            if isinstance(pred, tuple):
                pred, pool_loss = pred
                loss = self.loss_function(pred, y) + pool_loss
                #mask = ~torch.isnan(pred) & ~torch.isnan(y)
                #pred = pred[mask]
                #y = y[mask]
            else:
                loss = self.loss_function(pred, y)
            # loss = loss + el
            loss_val += loss.item()
            y_true.append(batch.y)
            #if self.args.dataset in ['qm7', 'qm9']:
                #pred = LabelNormalizer(pred, device)
            y_scores.append(pred)
        loss_val /= len(loader)   
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

        # 检查并打印 y_true 和 y_scores 中 NaN 的数量
        nan_count_y_true = np.isnan(y_true).sum()
        nan_count_y_scores = np.isnan(y_scores).sum()
        #print(f"Number of NaN in y_true: {nan_count_y_true}")
        #print(f"Number of NaN in y_scores: {nan_count_y_scores}")
        if y_true.shape[0] != y_scores.shape[0]:
            min_size = min(y_true.shape[0], y_scores.shape[0])
            y_true = y_true[:min_size]
            y_scores = y_scores[:min_size]
        #print("loss={:.4f} ".format(loss_val), end='')
        if self.args.dataset in ['esol', 'freesolv', 'lipo']:   
            val = mean_squared_error(y_true, y_scores, squared=False)
            print("RMSE={:.4f} ".format(val), end='')
        else:
            val = mean_absolute_error(y_true, y_scores) # , multioutput='raw_values'

            # print("MAE=", end='')
            # print(val)
            print("MAE={:.4f} ".format(val), end='')
        return loss_val, val
    

























































































































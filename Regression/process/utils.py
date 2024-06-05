# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:00:24 2022

@author: Fanding Xu
"""
import numpy as np
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from collections import defaultdict
from databuild import GenerateDataset
from torch_geometric.loader import DataLoader
import matplotlib.colors as mcolors
import random
from prettytable import PrettyTable
from itertools import product

class TSNESHOW():
    def __init__(self, loaders, model, device):
        self.color = np.array( ["hotpink", "#88c999", "cyan"])
        self.edge = np.array( ["blue", "teal", "y"])
        x = []
        y = []
        model.eval()
        for loader in loaders:
            for batch in loader:
                batch = batch.to(device)
                y.append(batch.y)
                with torch.no_grad():
                    embd = model.GNN(batch)
                if isinstance(embd, tuple):
                    embd, _ = embd
                x.append(embd)
        self.x = torch.cat(x, dim=0).cpu().numpy()
        self.Y = torch.cat(y, dim=0).cpu().int().numpy().flatten()
        self.classes = np.unique(self.Y)
        tsne = TSNE(n_components=2, early_exaggeration=20, init='pca', random_state=123)
        X = tsne.fit_transform(self.x)
        X_min, X_max = X.min(0), X.max(0)
        self.X = (X - X_min) / (X_max - X_min)  # 归一化
        
    def plot(self, title=None, s=13, linewidths=0.7):
        plt.figure()
        for i, cl in enumerate(self.classes):
            mask = self.Y == cl
            plt.scatter(self.X[mask, 0], self.X[mask, 1],
                        c=self.color[i], edgecolors=self.edge[i],
                        s=s, linewidths=linewidths, label=str(i))
        # plt.legend()
        plt.xticks([])
        plt.yticks([])
        if title is not None:
            plt.title(title)


class TSNESHOW_split():
    def __init__(self, loaders, model, device, split=3):
        self.split = split
        self.color = np.array( ["hotpink", "#88c999", "cyan"])
        self.edge = np.array( ["blue", "teal", "y"])
        x = []
        y = []
        for loader in loaders:
            for batch in loader:
                batch = batch.to(device)
                y.append(batch.y)
                with torch.no_grad():
                    embd = model.GNN(batch)
                if isinstance(embd, tuple):
                    embd, _ = embd
                x.append(embd)
        x = torch.cat(x, dim=0).cpu().numpy()
        self.x = np.split(x, split, axis=1)
        self.Y = torch.cat(y, dim=0).cpu().int().numpy().flatten()
        # self.Y = np.split(Y, split)
        self.X = []
        for x in self.x:
            tsne = TSNE(n_components=2, early_exaggeration=20, init='pca', random_state=123)
            X = tsne.fit_transform(x)
            X_min, X_max = X.min(0), X.max(0)
            self.X.append((X - X_min) / (X_max - X_min))  # 归一化
        
    def plot(self, title=None, s=13, linewidths=0.7):
        for i in range(self.split):
            X = self.X[i]
            plt.figure()
            for j in range(2):
                mask = self.Y == j
                plt.scatter(X[mask, 0], X[mask, 1],
                            c=self.color[j], edgecolors=self.edge[j],
                            s=s, linewidths=linewidths, label=str(j))
            # plt.legend()
            plt.xticks([])
            plt.yticks([])
            if title is not None:
                plt.title(title[i])


class ScaffoldTSNE():
    def __init__(self, smiles_list, batch_size=64,
                 **kwargs):

        self.smiles = np.array(smiles_list)
        self._scaffold_filter(**kwargs)
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        dataset = GenerateDataset(self.mols, self.mol_labels,
                                  inMemory=True, overwrite=True, isinfo=False,
                                  filename="scaffold_tsne.pt")
        self.loader = DataLoader(dataset, batch_size, shuffle=False)
        
    def _scaffold_filter(self, min_nums=10, tol=5, include_chirality=False, seed=1234):
        max_nums = min_nums + tol
        scaffold_dict = defaultdict(list)
        mol_label = []
        for smiles in self.smiles:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
            mol_label.append(scaffold)
            scaffold_dict[scaffold].append(smiles)     
        
        self.smiles = []
        self.mol_labels = []
        self.labels = []
        i = 0
        random.seed(seed)
        for k, v in scaffold_dict.items():
            if len(v) >= min_nums:
                self.labels.append(k)
                smiles = random.sample(v, max_nums) if len(v) > max_nums else v
                self.smiles.extend(smiles)
                self.mol_labels.extend(np.ones(len(smiles), dtype=np.int32) * i)
                i += 1

    def fit(self, model, device):
        model.eval()
        y = []
        x = []
        for batch in self.loader:
            batch = batch.to(device)
            y.append(batch.y)
            with torch.no_grad():
                embd = model.GNN(batch)
            if isinstance(embd, tuple):
                embd, _ = embd
            x.append(embd)

        self.x = torch.cat(x, dim=0).cpu().numpy()
        self.Y = torch.cat(y, dim=0).cpu().int().numpy().flatten()
        tsne = TSNE(n_components=2, early_exaggeration=20, init='pca', random_state=123)
        X = tsne.fit_transform(self.x)
        X_min, X_max = X.min(0), X.max(0)
        self.X = (X - X_min) / (X_max - X_min)  # 归一化

    def plot(self, title=None, s=13, linewidths=0.7,
             show_legend=True):
        plt.figure()
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i in range(len(self.labels)):
            mask = self.Y == i
            plt.scatter(self.X[mask, 0], self.X[mask, 1],
                        c=colors[i%len(colors)], edgecolors=colors[i%len(colors)],
                        s=s, linewidths=linewidths, label=self.labels[i])
        plt.xticks([])
        plt.yticks([])
        if show_legend:
            plt.legend()
        if title is not None:
            plt.title(title)
        


class GridSearch():
    def __init__(self, search_space):
        self.results = []
        self.grid = list(product(*search_space.values()))
        head = list(search_space.keys())
        head.append('reasult')
        self.table = PrettyTable(head)
        self.table.add_rows(np.hstack([np.array(self.grid), np.empty([len(self.grid), 1], dtype=object)]))
        self.idx = 0
        
    def report(self, val):
        mean = val.mean()
        std = val.std()
        self.results.append([mean, std])
        self.table.rows[self.idx][-1] = "{:.3f} +/- {:.3f}".format(mean, std)
        print("\n=====================\nGroup {:d} over\nResults: {:.3f} +/- {:.3f}\n=====================\n\n".format(self.idx, mean, std))
        self.idx += 1
    
    def show(self):
        print(self.table)
    
    
    def conclusion(self, mode='max'):
        assert mode in ['max', 'min']
        results = np.array(self.results)[:,0]
        if mode == 'max':
            idx = results.argmax()
        else:
            idx = results.argmin()
        self.table.rows[idx][-1] = "\033[0;33;40m"+self.table.rows[idx][-1]+"\033[0m"
        print("Results Table")
        self.show()
        
        print("\nThe best hyper_para group is: ")
        print(self.table.get_string(start=idx, end=idx+1))




























































































































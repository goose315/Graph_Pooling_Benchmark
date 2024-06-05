# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:25:05 2022

@author: Fanding Xu
"""

import numpy as np
import torch
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, LeakyReLU
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dense_to_sparse
from .utils import reset


class EGIN(MessagePassing):
    def __init__(self,
                 in_channels_x,
                 in_channels_e,
                 out_channels_x,
                 out_channels_e=None,
                 edge_batchnorm=False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        if out_channels_e is None:
            out_channels_e = out_channels_x
        self.edge_batchnorm = edge_batchnorm

        self.mlp_x = Sequential(Linear(in_channels_x + out_channels_e, out_channels_x),
                                BatchNorm1d(out_channels_x),
                                ReLU(),
                                Linear(out_channels_x, out_channels_x),
                                BatchNorm1d(out_channels_x),
                                ReLU())

        self.lin_e = Linear(in_channels_e + in_channels_x, out_channels_e)
        if edge_batchnorm:
            self.bn_e = BatchNorm1d(out_channels_e)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp_x)
        self.lin_e.reset_parameters()
        if self.edge_batchnorm:
            self.bn_e.reset_parameters()

    def forward(self, x, edge_index, edge_attr, mask=None) -> Tensor:
        self.edge_attr = edge_attr
        self.edge_index = edge_index
        edge_out = self.lin_e(torch.cat([edge_attr, x[edge_index[0]]], -1))
        if self.edge_batchnorm:
            edge_out = self.bn_e(edge_out)
        edge_out = F.relu(edge_out)
        size = (x.shape[0], x.shape[0])

        x_out = self.propagate(edge_index=edge_index, size=size, x=x, edge_attr=edge_out)
        if mask is not None:
            x_out = x_out[mask]

        return x_out, edge_out

    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr

    def update(self, aggr_out, x):
        x_out = torch.cat([x, aggr_out], dim=-1)
        x_out = self.mlp_x(x_out)
        return x_out


class DenseEGIN(MessagePassing):
    def __init__(self,
                 in_channels_x,
                 in_channels_e,
                 out_channels_x,
                 out_channels_e=None,
                 edge_batchnorm=False,
                 drop_out=None,  # 添加 drop_out 参数
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        if out_channels_e is None:
            out_channels_e = out_channels_x
        self.edge_batchnorm = edge_batchnorm
        self.drop_out = drop_out  # 保存 drop_out 参数

        self.mlp_x = Sequential(Linear(in_channels_x + out_channels_e, out_channels_x),
                                ReLU(),
                                Linear(out_channels_x, out_channels_x))

        self.lin_e = Linear(in_channels_e + in_channels_x, out_channels_e)
        if edge_batchnorm:
            self.bn_e = BatchNorm1d(out_channels_e)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp_x)
        self.lin_e.reset_parameters()
        if self.edge_batchnorm:
            self.bn_e.reset_parameters()

    def forward(self, x, adj, e, mask=None, add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        e = e.unsqueeze(0) if e.dim() == 3 else e
        B, N, _ = adj.size()
        e_mask = adj.to(torch.bool)
        e_out = torch.cat([e, x.unsqueeze(2).repeat(1, 1, N, 1)], -1)
        e_out = self.lin_e(e_out) * e_mask.unsqueeze(-1)
        message = torch.einsum("bijk->bik", e_out)
        x_out = torch.cat([x, message], dim=-1)
        x_out = self.mlp_x(x_out)
        if mask is not None:
            x_out = x_out * mask.view(B, N, 1).to(x.dtype)
        return x_out, e_out






















































































































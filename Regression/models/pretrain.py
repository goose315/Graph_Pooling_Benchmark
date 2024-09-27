# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:15:31 2022

@author: Fanding Xu
"""

import torch
from layers import EGIN, EdgePropagate
from layers.utils import sum_undirected
from databuild import TaskConfig

class FormerLayer(torch.nn.Module):
    def __init__(self,
                 in_channels_x, in_channels_e,
                 out_channels_x, out_channels_e = None,
                 mlp_hidden_x = [128],
                 mlp_hidden_e = []):
        super().__init__()
        out_channels_e = out_channels_e if out_channels_e else in_channels_x    
        self.conv = EGIN(in_channels_x, in_channels_e,
                         out_channels_x, out_channels_e,
                         mlp_hidden_x = mlp_hidden_x,
                         mlp_hidden_e = mlp_hidden_e,
                         mlp_batchnorm = True)
        self.edge_propagate = EdgePropagate()
        self.fc_k = torch.nn.Linear(out_channels_e, out_channels_e)
        self.proj = torch.nn.Linear(out_channels_e, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc_k.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_attr = self.conv(x, edge_index, edge_attr)
        edge_attr = self.edge_propagate(edge_index, edge_attr)
        K = sum_undirected(edge_index, edge_attr)
        K = self.fc_k(K)
        return K


if __name__ == "__main__":  


























































































































# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:44:29 2022

@author: Fanding Xu
"""
import torch
from torch import Tensor
import math
from torch_scatter import scatter_add, scatter_mean
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_scipy_sparse_matrix


def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

def connection(edge_target, num_nodes):
    """
    This function can find which nodes will be pooled together as a cluster and return an index
    """
    csr = csr_matrix(to_scipy_sparse_matrix(edge_target, num_nodes=num_nodes))
    _, comp = connected_components(csr)
    return torch.tensor(comp, dtype=torch.int64, device=edge_target.device)

def generate_edge_batch(edge_index, batch):
    return batch.index_select(0, edge_index[0])

def sum_undirected(edge_index, edge_attr):
    idx_rev = torch.sort(edge_index[1], stable=True)[1]
    return edge_attr + edge_attr[idx_rev]

def attr_to_unit(attr, unit_info):
    num_units = unit_info.max()
    unit_attr = scatter_add(attr, unit_info[0], dim=0, dim_size=num_units)
    unit_attr2 = scatter_add(attr, unit_info[1], dim=0, dim_size=num_units+1)
    unit_attr = unit_attr + unit_attr2[:-1]
    return unit_attr

def generate_unit_batch(batch, unit_info):
    num_units = unit_info.max()
    return scatter_mean(batch, unit_info[0], dim=0, dim_size=num_units)

def sum_unique_undirected(edge_index, edge_attr=None):
    new_index, unique_index = edge_index.sort(dim=0, stable=True)[0].unique(dim=1, return_inverse=True)
    if edge_attr is not None:
        return new_index, unique_index, scatter_add(edge_attr, unique_index, dim=0)
    return new_index, unique_index


def uniform(size: int, value):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)









































































































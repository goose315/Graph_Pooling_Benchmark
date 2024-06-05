import math
import torch
import time

from torch_scatter import scatter_add, scatter_mean, scatter_softmax, scatter_max, scatter_min

from torch_geometric.nn import global_add_pool

from torch_geometric.utils import add_remaining_self_loops, to_undirected, remove_self_loops, to_dense_batch, coalesce
from .utils import (reset, connection, generate_edge_batch, sum_undirected,
                    attr_to_unit, generate_unit_batch, uniform)
from .egin_conv import EGIN  

def edge_prop(edge_index, edge_attr):
    edge_sums = scatter_add(edge_attr, edge_index[1], dim=0)[edge_index[0]]
    edge_out = edge_sums + edge_attr
    # idx_rev = torch.sort(edge_index[1], stable=True)[1]
    # edge_reduce = edge_attr[idx_rev]
    # edge_out = edge_out - edge_reduce
    
    return edge_out

def clustering(x, edge_index, perm):   
    edge_target = edge_index[..., perm] # edge_index that will be pooled
    comp = connection(edge_target, x.size(0)) # an index for which nodes will be pooled togethor 
    x_mask = torch.zeros([x.size(0)], dtype=torch.bool, device=x.device)
    edge_mask = torch.zeros([edge_index.size(1)], dtype=torch.bool, device=x.device)
    x_mask[edge_target[0]] = True
    edge_mask[perm] = True
    return comp, x_mask, edge_mask, edge_target

def edge_reduce(edge_index, edge_attr, comp, edge_mask):
    row, col = edge_index
    row = comp[row]
    col = comp[col]
    mask = edge_mask.logical_not()
    row = row[mask] # dropout the pooled edges
    col = col[mask] # dropout the pooled edges
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = edge_attr[mask]
    return edge_index, edge_attr
    
class MUSEPool(torch.nn.Module):
    def __init__(self, in_channels_x, in_channels_e,
                 out_channels_x=None, out_channels_e=None, 
                 threshold=0):
        super().__init__()
        if out_channels_x is None:
            out_channels_x = in_channels_x
        if out_channels_e is None:
            out_channels_e = in_channels_e
        self.out_channels_x = out_channels_x
        self.threshold=threshold
        # self.scoring = torch.nn.Linear(in_channels_e, 1, bias=False)
        self.S = torch.nn.Parameter(torch.Tensor(1, in_channels_e))
        self.mlp_e = torch.nn.Sequential(torch.nn.Linear(in_channels_e, in_channels_e),
                                         torch.nn.BatchNorm1d(in_channels_e),
                                         torch.nn.ReLU(),)
                                         # torch.nn.Linear(in_channels_e, out_channels_e),
                                         # torch.nn.BatchNorm1d(in_channels_e),
                                         # torch.nn.ReLU())
        self.conv = EGIN(in_channels_x, in_channels_e, out_channels_x, out_channels_e)
        # self.lin = torch.nn.Linear(in_channels_x * 2, out_channels_x)
        self.bn_x = torch.nn.BatchNorm1d(in_channels_x)
        self.bn = torch.nn.BatchNorm1d(out_channels_x)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        # self.scoring.reset_parameters()
        self.conv.reset_parameters()
        # self.lin.reset_parameters()
        self.bn_x.reset_parameters()
        self.bn.reset_parameters()
        pass
    
    def forward(self, x, edge_index, edge_attr,
                batch = None, batch_size = None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if batch_size is None:
            batch_size = batch[-1] + 1
        num_nodes = x.size(0)
        
        ################ edge_prop ################
        edge_batch = generate_edge_batch(edge_index, batch)
        edge_attr = edge_prop(edge_index, edge_attr)
        edge_attr = self.mlp_e(edge_attr)      
        
        ################ scoring ################
        _, unit_attr = to_undirected(edge_index, edge_attr, num_nodes)
        score = torch.sigmoid(self.S.matmul(unit_attr.T).view(-1) / self.S.norm(p=2, dim=-1))
        # score = torch.sigmoid(self.scoring(unit_attr).view(-1))
        self.s = score
        
        scores_mean = scatter_mean(score, edge_batch, dim=0)[edge_batch]
        perm = ((score <= self.threshold) & (score < scores_mean)).nonzero(as_tuple=False).view(-1)
        # perm = (score < scores_mean).nonzero(as_tuple=False).view(-1)
        edge_attr = edge_attr * score.unsqueeze(1)
        self.p = perm
        
        ################ shrinkage ################
        comp, x_mask, edge_mask, edge_target = clustering(x, edge_index, perm)
        batch_out = scatter_mean(batch, comp, dim=-1)
        self.comp = comp
        self.tar = edge_target
        num_nodes = comp.max().item() + 1
        x_sparse, edge_sparse = self.conv(x, edge_target, edge_attr[perm], x_mask)
        x_pool = scatter_add(x_sparse, comp[x_mask], dim=0, dim_size=num_nodes)
        # x_prsv = scatter_add(x[x_mask.logical_not()], comp[x_mask.logical_not()], dim=0, dim_size=num_nodes)
        # V = global_add_pool(x_pool, batch_out)    
        
        mask_not = x_mask.logical_not()
        x_prsv = x[mask_not]
        x_pool[comp[mask_not]] = x_prsv
        x_out = x_pool
        # x_out = self.bn_x(x_pool)
        # x_out = x_pool + x_prsv
        # x_out = self.bn_x(x_out)
        
        ################ extraction ################
        
        edge_index_out, edge_attr_out = edge_reduce(edge_index, edge_attr, comp, edge_mask)
        edge_index_out, edge_attr_out = coalesce(edge_index_out, edge_attr_out)
        xg = global_add_pool(x_out, batch_out, batch_size)
        
        # xg = self.bn(self.lin(torch.cat([xg, V], dim=-1)))
        
        
        xg = self.bn(xg)
        new_graph = [x_out, edge_index_out, edge_attr_out, batch_out]
        
        return new_graph, xg
        
        
        
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
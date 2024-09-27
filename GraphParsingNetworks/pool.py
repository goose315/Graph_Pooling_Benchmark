from torch import Tensor
from torch_scatter import segment_csr
from torch_geometric.utils import degree
import torch

def global_add_pool(x: Tensor, batch: Tensor):
    sorted, indices = batch.sort(stable=True)
    x = x[indices.long()]
    deg = degree(sorted)
    ptr = deg.cumsum(dim=0)
    ptr = torch.cat([torch.tensor([0], device=x.device), ptr], dim=0).long()
    return segment_csr(x, ptr, reduce="sum")

def global_mean_pool(x: Tensor, batch: Tensor):
    sorted, indices = batch.sort(stable=True)
    x = x[indices.long()]
    deg = degree(sorted)
    ptr = deg.cumsum(dim=0)
    ptr = torch.cat([torch.tensor([0], device=x.device), ptr], dim=0).long()
    return segment_csr(x, ptr, reduce="mean")

def global_max_pool(x: Tensor, batch: Tensor):
    sorted, indices = batch.sort(stable=True)
    x = x[indices.long()]
    deg = degree(sorted)
    ptr = deg.cumsum(dim=0)
    ptr = torch.cat([torch.tensor([0], device=x.device), ptr], dim=0).long()
    return segment_csr(x, ptr, reduce="max")

def global_min_pool(x: Tensor, batch: Tensor):
    sorted, indices = batch.sort(stable=True)
    x = x[indices.long()]
    deg = degree(sorted)
    ptr = deg.cumsum(dim=0)
    ptr = torch.cat([torch.tensor([0], device=x.device), ptr], dim=0).long()
    return segment_csr(x, ptr, reduce="min")
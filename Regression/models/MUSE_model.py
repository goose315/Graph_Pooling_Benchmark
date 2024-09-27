import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
# from torch_geometric.nn import GINConv, GINEConv
from layers import EGIN, MUSEPool
from layers.utils import generate_edge_batch, reset

class MUSEBlock(torch.nn.Module):
    def __init__(self,
                 in_channels_x, in_channels_e,
                 hidden_channels_x, hidden_channels_e=None,
                 out_channels_x=None, out_channels_e=None,
                 threshold = 0,
                 hierarchy = 0,
                 split = True):          
        super().__init__()
        if hidden_channels_e is None:
            hidden_channels_e = hidden_channels_x
        if out_channels_x is None:
            out_channels_x = hidden_channels_x
        if out_channels_e is None:
            out_channels_e = hidden_channels_e
        self.conv = EGIN(in_channels_x, in_channels_e,
                         hidden_channels_x, hidden_channels_e)
         
        self.pool = MUSEPool(hidden_channels_x, hidden_channels_e,
                             out_channels_x = out_channels_x,
                             out_channels_e = out_channels_e,
                             threshold = threshold) 
        
        
    def forward(self, graph, batch_size):
        assert len(graph) == 4, "graph should include x, edge_index, edge_attr and batch"
        x, edge_index, edge_attr, batch = graph
        x, edge_attr = self.conv(x, edge_index, edge_attr)
        graph, xg = self.pool(x, edge_index, edge_attr, batch, batch_size)
        return graph, xg


class MUSEPred(torch.nn.Module):
    def __init__(self,
                 in_channels_x, in_channels_e,
                 hidden_channels_x, hidden_channels_e,
                 out_channels_x=None, out_channels_e=None,
                 threshold = 0,
                 num_classes = 1,
                 drop_out = None,
                 block_nums = 3,
                 lin_before_conv = False):
        super().__init__()
        if out_channels_x is None:
            out_channels_x = hidden_channels_x
        if out_channels_e is None:
            out_channels_e = hidden_channels_e
        self.lin_before_conv = lin_before_conv
        self.drop_out = drop_out
        self.blocks = torch.nn.Sequential()
        in_x, in_e = in_channels_x, in_channels_e
        if lin_before_conv:
            self.lin_x = torch.nn.Linear(in_x, hidden_channels_x)
            self.bn_x = torch.nn.BatchNorm1d(hidden_channels_x)
            self.lin_e = torch.nn.Linear(in_e, hidden_channels_e)
            self.bn_e = torch.nn.BatchNorm1d(hidden_channels_e)
            in_x, in_e = hidden_channels_x, hidden_channels_e

        embd_length = 0     
        hidden_x, hidden_e = hidden_channels_x, hidden_channels_e
        for hierarchy in range(block_nums):
            if hierarchy == block_nums - 1:
                hidden_x, hidden_e = out_channels_x, out_channels_e
            self.blocks.append(MUSEBlock(in_channels_x = in_x, in_channels_e = in_e,
                                         hidden_channels_x = hidden_x, hidden_channels_e = hidden_e,
                                         threshold = threshold))
            embd_length += hidden_x
            in_x, in_e = hidden_x, hidden_e
        
        self.lin = torch.nn.Linear(embd_length, num_classes)
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(self.GNN.embd_length, self.GNN.embd_length//2),
        #                                torch.nn.ReLU(),
        #                                torch.nn.Linear(self.GNN.embd_length//2, num_classes))
        
        self.reset_parameters()
        

    def reset_parameters(self):
        if self.lin_before_conv:
            self.lin_x.reset_parameters()
            self.lin_e.reset_parameters()
        self.lin.reset_parameters()
        # reset(mlp)

    def GNN(self, data):      
        self.comps = []
        self.tars = []
        if data.batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
            batch_size = 1
        else:
            batch = data.batch
            batch_size = data.batch[-1].item() + 1
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if self.lin_before_conv:
            x = self.bn_x(self.lin_x(data.x)).relu()
            edge_attr = self.bn_e(self.lin_e(data.edge_attr)).relu()
        graph = [x, edge_index, edge_attr, batch]
        xgs = []
        
        for block in self.blocks:
            graph, xg = block(graph, batch_size)
            self.comps.append(block.pool.comp)
            self.tars.append(block.pool.tar)
            if self.drop_out is not None:
                xg = F.dropout(xg, p=self.drop_out, training=self.training)
            xgs.append(xg)
        return torch.cat(xgs, dim=-1)


    def forward(self, data):
        embd = self.GNN(data)
        pred = self.lin(embd)
        return pred



class EGINPred(torch.nn.Module):
    def __init__(self, in_channels_x, in_channels_e,
                 hidden_channels_x, hidden_channels_e,
                 out_channels_x=None, out_channels_e=None,
                 num_classes = 1,
                 drop_out = None,
                 block_nums = 3,
                 lin_before_conv = False,
                 **kargs): 
        super().__init__()
        if out_channels_x is None:
            out_channels_x = hidden_channels_x
        if out_channels_e is None:
            out_channels_e = hidden_channels_e
        self.lin_before_conv = lin_before_conv
        self.drop_out = drop_out
        self.blocks = torch.nn.Sequential()
        
        in_x, in_e = in_channels_x, in_channels_e
        if lin_before_conv:
            self.lin_x = torch.nn.Linear(in_x, hidden_channels_x)
            self.bn_x = torch.nn.BatchNorm1d(hidden_channels_x)
            self.lin_e = torch.nn.Linear(in_e, hidden_channels_e)
            self.bn_e = torch.nn.BatchNorm1d(hidden_channels_e)
            in_x, in_e = hidden_channels_x, hidden_channels_e
        
        embd_length = 0     
        hidden_x, hidden_e = hidden_channels_x, hidden_channels_e
        for hierarchy in range(block_nums):
            if hierarchy == block_nums - 1:
                hidden_x, hidden_e = out_channels_x, out_channels_e
            self.blocks.append(EGIN(in_x, in_e, hidden_x))
            embd_length += hidden_x
            in_x, in_e = hidden_x, hidden_e
        self.lin = torch.nn.Linear(embd_length, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        if self.lin_before_conv:
            self.lin_x.reset_parameters()
            self.lin_e.reset_parameters()
        self.lin.reset_parameters()


    def GNN(self, data):      

        if data.batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        else:
            batch = data.batch
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if self.lin_before_conv:
            x = self.bn_x(self.lin_x(data.x)).relu()
            edge_attr = self.bn_e(self.lin_e(data.edge_attr)).relu()
        xgs = []
        
        for block in self.blocks:
            x, edge_attr = block(x, edge_index, edge_attr)
            xg = global_add_pool(x, batch)
            if self.drop_out is not None:
                xg = F.dropout(xg, p=self.drop_out, training=self.training)
            xgs.append(xg)
        return torch.cat(xgs, dim=-1)

    def forward(self, data):
        embd = self.GNN(data)
        pred = self.lin(embd)
        return pred




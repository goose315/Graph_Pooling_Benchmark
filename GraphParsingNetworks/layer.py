import torch
from torch.nn import Linear, ReLU, Dropout, LayerNorm, Module, ModuleList
from torch_geometric.utils import subgraph, k_hop_subgraph, remove_isolated_nodes, add_self_loops, subgraph, index_to_mask, sort_edge_index, remove_self_loops, dropout_adj, segregate_self_loops
from torch_sparse import SparseTensor, transpose, coalesce
from pool import global_mean_pool, global_min_pool
import numpy as np
from sweep import GetRunTime
from torch_scatter import scatter
import numpy as np

class GNN(Module):
    def __init__(self, hidden_channel, num_layers, dropout, gnn_model, act_final):
        super().__init__()

        m = ModuleList()
        for i in range(num_layers):
            if gnn_model=='GCN':
                from layers.gcn_conv import GCNConv
                m.append(GCNConv(in_channels=hidden_channel, out_channels=hidden_channel))
            elif gnn_model=='GAT':
                from layers.gat_conv import GATConv
                m.append(GATConv(in_channels=hidden_channel, out_channels=hidden_channel))
            elif gnn_model=='GIN':
                from layers.gin_conv import GINConv
                m.append(GINConv(
                    torch.nn.Sequential(
                        Linear(hidden_channel, hidden_channel),
                        ReLU(),
                        Linear(hidden_channel, hidden_channel),
                        ReLU(),
                ), train_eps=False))
            if i < num_layers-1 or act_final==True:
                m.append(ReLU())
                m.append(Dropout(dropout))
        self.gnn = m

        self.norm = None
        if num_layers>0:
            self.norm = LayerNorm(hidden_channel)

    def forward(self, x, adj_t):
        init_x = x

        for i in range(len(self.gnn)):
            if i%3==0:
                x = self.gnn[i](x, adj_t)
            else:
                x = self.gnn[i](x)

        if self.norm!=None:
            x = self.norm(init_x+x)

        return x

class MLP(Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers, dropout, norm_mode, act_final):
        super().__init__()

        m = ModuleList()
        for i in range(num_layers):
            if num_layers==1:
                m.append(Linear(in_channel, out_channel))
            else:
                if i==0:
                    m.append(Linear(in_channel, hidden_channel))
                elif i==num_layers-1:
                    m.append(Linear(hidden_channel, out_channel))
                else:
                    m.append(Linear(hidden_channel, hidden_channel))
                if i < num_layers-1 or act_final==True:
                    if norm_mode=='insert':
                        m.append(LayerNorm(hidden_channel))
                    m.append(ReLU())
                    m.append(Dropout(dropout))
        self.mlp = m

        self.norm = None
        if num_layers>0 and norm_mode=='post':
            self.norm = LayerNorm(hidden_channel)

    def forward(self, x):
        init_x = x

        for i in range(len(self.mlp)):
            x = self.mlp[i](x)

        if self.norm!=None:
            x = self.norm(init_x+x)

        return x

class ParsingNet_GPU(Module):
    def __init__(self, channel, dropout_network, dropout_parsing, layer_parsingnet, link_ignore_self_loop):
        super().__init__()
        self.dropout_parsing = dropout_parsing
        self.link_ignore_self_loop = link_ignore_self_loop
        self.edge_net = MLP(in_channel=channel, hidden_channel=channel, out_channel=1, num_layers=layer_parsingnet, dropout=dropout_network, norm_mode='insert', act_final=False)

    # @profile
    def forward(self, x, edge_index, batch):
        # remove self-loops and then (optionally) add self-loops
        device = x.device
        row, col, edge_attr = edge_index.coo()
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        if self.link_ignore_self_loop==False:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value='mean', num_nodes=x.size(0))

        # drop edge
        edge_index, edge_attr = dropout_adj(
            edge_index = edge_index,
            edge_attr = edge_attr,
            p = self.dropout_parsing,
            force_undirected = True,
            num_nodes = batch.shape[0],
            training = self.training
        )

        # compute edge score (on the dropped graph)
        edge_score = x[edge_index[0]] * x[edge_index[1]]
        edge_score = self.edge_net(edge_score)
        edge_score = torch.sigmoid(edge_score).squeeze(dim=-1)
        num_nodes_full = batch.shape[0]

        device_original = device
        device = 'cpu'
        edge_index = edge_index.to(device)
        edge_score = edge_score.to(device)
        batch = batch.to(device)
        torch.set_num_threads(1)

        # remove isolated nodes and save their masks
        edge_index_temp, edge_score_temp, mask = remove_isolated_nodes(edge_index, edge_score, num_nodes=num_nodes_full)
        mask_nodes = (~mask).nonzero(as_tuple=False).view(-1)
        unmask_nodes = (mask).nonzero(as_tuple=False).view(-1)
        num_nodes_connected = unmask_nodes.shape[0]

        # get edges and scores for self-loop on isolated nodes
        edge_index_, edge_score_, edge_mask_ = subgraph(mask_nodes, edge_index, edge_score, num_nodes=num_nodes_full, return_edge_mask=True)
        edge_index_conn, edge_score_conn, edge_index_loop, edge_score_loop = segregate_self_loops(edge_index_, edge_score_)
        edge_index, edge_score = edge_index_temp, edge_score_temp

        # sort edges (in the connected graph) based on scores
        edge_score_sorted, indices = edge_score.sort(stable=True, descending=True)
        edge_index_sorted = edge_index[:,indices]

        # get each node's dominant node (in the connected graph)
        index, _ = transpose(torch.stack([torch.arange(indices.shape[0], device=device), indices], dim=0), None, indices.shape[0], indices.shape[0])
        node2dom_edge_sorted_rank = scatter(index[1], edge_index[0], reduce='min')
        node2dom_edge = edge_index_sorted[:,node2dom_edge_sorted_rank]
        node2dom_score = edge_score_sorted[node2dom_edge_sorted_rank]
        _, indices_ = node2dom_score.sort(stable=True, descending=True)
        edge_index_dom = node2dom_edge[:,indices_]

        # gen vars for parsing algo below
        node2comm_temp = -1*torch.ones(num_nodes_connected, device=device).long()
        edge_mask_inside = ~(torch.ones(edge_index.shape[1], device=device)).bool()
        batch_temp = batch[unmask_nodes]
        num_batch_temp = batch_temp.max().item()+1 if batch_temp.shape[0]!=0 else 0
        index_comm = 0
        iter = 0

        # core parsing algo, loop until there is no more dom edges
        while edge_index_dom.shape[1]!=0:
            node_batch = batch_temp[edge_index_dom[0]]
            node_set_idx = scatter(torch.arange(edge_index_dom[0].shape[0], device=device), node_batch, reduce='min') # select top rank edge for each batch
            node_set = edge_index_dom[:,node_set_idx].unique()

            while True:
                iter += 1
                edge_mask = np.isin(edge_index_dom[1], node_set)
                subset = torch.cat([edge_index_dom[0][edge_mask], node_set]).unique()

                batch_set = batch_temp[subset]
                index_values_temp = index_comm+torch.arange(batch_set.unique().shape[0], device=device)
                index_values = -1*torch.ones(num_batch_temp, device=device).long()
                index_values[batch_set.unique()] = index_values_temp

                edge_index_dom = edge_index_dom[:,~edge_mask]

                node2comm_temp[subset] = index_values[batch_set]

                if subset.shape[0]<=node_set.shape[0]:
                    edge_mask_ = np.isin(edge_index, node_set)
                    edge_mask_ = edge_mask_[0] & edge_mask_[1]

                    edge_mask_inside[edge_mask_] = True
                    index_comm = node2comm_temp.max()+1 if node2comm_temp.shape[0]!=0 else 0
                    break
                else:
                    node_set = subset

        device = device_original
        edge_index = edge_index.to(device)
        edge_score = edge_score.to(device)
        batch = batch.to(device)

        mask_nodes = mask_nodes.to(device)
        unmask_nodes = unmask_nodes.to(device)
        node2comm_temp = node2comm_temp.to(device)
        edge_mask_inside = edge_mask_inside.to(device)
        edge_index_loop = edge_index_loop.to(device)
        edge_score_loop = edge_score_loop.to(device)

        node2comm = -1*torch.ones(num_nodes_full, device=device).long()
        node2comm[unmask_nodes] = node2comm_temp
        node2comm[mask_nodes] = index_comm+torch.arange(mask_nodes.shape[0], device=device)
        num_comm = node2comm.max().item()+1 if node2comm.shape[0]!=0 else 0
        num_batch = batch.max().item()+1 if batch.shape[0]!=0 else 0
        s = SparseTensor(
                    row = torch.arange(node2comm.shape[0], device=device),
                    col = node2comm,
                    sparse_sizes = (num_nodes_full, num_comm)
                ).to_device(batch.device)

        adj_comm_temp = node2comm_temp[edge_index[:,~edge_mask_inside]]
        adj_comm, _ = coalesce(adj_comm_temp, None, m=num_comm, n=num_comm)
        adj_community = SparseTensor.from_edge_index(
                    edge_index = adj_comm,
                    sparse_sizes = (num_comm, num_comm)
                ).to_device(batch.device)

        index_, _ = coalesce(torch.stack([batch, node2comm]), None, m=num_batch, n=num_comm)
        new_batch = sort_edge_index(index_, sort_by_row=False)[0]

        if new_batch.shape[0]==batch.shape[0]:
            flag = False
        else:
            flag = True

        uniques, indexs, counts = torch.unique(batch, return_inverse=True, return_counts=True)
        mask1 = (counts==1)[indexs].nonzero().squeeze(dim=-1).tolist()

        mask2 = node2comm[mask1].tolist()

        edge2comm = node2comm_temp[edge_index[:,edge_mask_inside][0]]
        mask_connected_comm = index_to_mask(node2comm[unmask_nodes].unique(), size=new_batch.shape[0])
        node_score_temp = global_mean_pool(edge_score[edge_mask_inside], edge2comm)
        node_score = torch.ones(new_batch.shape[0], device=device)
        node_score[mask_connected_comm] = node_score_temp
        node_score[node2comm[edge_index_loop[0]]] = edge_score_loop

        uniques, indexs, link_counts_temp = torch.unique(edge2comm, return_inverse=True, return_counts=True)
        link_counts = torch.ones(new_batch.shape[0], device=device).float()
        link_counts[mask_connected_comm] = link_counts_temp.float()

        assert adj_community.is_symmetric()
        assert not torch.any(link_counts==0)

        return s, adj_community, new_batch, mask1, mask2, flag, node_score, link_counts
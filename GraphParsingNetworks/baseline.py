from torch_geometric.nn import EdgePooling, GraphMultisetTransformer, global_mean_pool
from torch.nn import Module, Linear
import torch.nn.functional as F
import torch


class EdgePool(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        if params['task'] in ["ogbg-molpcba"]:
            from layers.gcn_edge import GNN as GNN1
            self.in_encoder = GNN1(num_layer=params['layer_gnn_edge'], emb_dim=params['hidden_channel'], drop_ratio=params['dropout_network'])
        else:
            self.in_encoder = Linear(params['in_channel'], params['hidden_channel'])
        
        self.convs = torch.nn.ModuleList()
        from layers.gcn_conv import GCNConv
        for i in range(7):
            self.convs.append(GCNConv(params['hidden_channel'], params['hidden_channel']))

        self.pools = torch.nn.ModuleList()
        for i in range(3):
            self.pools.append(EdgePooling(in_channels=params['hidden_channel'], dropout=0.2))

        self.predictor = Linear(params['hidden_channel'], params['out_channel'])

    def forward(self, data):
        h = data.x
        adj_t = data.adj_t
        batch = data.batch
        torch.set_num_threads(1)

        if self.params['task'] in ["ogbg-molpcba"]:
            h = self.in_encoder(h, adj_t)
        else:
            h = self.in_encoder(h)

        row, col, val = adj_t.coo()
        edge_index = torch.stack([row, col], dim=0)

        for i in range(7):
            h = self.convs[i](h, edge_index)
            h = F.relu(h)

            if i in [0, 2, 4]:
                h, edge_index, batch, unpool_info = self.pools[i//2](h, edge_index, batch)

        if h.shape[0]!=batch.max().item()+1:
            h = global_mean_pool(h, batch)

        h = self.predictor(h)

        encode_values = dict(zip(['x', 'num_pooling_layers'], [h, 0]))

        return encode_values


class GMT(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        if params['task'] in ["ogbg-molpcba"]:
            from layers.gcn_edge import GNN as GNN1
            self.in_encoder = GNN1(num_layer=params['layer_gnn_edge'], emb_dim=params['hidden_channel'], drop_ratio=params['dropout_network'])
        else:
            self.in_encoder = Linear(params['in_channel'], params['hidden_channel'])

        from layers.gcn_conv import GCNConv as Conv
        self.pool = GraphMultisetTransformer(in_channels=params['hidden_channel'], hidden_channels=params['hidden_channel'], out_channels=params['out_channel'], Conv=Conv)

        self.predictor = Linear(params['hidden_channel'], params['out_channel'])

    def forward(self, data):
        h = data.x
        adj_t = data.adj_t
        batch = data.batch
        torch.set_num_threads(1)

        if self.params['task'] in ["ogbg-molpcba"]:
            h = self.in_encoder(h, adj_t)
        else:
            h = self.in_encoder(h)

        row, col, val = adj_t.coo()
        edge_index = torch.stack([row, col], dim=0)

        h = self.pool(h, batch, edge_index)

        encode_values = dict(zip(['x', 'num_pooling_layers'], [h, 0]))

        return encode_values
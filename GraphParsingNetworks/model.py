import torch
from torch.nn import Module
from torch_sparse import SparseTensor
from layer import GNN, MLP
from pool import global_mean_pool


class GPNN(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        from layer import ParsingNet_GPU as ParsingNet

        if 'gnn_model' in params:
            if params['gnn_model']=='GIN':
                gnn_model='GIN'
            elif params['gnn_model']=='GAT':
                gnn_model='GAT'
        else:
            gnn_model='GCN'

        if params['task'] in ["ogbg-molhiv", "ogbg-molpcba"]:
            from layers.gcn_edge import GNN as GNN_edge
            self.gnn_edge = GNN_edge(num_layer=params['layer_gnn_edge'], emb_dim=params['hidden_channel'], drop_ratio=params['dropout_network'])

        self.input_trans = MLP(in_channel=params['in_channel'], hidden_channel=params['hidden_channel'], out_channel=params['hidden_channel'], num_layers=params['layer_trans'], dropout=params['dropout_network'], norm_mode='insert', act_final=params['act_final'])

        self.gnn1 = GNN(hidden_channel=params['hidden_channel'], num_layers=params['layer_gnn1'], dropout=params['dropout_network'], gnn_model=gnn_model, act_final=params['act_final'])

        self.parsing_net = ParsingNet(channel=params['hidden_channel'], dropout_network=params['dropout_network'], dropout_parsing=params['dropout_parsing'], layer_parsingnet=params['layer_parsingnet'], link_ignore_self_loop=params['link_ignore_self_loop'])

        if params['layer_gnn2']=="share":
            self.gnn2 = self.gnn1
            layer_gnn2 = params['layer_gnn1']
        else:
            if params['layer_gnn2']=="follow":
                layer_gnn2 = params['layer_gnn1']
            else:
                layer_gnn2 = params['layer_gnn2']
            self.gnn2 = GNN(hidden_channel=params['hidden_channel'], num_layers=layer_gnn2, dropout=params['dropout_network'], gnn_model=gnn_model, act_final=params['act_final'])

        if params['task_type']=="task_node":
            if params['layer_gnn3']=="share":
                self.gnn3 = self.gnn2
            else:
                if params['layer_gnn3']=="follow":
                    layer_gnn3 = layer_gnn2
                else:
                    layer_gnn3 = params['layer_gnn3']
                self.gnn3 = GNN(hidden_channel=params['hidden_channel'], num_layers=layer_gnn3, dropout=params['dropout_network'], gnn_model=gnn_model, act_final=params['act_final'])

        self.deepsets_pre = MLP(in_channel=params['hidden_channel'], hidden_channel=params['hidden_channel'], out_channel=params['hidden_channel'], num_layers=params['layer_deepsets'], dropout=params['dropout_network'], norm_mode='post', act_final=params['act_final'])
        self.deepsets_post = MLP(in_channel=params['hidden_channel'], hidden_channel=params['hidden_channel'], out_channel=params['hidden_channel'], num_layers=params['layer_deepsets'], dropout=params['dropout_network'], norm_mode='post', act_final=params['act_final'])

        self.predictor = MLP(in_channel=params['hidden_channel'], hidden_channel=params['hidden_channel'], out_channel=params['out_channel'], num_layers=params['layer_trans'], dropout=params['dropout_network'], norm_mode='insert', act_final=False)

    # @profile
    def forward(self, data):
        h = data.x # shape: [num_nodes, num_feature]
        adj_t = data.adj_t # shape: [num_nodes, num_nodes]

        if self.params['task_type']=='task_graph':
            batch = data.batch # shape: [num_nodes]
            batch_size = torch.max(batch).item()+1
        elif self.params['task_type']=='task_node':
            batch_size = h.shape[0]
            batch = h.new_zeros(batch_size, dtype=torch.int64) # shape: [num_nodes]
        assignments = []

        if self.params['task'] in ["ogbg-molhiv", "ogbg-molpcba"]:
            hs = []
            h = self.gnn_edge(h, adj_t)
            row, col, val = adj_t.coo()
            adj_t = SparseTensor(
                row = row,
                col = col,
                sparse_sizes=(h.shape[0], h.shape[0])
            ).to_device(h.device)
        else:
            h = self.input_trans(h) # shape: [num_nodes, num_hidden]

        assert adj_t.is_symmetric()

        # pooling process
        flag = True
        while flag:
            if self.params['task'] in ["ogbg-molhiv", "ogbg-molpcba"]:
                hs.append(global_mean_pool(h, batch))
            h_init = h
            adj_t_init = adj_t

            # apply GNN to extract neighbor feature
            h_gnn1 = self.gnn1(h_init, adj_t_init)

            # apply neural parsing to compute link score, parsing link score, and construct node assignments and community scores
            s, adj_t, batch, mask1, mask2, flag, node_score, link_counts = self.parsing_net(h_gnn1, adj_t, batch)

            # apply GNN to extract neighbor feature, however from the parsed graph
            if self.params['layer_gnn2']=="share":
                h_gnn2 = h_gnn1
            else:
                h_gnn2 = self.gnn2(h_init, adj_t_init)

            if flag==False and self.params['task_type']=='task_graph':
                s = SparseTensor(
                    row = torch.tensor(list(range(batch.shape[0]))),
                    col = batch.cpu(),
                    sparse_sizes=(batch.shape[0], batch_size)
                ).to_device(batch.device)
                adj_t = None
                batch = torch.tensor(list(range(batch_size)), device=batch.device)

            if self.params['task_type']=='task_node':
                batch_size = batch.shape[0]

            if flag==True or batch.shape[0]!=batch_size:
                # neural pooling
                h = self.deepsets_pre(h_gnn2)
                h = s.t() @ h # shape: [num_communities, num_hidden]
                h = self.deepsets_post(h)

                # make gradients flow to the neural parser
                h = h * node_score.view(-1,1)
                h = h * link_counts.view(-1,1)

                if self.params['task_type']=='task_node':
                    assignments.append({'s': s, 'adj_t': adj_t_init, 'h':h_init, 'link_counts':link_counts})
                elif self.params['task_type']=='task_graph':
                    # for batch that converged to one node, preserve their initial feature
                    h[mask2,:] = h_init[mask1,:]
                    assignments.append(0)

            elif self.params['task_type']=='task_graph':
                h = s.t() @ h_init # shape: [num_communities, num_hidden]

        # un-pooling for node task
        if self.params['task_type']=='task_node':
            assignments.reverse()

            for i in range(len(assignments)):
                s, adj_t, h_temp, link_counts = assignments[i].values()
                link_counts = 1/link_counts

                h = h * link_counts.view(-1,1)
                h = s @ h
                h = h + h_temp
                h = self.gnn3(h, adj_t)

        if self.params['task'] in ["ogbg-molhiv", "ogbg-molpcba"]:
            h = torch.sum(torch.stack(hs), dim=0)
        
        h = self.predictor(h) # shape: [num_batch, num_class]
        encode_values = dict(zip(['x', 'num_pooling_layers'], [h, len(assignments)]))

        return encode_values
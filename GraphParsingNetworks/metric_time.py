import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from layer import MLP, GNN
from run import setup_seed
import line_profiler as lp
import time
import json

params = {
    'layer_gnn1': 2,
    'layer_parsingnet': 2,
    'layer_deepsets': 2,

    'hidden_channel': 128,
    'dropout_network': 0,
    'dropout_parsing': 0,
    'link_ignore_self_loop': True,
    'act_final': False,

    # 'num_nodes': 1100,
    'edge_prob': 0.2,
    'batch_size': 50,

    'gpu_index': 7,
    'type_parsing': 'GPU'
}

dict_time_avg = {}
for num in [1100,1000,900,800,700,600,500,400,300,200,100]:
    print('----------------------------------------------------')
    print('Num of nodes: ', num)
    params['num_nodes'] = num

    setup_seed(0)

    device = torch.device('cuda:%s'%(params['gpu_index']) if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    # syntactic data generation
    data_list = []
    for i in range(params['batch_size']):
        x = torch.rand(params['num_nodes'], params['hidden_channel'], device=device)
        edge_index = erdos_renyi_graph(num_nodes=params['num_nodes'], edge_prob=params['edge_prob'], directed=False).to(device)
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
    loader = DataLoader(data_list, batch_size=params['batch_size'])
    data = [i for i in loader][0]
    edge_index = SparseTensor.from_edge_index(data.edge_index)
    x = data.x
    batch = data.batch

    if params['type_parsing']=='GPU':
        from layer import ParsingNet_GPU as ParsingNet
    elif params['type_parsing']=='CPU':
        from layer import ParsingNet_CPU as ParsingNet

    gnn1 = GNN(hidden_channel=params['hidden_channel'], num_layers=params['layer_gnn1'], dropout=params['dropout_network'], gnn_model='GCN', act_final=params['act_final']).to(device)
    gnn1.eval()

    parsing_net = ParsingNet(channel=params['hidden_channel'], dropout_network=params['dropout_network'], dropout_parsing=params['dropout_parsing'], layer_parsingnet=params['layer_parsingnet'], link_ignore_self_loop=params['link_ignore_self_loop']).to(device)
    parsing_net.eval()

    deepsets_pre = MLP(in_channel=params['hidden_channel'], hidden_channel=params['hidden_channel'], out_channel=params['hidden_channel'], num_layers=params['layer_deepsets'], dropout=params['dropout_network'], norm_mode='post', act_final=params['act_final']).to(device)
    deepsets_pre.eval()

    deepsets_post = MLP(in_channel=params['hidden_channel'], hidden_channel=params['hidden_channel'], out_channel=params['hidden_channel'], num_layers=params['layer_deepsets'], dropout=params['dropout_network'], norm_mode='post', act_final=params['act_final']).to(device)
    deepsets_post.eval()

    def once_forward(gnn1, parsing_net, deepsets_pre, deepsets_post, x, edge_index, batch):
        # apply GNN to extract neighbor feature
        h_gnn1 = gnn1(x, edge_index)

        # apply neural parsing to compute link score, parsing link score, and construct node assignments and community scores
        s, adj_t, batch, mask1, mask2, flag, node_score, link_counts = parsing_net(h_gnn1, edge_index, batch)

        # neural pooling
        h = deepsets_pre(h_gnn1)
        h = s.t() @ h # shape: [num_communities, num_hidden]
        h = deepsets_post(h)

        return s, adj_t, batch, mask1, mask2, flag, node_score, link_counts

    time_list = []
    for i in range(10):
        tik = time.time()
        s, adj_t, batch_new, mask1, mask2, flag, node_score, link_counts = once_forward(gnn1, parsing_net, deepsets_pre, deepsets_post, x, edge_index, batch)
        tok = time.time()
        time_once = tok-tik
        time_list.append(time_once)
        print('Time once: ', time_once)

    print('--------------------------')
    print('Time overall: ', sum(time_list))
    print('Time avg: ', sum(time_list)/len(time_list))
    dict_time_avg[num] = sum(time_list)/len(time_list)

json.dump(dict_time_avg, open('remote/time.json', 'w'))
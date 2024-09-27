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
    # 'edge_prob': 0.2,
    'batch_size': 1,

    'gpu_index': 7,
    'type_parsing': 'GPU'
}

device = torch.device('cuda:%s'%(params['gpu_index']) if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)

memo_res_start = torch.cuda.mem_get_info(device)[0]/(1024*1024)
memo_res = memo_res_start

dict_memo_avg = {}
for num in [1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]:
    print('----------------------------------------------------')
    print('Num of nodes: ', num)
    params['num_nodes'] = num
    params['edge_prob'] = 2*(2*num)/(num*num)

    setup_seed(0)

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

    memo_res = min(memo_res, torch.cuda.mem_get_info(device)[0]/(1024*1024))

    def once_forward(gnn1, parsing_net, deepsets_pre, deepsets_post, x, edge_index, batch, device):
        memo_res_start = torch.cuda.mem_get_info(device)[0]/(1024*1024)
        memo_res = memo_res_start

        # apply GNN to extract neighbor feature
        h_gnn1 = gnn1(x, edge_index)
        memo_res = min(memo_res, torch.cuda.mem_get_info(device)[0]/(1024*1024))

        # apply neural parsing to compute link score, parsing link score, and construct node assignments and community scores
        s, adj_t, batch, mask1, mask2, flag, node_score, link_counts = parsing_net(h_gnn1, edge_index, batch)
        memo_res = min(memo_res, torch.cuda.mem_get_info(device)[0]/(1024*1024))

        # neural pooling
        h = deepsets_pre(h_gnn1)
        h = s.t() @ h # shape: [num_communities, num_hidden]
        h = deepsets_post(h)
        memo_res = min(memo_res, torch.cuda.mem_get_info(device)[0]/(1024*1024))

        return s, adj_t, batch, mask1, mask2, flag, node_score, link_counts, memo_res

    memo_list = []
    s, adj_t, batch_new, mask1, mask2, flag, node_score, link_counts, memo_res = once_forward(gnn1, parsing_net, deepsets_pre, deepsets_post, x, edge_index, batch, device)
    memo_once = memo_res_start - memo_res
    memo_list.append(memo_once)
    print('Memo once: ', memo_once)
    # torch.cuda.empty_cache()

    print('--------------------------')
    print('Memo avg: ', sum(memo_list)/len(memo_list))
    dict_memo_avg[num] = sum(memo_list)/len(memo_list)

json.dump(dict_memo_avg, open('remote/memo.json', 'w'))
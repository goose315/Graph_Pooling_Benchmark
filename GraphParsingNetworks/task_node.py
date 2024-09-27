import torch
from transformers.optimization import get_cosine_schedule_with_warmup
import os
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB
import torch.nn.functional as F

def get_trainer(params):
    # get datasets
    dataset_name = params['task']
    split = params['index_split']

    if dataset_name in ['wisconsin', 'PubMed', 'cornell']:
        dataset = WebKB(root='/data1/Pooling', name='%s'%(dataset_name.split('_')[0]), split='public', transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor()]))
    elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='/data1/Pooling', name='%s'%(dataset_name.split('_')[0]), split='public', transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor()]))

    data = dataset[0]

    if dataset_name in ['wisconsin', 'PubMed', 'cornell']:
        split_str = "%s_split_0.6_0.2_%s.npz"%(dataset_name.split('_')[0].lower(), str(split))
        split_file = np.load(os.path.join('datasets/geomgcn/', split_str))
        data.train_mask = torch.Tensor(split_file['train_mask'])==1
        data.val_mask = torch.Tensor(split_file['val_mask'])==1
        data.test_mask = torch.Tensor(split_file['test_mask'])==1
    elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        split_str = "%s_split_0.6_0.2_%s.npz"%(dataset_name.split('_')[0].lower(), str(split))
        split_file = np.load(os.path.join('datasets/geomgcn/', split_str))
        data.train_mask = torch.Tensor(split_file['train_mask'])==1
        data.val_mask = torch.Tensor(split_file['val_mask'])==1
        data.test_mask = torch.Tensor(split_file['test_mask'])==1

    params['in_channel']=data.num_features
    params['out_channel']=dataset.num_classes

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get model
    if params['model'] in ['GPNN']:
        from model import GPNN as Encoder
        model = Encoder(params)

    # get criterion
    criterion = torch.nn.NLLLoss()
    
    # get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # get scheduler
    if params['lr_scheduler']==True:
        scheduler = get_cosine_schedule_with_warmup(optimizer, params['patience'], params['epochs'])
    else:
        scheduler = None

    # get trainer
    if params['task'] in ['wisconsin', 'PubMed', 'cornell', 'Cora', 'CiteSeer', 'PubMed']:
        trainer = dict(zip(['data', 'model', 'criterion', 'optimizer', 'scheduler', 'params'], [data, model, criterion, optimizer, scheduler, params]))

    return trainer

def get_metric(trainer, stage):
    # load variables
    if trainer['params']['task'] in ['wisconsin', 'PubMed', 'cornell', 'Cora', 'CiteSeer', 'PubMed']:
        data, device, model, criterion, optimizer, scheduler, params = trainer.values()

    # set train/evaluate mode and device for model
    model = model.to(device)
    if stage=='train':
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()

    # training/evaluating
    data = data.to(device)
    encode_values = model(data)
    h = encode_values['x']
    out = F.log_softmax(h, dim=-1)

    stage = 'val' if stage=='valid' else stage

    if stage=='train':
        for _, mask_tensor in data(stage+'_mask'):
            mask = mask_tensor

        loss = criterion(out[mask], data.y[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_norm'])
        optimizer.step()
        optimizer.zero_grad()
        if params['lr_scheduler']==True:
            scheduler.step()

        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        metrics = dict(zip(['metric', 'loss', 'encode_values'], [acc, loss.item(), encode_values]))

    elif stage=='valid_test':
        metrics = {}
        for stage_temp in ['val', 'test']:
            for _, mask_tensor in data(stage_temp+'_mask'):
                mask = mask_tensor

            loss = criterion(out[mask], data.y[mask])

            acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
            metrics_temp = dict(zip(['metric', 'loss', 'encode_values'], [acc, loss.item(), encode_values]))

            metrics[stage_temp] = metrics_temp

    return metrics
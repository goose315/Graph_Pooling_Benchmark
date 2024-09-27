import sys
import torch
from transformers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torch_geometric.transforms as T
from datasets.asap.utils.data_asap import num_graphs

def get_trainer(params):
    evaluator = None

    # get datasets/dataloaders
    if params['splits']=="GMT":
        sys.path.append('datasets/gmt/')
        from datasets.gmt.dataloader import get_dataloaders
        if params['task'] in ['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'COLLAB', 'ENZYMES']:
            train_loader, val_loader, test_loader, num_features, num_classes = get_dataloaders(task=params['task'], fold_number=params['index_split'], batch_size=params['batch_size'])
            params['in_channel'] = num_features
            params['out_channel'] = num_classes
    elif params['splits']=="ASAP":
        sys.path.append('datasets/asap/')
        from datasets.asap.dataloader_asap import get_dataloaders
        if params['task'] in ["FRANKENSTEIN", "NCI109", "NCI1", "PROTEINS", "DD"]:
            train_loader, val_loader, test_loader, num_features, num_classes = get_dataloaders(task=params['task'], seed=params['seed'], fold_number=params['index_split'], batch_size=params['batch_size'])
            params['in_channel'] = num_features
            params['out_channel'] = num_classes
    elif params['splits']=="OGB":
        from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
        from torch_geometric.loader import DataLoader
        dataset = PygGraphPropPredDataset(root='datasets/pyg/', name=params['task'], transform=T.ToSparseTensor(attr='edge_attr'))
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(params['task'])
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=params['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=params['batch_size'], shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=params['batch_size'], shuffle=False, num_workers=0)
        params['eval_metric'] = dataset.eval_metric
        params['in_channel'] = params['hidden_channel']
        params['out_channel'] = dataset.num_tasks

    # get device
    device = torch.device('cuda:%s'%(params['gpu_index']) if torch.cuda.is_available() else 'cpu')
    print("GPU device: [%s]"%(device))

    # get model
    if params['model'] in ['GPNN']:
        from model import GPNN as Encoder
    elif params['model'] in ['EdgePool']:
        from baseline import EdgePool as Encoder
    elif params['model'] in ['GMT']:
        from baseline import GMT as Encoder
    model = Encoder(params)

    # get criterion and optimizer
    if params['task'] in ["ogbg-molhiv", "ogbg-molpcba"]:
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    else:
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # get scheduler
    if params['lr_scheduler']==True:
        scheduler = get_cosine_schedule_with_warmup(optimizer, params['patience']*len(train_loader), params['epochs']*len(train_loader))
    else:
        scheduler = None

    # get trainer
    trainer = dict(zip(['train_loader', 'val_loader', 'test_loader', 'device', 'model', 'criterion', 'optimizer', 'scheduler', 'evaluator', 'params'], [train_loader, val_loader, test_loader, device, model, criterion, optimizer, scheduler, evaluator, params]))

    return trainer

def get_metric(trainer, stage):
    # load variables
    train_loader, val_loader, test_loader, device, model, criterion, optimizer, scheduler, evaluator, params = trainer.values()
    if stage=='train':
        data_loader = train_loader
    elif stage=='valid':
        data_loader = val_loader
    elif stage=='test':
        data_loader = test_loader

    # set train/evaluate mode and device for model
    model = model.to(device)
    if stage=='train':
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()

    if params['task'] in ["ogbg-molhiv", "ogbg-molpcba"]:
        y_true = []
        y_pred = []
        total_loss = 0.
        num_g_total = 0.

        for batch in data_loader:
            if stage=='train':
                if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                    continue
            else:
                if batch.x.shape[0] == 1:
                    continue

            num_g = num_graphs(batch)
            num_g_total += num_g
            batch = batch.to(device)
            encode_values = model(batch)
            pred = encode_values['x']

            is_labeled = batch.y == batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            total_loss += loss.item()*num_g

            if stage=='train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if params['lr_scheduler']==True:
                    scheduler.step()

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        total_loss = total_loss/num_g_total

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        metric = evaluator.eval(input_dict)[params['eval_metric']]

    else:
        total_loss = 0.
        correct = 0.
        num_g_total = 0

        for data in data_loader:
            num_g = num_graphs(data)
            num_g_total += num_g
            data = data.to(device)
            encode_values = model(data)
            h = encode_values['x']

            out = F.log_softmax(h, dim=-1)
            loss = criterion(out, data.y)
            total_loss += loss.item()*num_g

            if stage=='train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                if params['lr_scheduler']==True:
                    scheduler.step()

            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()

        total_loss = total_loss/num_g_total
        metric = correct/num_g_total

    metrics = dict(zip(['metric', 'loss'], [metric, total_loss]))

    return metrics
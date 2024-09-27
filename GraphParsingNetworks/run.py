from torch.utils.tensorboard import SummaryWriter
import torch
import importlib
import os
import random
import numpy as np
import wandb
import json
import hashlib
import tempfile
import shutil
import sys
import traceback
import time

# function params: wandb_base, sweep_id, gpu_index, code_fullname, save_model
# params from wandb: task_type, seed, epochs, log_freq, stop_item, patience, task, index_split, batch_size, model, learning_rate, weight_decay, grad_norm, hidden_channel, heads, dropout_rate
# generated wandb params: params_hash, gpu_index
# generated temporal params: in_channel, out_channel

# generate hash tag for one set of hyper parameters
def get_hash(dict_in, hash_keys, ignore_keys):
    dict_in = {k:v for k,v in dict_in.items() if k in hash_keys}
    dict_in = {k:v for k,v in dict_in.items() if k not in ignore_keys}
    hash_out = hashlib.blake2b(json.dumps(dict_in, sort_keys=True).encode(), digest_size=4).hexdigest()
    return str(hash_out)

# fix random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# running and evaluating, return ID of this run
def runner(wandb_base, sweep_id, gpu_index, code_fullname, save_model):
    if wandb_base=='temp':
        folder_temp = tempfile.TemporaryDirectory()
        tmpdirname = folder_temp.name
        os.chmod(tmpdirname, 0o777)
        dir_name = tmpdirname
    elif wandb_base=='remote':
        dir_name = 'remote'
        
    wandb.init(dir=dir_name, reinit=True, group=sweep_id)

    try:
        wandb.use_artifact(code_fullname, type='code')

        params_hash = get_hash(wandb.config, wandb.config['hash_keys'], wandb.config['ignore_keys'])
        wandb.config.update({'params_hash':params_hash}, allow_val_change=True)
        wandb.config.update({'gpu_index':gpu_index}, allow_val_change=True)

        params = dict(wandb.config)
        print("This trial's parameters: %s"%(params))

        if save_model==True:
            checkpoint_path = os.path.join(wandb.run.dir, 'checkpoint')
            os.makedirs(checkpoint_path)
        tensorboard_path = os.path.join(wandb.run.dir, 'tensorboard')
        os.mkdir(tensorboard_path)

        get_trainer = importlib.import_module(params['task_type']).get_trainer
        get_metric = importlib.import_module(params['task_type']).get_metric

        seed = params['seed']
        if seed!="None":
            setup_seed(seed)

        trainer = get_trainer(params)
        writer = SummaryWriter(log_dir=tensorboard_path)

        bad_cnt = 0
        best_test_metric = 0
        best_val_metric = 0
        best_val_loss = 1e10

        time_all = []

        for epoch in range(params['epochs']):
            start_time = time.time()
            metrics = get_metric(trainer=trainer, stage='train')
            end_time = time.time()
            time_consumed_train = end_time-start_time
            train_metric, train_loss = metrics['metric'], metrics['loss']

            if params['task_type']=='task_graph':
                start_time = time.time()
                metrics = get_metric(trainer=trainer, stage='valid')
                end_time = time.time()
                time_consumed_val = end_time-start_time
                val_metric, val_loss = metrics['metric'], metrics['loss']

                start_time = time.time()
                metrics = get_metric(trainer=trainer, stage='test')
                end_time = time.time()
                time_consumed_test = end_time-start_time
                test_metric, test_loss = metrics['metric'], metrics['loss']

            elif params['task_type']=='task_node':
                if epoch%10==0:
                    start_time = time.time()
                    metrics = get_metric(trainer=trainer, stage='valid_test')
                    end_time = time.time()

                    time_consumed_val = end_time-start_time
                    time_consumed_test = 0

                    val_metric, val_loss = metrics['val']['metric'], metrics['val']['loss']
                    test_metric, test_loss = metrics['test']['metric'], metrics['test']['loss']

            time_all.append(time_consumed_val+time_consumed_test+time_consumed_train)

            if epoch%params['log_freq']==0:
                wandb.log({'metric/train':train_metric, 'metric/val':val_metric, 'metric/test':test_metric, 'loss/train':train_loss, 'loss/val':val_loss, 'loss/test':test_loss, 'metric/best':best_test_metric, 'time/train':float(time_consumed_train), 'time/val':float(time_consumed_val), 'time/test':float(time_consumed_test)})

            if params['stop_item']=='metric_val':
                if val_metric>best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric
                    bad_cnt = 0
                    if save_model==True:
                        torch.save(trainer['model'].state_dict(), os.path.join(checkpoint_path,'model.pt'))
                        json.dump(dict(params), open(os.path.join(checkpoint_path,'model_config.json'), 'w'))
                else:
                    bad_cnt += 1
            elif params['stop_item']=='loss_val':
                if val_loss<best_val_loss:
                    best_val_loss = val_loss
                    best_test_metric = test_metric
                    bad_cnt = 0
                    if save_model==True:
                        torch.save(trainer['model'].state_dict(), os.path.join(checkpoint_path,'model.pt'))
                        json.dump(dict(params), open(os.path.join(checkpoint_path,'model_config.json'), 'w'))
                else:
                    bad_cnt += 1
            if params['task_type']=='task_graph' and bad_cnt==params['patience']:
                break
            elif params['task_type']=='task_node' and bad_cnt==params['patience']/10:
                break
        
        print('Final metric is [%s]'%(best_test_metric))
        writer.close()
        wandb.run.summary["metric/final"] = best_test_metric
        wandb.run.summary["time/avg"] = sum(time_all)/len(time_all) if len(time_all)!=0 else 0
        wandb.run.summary["time/total"] = sum(time_all) if len(time_all)!=0 else 0
    
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)

    wandb.finish()
    if wandb_base=='temp':
        shutil.rmtree(tmpdirname)

    return str(wandb.run.id)
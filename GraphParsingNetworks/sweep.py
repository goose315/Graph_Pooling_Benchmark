import time
import wandb
import yaml
import argparse
import time
import shutil
import os

def GetRunTime(func):
    def call_func(*args, **kwargs):
        begin_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        Run_time = end_time - begin_time
        print("Execution time for func [%s] is [%s]"%(str(func.__name__), str(Run_time)))
        return ret
    return call_func

def get_timestamp():
    time.tzset()
    now = int(round(time.time()*1000))
    timestamp = time.strftime('%Y-%m%d-%H%M',time.localtime(now/1000))
    return timestamp

# calculate the size of a sweep's search space or the number of runs
def count_sweep(mode, entity, project, id):
    # mode: size_space, num_runs
    api = wandb.Api()
    sweep = api.sweep('%s/%s/%s'%(entity, project, id))
    if mode=='size_space':
        cnt = 1
        params= sweep.config['parameters']
        for key in params.keys():
            cnt *= len(params[key]['values'])
    elif mode=='num_runs':
        cnt = len(sweep.runs)
    return cnt

# get configs for a sweep from .yaml file
def get_configs_from_file(path_yaml):
    dict_yaml = yaml.load(open(path_yaml).read(), Loader=yaml.Loader)
    sweep_config = dict_yaml['sweep_config']
    params_config = dict_yaml['params_config']
    search_space = {}
    hash_keys = []
    for k,v in params_config.items():
        search_space[k] = {"values":v}
        if len(v)>1:
            hash_keys.append(k)
    search_space['hash_keys'] = {"values":[hash_keys]}
    sweep_config['parameters'] = search_space
    return sweep_config

# get configs for a sweep from a wandb run and it's sweep
def get_configs_from_run(entity, project, run_id):
    api = wandb.Api()
    run = api.run('%s/%s/%s'%(entity, project, run_id))
    sweep_config = dict(run.sweep.config)
    for key in dict(run.config):
        if key in sweep_config['parameters'].keys():
            sweep_config['parameters'][key] = {'values':[run.config[key]]}
    return sweep_config

# modify some specific hyper parameters in sweep's config
def modify_sweep(sweep_config, dict_new):
    for key in dict_new.keys():
        sweep_config['parameters'][key] = {'values':dict_new[key]}
    return sweep_config

if __name__ == '__main__':
    # parse parameters and generate sweep configs
    parser = argparse.ArgumentParser(description='Generating sweep from yaml file or previous runs.')
    parser.add_argument('--entity', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--source', type=str)
    parser.add_argument('--info', type=str)
    parser.add_argument('--modify', type=str)
    args = parser.parse_args()
    print(args)
    if args.source=='file':
        sweep_config = get_configs_from_file(path_yaml=args.info)
    elif args.source=='run':
        sweep_config = get_configs_from_run(entity=args.entity, project=args.project, run_id=args.info)
    if args.modify:
        sweep_config = modify_sweep(sweep_config=sweep_config, dict_new=eval(args.modify))

    # generate sweep
    sweep_id = wandb.sweep(sweep_config, entity=args.entity, project=args.project)
    time.sleep(3)
    size_sweep = count_sweep(mode='size_space', entity=args.entity, project=args.project, id=sweep_id)

    # backup yaml file
    if args.source=='file':
        timestamp = get_timestamp()
        backup_filename = '%s_%s.yaml'%(timestamp, sweep_id)
        shutil.copy(args.info, 'configs/%s'%(backup_filename))

    print('Create new sweep [%s].'%(sweep_id))
    print('Sweep search space size: [%s]'%(size_sweep))
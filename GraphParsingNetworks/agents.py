import os
import wandb
from multiprocessing import Process, Queue
import time
from run import runner
from sweep import count_sweep, get_timestamp
import argparse
import glob
import tempfile

def agent(entity, project, q, wandb_base, sweep_id, gpu_index, agent_package, code_fullname, save_model):
    print('Agent started with GPU [%s].'%(gpu_index))
    try:
        wandb.agent(sweep_id, function=lambda:runner(wandb_base, sweep_id, gpu_index, code_fullname, save_model), entity=entity, project=project, count=agent_package)
    except:
        time.sleep(180)
        print('Some error with this agent, skip.')
    q.put(gpu_index)
    print('Agent finished and release GPU [%s].'%(gpu_index))

if __name__ == '__main__':
    # parse parameters
    parser = argparse.ArgumentParser(description='Generating and run agents for a sweep.')
    parser.add_argument('--entity', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--gpu_allocate', type=str)
    parser.add_argument('--wandb_base', type=str) # 'temp', 'remote'
    parser.add_argument('--mode', type=str) # 'parallel', 'one-by-one'
    parser.add_argument('--save_model', type=str) # 'True', 'False'
    args = parser.parse_args()
    print(args)
    if args.save_model=='True':
        save_model = True
    elif args.save_model=='False':
        save_model = False
    list_gpu_id = sum([[int(item.split(':')[0]) for i in range(int(item.split(':')[1]))] for item in args.gpu_allocate.split('-')], [])
    print('GPU allocate: [%s]'%(list_gpu_id))

    # backup current source code
    folder_temp = tempfile.TemporaryDirectory()
    tmpdirname = folder_temp.name
    os.chmod(tmpdirname, 0o777)
    timestamp = get_timestamp()
    wandb.init(dir=tmpdirname, entity=args.entity, project=args.project, name='%s_backup'%(timestamp), notes='Backup source code.')
    artifact = wandb.Artifact('source_code', type='code')
    for filename in glob.glob('**/*.py', recursive=True):
        if 'test' not in filename:
            if '/' in filename:
                if filename.split('/')[0]=='datasets':
                    artifact.add_file(filename)
            else:
                artifact.add_file(filename)
    wandb.log_artifact(artifact)
    wandb.finish()
    
    time.sleep(3)
    api = wandb.Api()
    artifact = api.artifact('%s/%s/source_code:latest'%(args.entity, args.project))
    code_fullname = '%s/%s/source_code:%s'%(args.entity, args.project, artifact.version)

    # get agent_package from sweep
    api = wandb.Api()
    sweep = api.sweep('%s/%s/%s'%(args.entity, args.project, args.sweep_id))
    # agent_package = sweep.config['parameters']['agent_package']['values'][0]
    agent_package = 1000
        
    # GPU and process manage
    q = Queue()
    for i in list_gpu_id:
        q.put(i)
    if args.mode=='parallel':
        print('Running in parallel mode.')
        os.environ["WANDB_START_METHOD"] = "thread"
    elif args.mode=='one-by-one':
        print('Running in one-by-one mode.')

    num_space = count_sweep(mode='size_space', entity=args.entity, project=args.project, id=args.sweep_id)
    while True:
        num_now = count_sweep(mode='num_runs', entity=args.entity, project=args.project, id=args.sweep_id)
        if num_now<num_space:
            gpu_index = q.get()
            if args.mode=='parallel':
                p = Process(target=agent, args=(args.entity, args.project, q, args.wandb_base, args.sweep_id, gpu_index, agent_package, code_fullname, save_model, ))
                p.start()
            elif args.mode=='one-by-one':
                agent(args.entity, args.project, q, args.wandb_base, args.sweep_id, gpu_index, agent_package, code_fullname, save_model)
        else:
            break
    
    print('All agents finished.')
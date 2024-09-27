# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:18:44 2022

@author: Fanding Xu
"""

import argparse
import os
import numpy as np
import torch
from rdkit import RDLogger  
from databuild import TaskConfig, GenerateDataset, MolDatasetInMemory, RandomSplit, set_seed, RandomScaffoldSplit, LabelNormalizer
from process import Trainer
from models.MUSE_model import MUSEPred, EGINPred
from models.baseline import AsymCheegerCut, Diff, MinCut, DMoN, Hosc, just_balance
from models.baseline import TopK, SAG, ASAP, PAN, CO, CGI, KMIS, GSA, HGPSL
model_dict = {
    'MinCut': MinCut,
    'TopK': TopK,
    'SAG': SAG,
    'ASAP': ASAP,
    'PAN': PAN,
    'CO': CO,
    'CGI': CGI,
    'KMIS': KMIS,
    'GSA': GSA,
    'HGPSL': HGPSL,
    'AsymCheegerCut': AsymCheegerCut,
    'Diff': Diff,
    'DMoN': DMoN,
    'Hosc': Hosc,
    'just_balance': just_balance
}
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="MESPool test", add_help=True)
    parser.add_argument('--dataset', type=str, default='bace',
                        help='which benchmark task to run (default: bace)')
    parser.add_argument('--cuda_num', type=int, default=0,
                        help='which gpu to use if any (-1 for cpu, default: 0)')
    parser.add_argument('--run_times', type=int, default=1,
                        help='how many times to run independently (default: 1)')
    parser.add_argument('--k', type=int, default=1,
                        help='k times independent data splitting (default: 1)')
    parser.add_argument('--split_seed', type=int, default=1234,
                        help='dataset split random seed (default: 1234)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Pytorch DataLoader num_workers (default: 0)')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='learning rate reduce rate (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', default=3, type=int,
                        help='learning rate reduce patience (default: 3)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--patience', type=int, default=15,
                        help='early stop patience (default: 15)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum training epochs (default: 500)')
    parser.add_argument('--min_epoch', type=int, default=1,
                        help='the model must train at least min_epoch times (default: 1)')
    parser.add_argument('--load_pretrain', action="store_true",
                        help='whether to load pretrained model (default: False)')
    parser.add_argument('--log_train_results', action="store_true",
                        help='whether to evaluate training set in each epoch, costs more time (default: False)')
    
    parser.add_argument('--lin_before_conv', action="store_true",
                        help='whether to set a linear layer before convolution layers(default: False)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--hidden_channels_x', type=int, default=64,
                        help='model hidden channels for node embeddings (default: 128)')
    parser.add_argument('--hidden_channels_e', type=int, default=128,
                        help='model hidden channels for edge embeddings (default: 128)')
    parser.add_argument('--threshold', type=int, default=0.5,
                        help='threshold (default: 0.5)')
    parser.add_argument('--topk_ratio', type=float, default=0.5, 
                    help='Ratio for TopK pooling (default: 0.5)')
    parser.add_argument('--sag_ratio', type=float, default=0.5,
                        help='ratio for SAGPooling (default: 0.5)')
    parser.add_argument('--asa_ratio', type=float, default=0.5,
                        help='ratio for ASAPooling (default: 0.5)')
    parser.add_argument('--pan_ratio', type=float, default=0.5,
                        help='ratio for PANPooling (default: 0.5)')
    parser.add_argument('--cgi_ratio', type=float, default=0.5,
                        help='ratio for CGIPooling (default: 0.5)')
    parser.add_argument('--gsa_ratio', type=float, default=0.5,
                        help='ratio for GSAPooling (default: 0.5)')      
    parser.add_argument('--pooling', type=str, default='TopK',
                        choices=model_dict.keys(),
                        help='which model to use (default: TopK)')
    args = parser.parse_args()
    if args.pooling == 'TopK':
        ratio = args.topk_ratio
    elif args.pooling == 'SAG':
        ratio = args.sag_ratio
    elif args.pooling == 'ASAP':
        ratio = args.asa_ratio
    elif args.pooling == 'PAN':
        ratio = args.pan_ratio
    elif args.pooling == 'CGI':
        ratio = args.cgi_ratio
    elif args.pooling == 'GSA':
        ratio = args.gsa_ratio    
    else:
        ratio = 0.5    
    is_cpu = True if args.cuda_num == -1 else False
    RDLogger.DisableLog('rdApp.*')
    task = TaskConfig(args.dataset)

    device = torch.device("cuda:" + str(args.cuda_num)) if not is_cpu and torch.cuda.is_available() else torch.device("cpu")
    if os.path.exists(task.data_path):
        dataset = MolDatasetInMemory(root='datasets/', filename = task.data_filename)
        print("Existed dataset loaded: " + task.data_path)
    else:
        dataset = GenerateDataset(task.mols, task.labels, inMemory=True, overwrite=True, filename=task.data_filename)
    print("\nCurrent dataset: "+args.dataset+", include {:d} molecules and {:d} ".format(len(dataset), task.num_classes)+task.task_type+" tasks\n")   

    loader_tr, loader_va, loader_te = RandomScaffoldSplit(dataset, task.smiles, k=args.k, drop_last=task.drop_last,
                                                          batch_size=task.batch_size, start_seed = args.split_seed,
                                                          null_value=0, frac_train=0.7, frac_valid=0.15, frac_test=0.15, num_workers=args.num_workers)
        
    load_path = None
 
    test_metric = []
    for i in range(args.k):  
        for j in range(args.run_times):
            model_class = model_dict[args.pooling]
            model = model_class(in_channels=dataset.num_node_features,
                    hidden_channels=args.hidden_channels_x,
                    out_channels=64,
                    num_classes=task.num_classes,
                    lin_before_conv=args.lin_before_conv,
                    ratio=ratio) 
            trainer = Trainer(args, model, device)      
            _, metric_te = trainer.fit_and_test(loader_tr[i], loader_va[i], loader_te[i], log_train_results=args.log_train_results,
                                                load_path=load_path, save_path='buffer/fintune_temp.pt')
            model.load_state_dict(torch.load('buffer/fintune_temp.pt'))
    
            test_metric.append(metric_te)
            print('\n********************{:d}\'s fold {:d}\'s run over********************'.format(i+1, j+1))
            # print('\n********************{:d}\'s fold over********************'.format(i+1))
            val = np.array(test_metric)
            if len(val.shape) == 1:
                print(task.eval_metric + ': {:.3f} +/- {:.3f}'.format(val.mean(), val.std()))   
            else:
                print('AUROC: {:.3f} +/- {:.3f}'.format(val[:,0].mean(), val[:,0].std())) 
                print('AUPRC: {:.3f} +/- {:.3f}'.format(val[:,1].mean(), val[:,1].std())) 
            print() 





















































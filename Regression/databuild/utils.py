
import pandas as pd
import numpy as np
import os
import torch
import json
import random

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from itertools import compress
from collections import defaultdict

from .mol_ops import (get_atoms_features_matrix,
                      get_bonds_features_matrix,
                      # get_fungroups,
                      get_decomp_mask,
                      get_decomp_labels)

from .dataset import MolDataset, MolDatasetInMemory

from models.baseline import (GCN, GAT, GIN, GraphSAGE, TopK, ASAP, EdgePool, GMT, SAG, Diff, MinCut,
                             TopK_e, SAG_e, EdgePool_e, HaarPool, Diff_e, MinCut_e)


dataset_list = ['bbbp', 'bace', 'clintox', 'hiv', 'tox21', 'esol', 'freesolv', 'lipo', 'muv', 'sider', 'muta', 'qm7', 'qm8', 'qm9', 'covid']
tasks = {'bbbp': ['p_np'],
         'bace': ['Class'],
         'clintox': ['FDA_APPROVED', 'CT_TOX'],
         'hiv': ['HIV_active'],
         'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
         'esol': ['measured log solubility in mols per litre'],
         'freesolv': ['expt'],
         'lipo': ['exp'],
         'muv': ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'],
         'sider': ['Hepatobiliary disorders',
                   'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
                   'Investigations', 'Musculoskeletal and connective tissue disorders',
                   'Gastrointestinal disorders', 'Social circumstances',
                   'Immune system disorders', 'Reproductive system and breast disorders',
                   'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                   'General disorders and administration site conditions',
                   'Endocrine disorders', 'Surgical and medical procedures',
                   'Vascular disorders', 'Blood and lymphatic system disorders',
                   'Skin and subcutaneous tissue disorders',
                   'Congenital, familial and genetic disorders',
                   'Infections and infestations',
                   'Respiratory, thoracic and mediastinal disorders',
                   'Psychiatric disorders', 'Renal and urinary disorders',
                   'Pregnancy, puerperium and perinatal conditions',
                   'Ear and labyrinth disorders', 'Cardiac disorders',
                   'Nervous system disorders',
                   'Injury, poisoning and procedural complications'],
         'qm7': ['u0_atom'],
         'qm8': ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'],
         # 'qm9': ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']
         'qm9': ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv'],
         'covid': ['isactive'],
         'muta': ['Mutagenicity']
         }

paths = {'bbbp': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/BBBP.csv',
         'bace': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/bace.csv',
         'clintox': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/clintox.csv',
         'hiv': '/data/ /Poolingg/Graph_Pooling_Benchmark/Regression/datasets/HIV.csv',
         'tox21': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/tox21.csv',
         'esol': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/esol.csv',
         'freesolv': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/freesolv.csv',
         'lipo': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/Lipophilicity.csv',
         'muv': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/muv.csv',
         'sider': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/sider.csv',
         'qm7': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/qm7.csv',
         'qm8': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/qm8.csv',
         'qm9': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/qm9.csv',
         'covid': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/covid.csv',
         'muta': '/data/ /Pooling/Graph_Pooling_Benchmark/Regression/datasets/Mutagenicity.csv'}

batch_sizes = {'bbbp': 64,
               'bace': 64,
               'clintox': 64,
               'hiv': 128,
               'tox21': 128,
               'esol': 16,
               'freesolv': 32,
               'lipo': 64,
               'muv': 256,
               'sider': 64,
               'qm7': 64,
               'qm8': 2048,
               'qm9': 2048,
               'covid': 128,
               'muta': 128}
#if args.pooling == 'AsymCheegerCut':
    #dbatch_sizes['qm7'] = 64
# baseline_list = ['gcn', 'gat', 'gin', 'sage', 'topk', 'asap', 'edgepool', 'diff', 'sag', 'mincut']
baseline_models = {'gcn': GCN,
                   'gat': GAT,
                   'gin': GIN,
                   'sage': GraphSAGE,
                   'topk': TopK,
                   'asap': ASAP,
                   'edgepool': EdgePool,
                   'sag': SAG,
                   'diff': Diff,
                   'mincut': MinCut,
                   'haar': HaarPool,
                   'topk_e': TopK_e,
                   'sag_e': SAG_e,
                   'edgepool_e': EdgePool_e,
                   'diff_e': Diff_e,
                   'mincut_e': MinCut_e}


def dataset_max_nodes(dataset):
    temp = 0
    for data in dataset:
        if data.x.size(0)>temp:
            temp = data.x.size(0)
    return temp

class TaskConfig():
    def __init__(self, dataset, semi_sup=False):
        assert dataset in dataset_list, "Please check the task name: should be one of {}".format(dataset_list)
        self.dataset = dataset
        df = pd.read_csv(paths[dataset], sep=',')
        smiles = df['smiles'].tolist()
        mols = [Chem.MolFromSmiles(i) for i in smiles]
        labels = df[tasks[dataset]]
        # if dataset == 'muv' or dataset == 'tox21':
        labels = labels.fillna(-1)
        self.mols, self.labels, self.smiles = DataPrePop(mols, labels.values.tolist(), smiles)
        
    @property
    def num_classes(self):
        return len(tasks[self.dataset])
        # return num_classes[self.task]
    
    @property
    def task_names(self):
        return tasks[self.dataset]
    
    @property
    def batch_size(self):
        return batch_sizes[self.dataset]
    
    @property
    def task_type(self):
        if self.dataset in ['bbbp', 'bace', 'clintox', 'hiv', 'tox21', 'muv', 'sider', 'covid', 'muta']:
            return 'classification'
        else:
            return 'regression'
        
    @property
    def data_filename(self):
        return self.dataset + '.pt'
        
    @property
    def data_path(self):
        return "datasets/processed/" + self.data_filename
    
    @property
    def drop_last(self):
        if self.dataset == 'freesolv':
            return True
        return False
    
    @property
    def eval_metric(self):
        if self.task_type == 'classification':
            return 'AUROC'
        elif self.dataset in ['esol', 'freesolv', 'lipo']:
            return 'RMSE'
        else:
            return 'MAE'

class BaselineConfig():
    def __init__(self, args):
        baseline = args.baseline
        file = args.para_config_path
        assert baseline in baseline_models.keys(), "Please check the baseline name: should be one of {}".format(baseline_models.keys())
        self.model = baseline_models[baseline]
        if file is not None and os.path.exists(file):
            self.config_args(args, file)
        
    def config_args(self, args, file):
        paraconfig = pd.read_csv(file, index_col='model')
        args.hidden_channels = int(paraconfig.loc[args.baseline, 'hidden_channels'])
        args.dropout = float(paraconfig.loc[args.baseline, 'dropout'])
        args.lr = float(paraconfig.loc[args.baseline, 'lr'])
        args.ratio = float(paraconfig.loc[args.baseline, 'ratio'])


class LabelNormalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor, device):
        """tensor is taken as a sample to calculate the mean and std"""
        self.device = device
        self.mean = torch.mean(tensor, dim=0).to(device)
        self.std = torch.std(tensor, dim=0).to(device)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean'].to(self.device)
        self.std = state_dict['std'].to(self.device)


def DataRead(task):
    df = pd.read_csv(paths[task], sep=',')
    smiles = df['smiles'].tolist()
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    labels = df[tasks[task]]
    if task == 'muv' or task == 'tox21':
        labels = labels.fillna(-1)
    mols, labels, smiles = DataPrePop(mols, labels.values.tolist(), smiles)
    return mols, labels, smiles, len(tasks[task])


def DataPrePop(mols, labels, smiles=None):
    """
    This function is to delete single atom molecules or whose SMILES filed converting to rdkit.Chem.rdchem.Mol
    mols and labels are List here
    """
    for i in range(len(mols))[::-1]:
        if mols[i] is None:
            mols.pop(i)
            labels.pop(i)
            if smiles is not None:
                smiles.pop(i)
        elif mols[i].GetNumAtoms() == 1:
            mols.pop(i)
            labels.pop(i)    
            if smiles is not None:
                smiles.pop(i)
    if smiles is not None:
        return mols, labels, smiles
    return mols, labels


def GenerateDataList(mols, labels, type_legnth=100, isinfo=True):
    """
    Parameters
    ----------
    mols : List
        A list of tpye rdkit.Chem.rdchem.Mol.
    labels : List
        A list of molecular property labels. Same length with mols.
    type_legnth : INT, optional
        The length of one-hot atom kinds. The default is 100.
    isinfo : BOOL, optional
        Whether consider the extra group information. The default is False.
    hierarchy : Lit, optional
        See databuild.molops.get_pool_labels
               
    Returns
    -------
    data_list : List
        A list of type torch_geometric.data.Data
    """
    assert len(mols) == len(labels), 'mols and labels must have the same length'    
    data_list = []
    length = len(mols)
    for i in range(length):
        mol = mols[i]
        # try:
        x = get_atoms_features_matrix(mol, type_legnth)
        # except IndexError:
        #     print('IndexError: Atom types more than {0}, please set a larger value of type_legnth'.format(type_legnth))
        edge_attr, edge_index = get_bonds_features_matrix(mol)
        y = labels[i]
        data = Data(x=torch.tensor(x, dtype=torch.float32),
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                    edge_index=torch.LongTensor(edge_index).t().contiguous(),
                    y=torch.tensor(y, dtype=torch.float32).unsqueeze(0)) 

        data_list.append(data)
        if i % 200 == 0 or i == length-1:
            print("\rPackaging molecules, finish {:.1f}%".format(i * 100 / (length - 1)), end="", flush=True)
    print(end="\n")        
    return data_list

       
def GenerateDataset(mols, labels, type_legnth=100,
                    inMemory = True,
                    overwrite = True, 
                    root='.\\datasets\\',
                    filename = 'data.pt'):
    print("Generating dataset...")
    if overwrite and os.path.exists(root+'processed\\'+filename):
        os.remove(root+'processed\\'+filename)
    data_list = GenerateDataList(mols, labels, type_legnth)
    if inMemory:
        return MolDatasetInMemory(data_list, root, filename)
    else:
        return MolDataset(data_list, root, filename)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def GenerateData(mol, type_legnth=100):
    try:
        x = get_atoms_features_matrix(mol, type_legnth)
    except IndexError:
        print('IndexError: Atom types more than {0}, please set a larger value of type_legnth'.format(type_legnth))
    edge_attr, edge_index = get_bonds_features_matrix(mol)
    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                edge_index=torch.LongTensor(edge_index).t().contiguous())
    data.batch = None

    return data


def KFoldSplit(dataset, k=10, batch_size=64, shuffle=True):
    assert k > 1
    size = len(dataset)
    flag = True 
    while flag:     
        np.random.seed(np.random.randint(50, 999999))        
        idxs = np.random.permutation(size) if shuffle else np.array(range(size))
        fold_size = size // k
        loader_tr = []
        loader_va = []
        loader_te = []    
        for i in range(k):
            idx_te = range(fold_size*i, fold_size*(i+1))
            idxs_te = idxs[list(idx_te)]  
            idxs_va, idxs_tr = np.split([x for x in idxs if x not in idxs_te], [fold_size])
            idxs_te, idxs_va, idxs_tr = idxs_te.tolist(), idxs_va.tolist(), idxs_tr.tolist()   
            loader_tr_n = DataLoader(dataset[idxs_tr], batch_size)
            loader_va_n = DataLoader(dataset[idxs_va], batch_size, shuffle=False, worker_init_fn=seed_worker)
            loader_te_n = DataLoader(dataset[idxs_te], batch_size, shuffle=False, worker_init_fn=seed_worker)  
            if auc_check(loader_tr_n) and auc_check(loader_va_n) and auc_check(loader_te_n):
                flag = False
            else:
                print("label imba, start resplitting")
                flag = True
                break                                 
            loader_tr.append(loader_tr_n)
            loader_va.append(loader_va_n)
            loader_te.append(loader_te_n)                           
    return loader_tr, loader_va, loader_te



def RandomSplit(dataset, batch_size=64, seed=1234):
    data_tr, data_te = train_test_split(dataset, test_size=0.1, random_state=seed)
    data_tr, data_va = train_test_split(data_tr, test_size=0.1, random_state=seed)
    loader_tr = DataLoader(data_tr, batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker)
    loader_va = DataLoader(data_va, batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker)
    loader_te = DataLoader(data_te, batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker)

    return loader_tr, loader_va, loader_te


def auc_check(loader):
    flag = True
    for i, batch in enumerate(loader):
        # ratio = (batch.y>0).nonzero().size(0) / batch.y.size(0)
        # print("batch {:d}: label ratio (1:0) = {:.4f}".format(i, ratio))
        if 1 not in batch.y:
            flag = False
            break
            # print("no positive label in batch[{:d}]".format(i))
        elif 0 not in batch.y:
            flag = False
            break
            # print("no negative label in batch[{:d}]".format(i))
    # if flag:
    #     print("label is clean")
    return flag

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)

def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                          frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = list(scaffolds.values())
    scaffold_sets = rng.permutation([item for sublist in scaffold_sets for item in sublist])

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) < n_total_valid:
            valid_idx.append(scaffold_set)
        elif len(test_idx) < n_total_test:
            test_idx.append(scaffold_set)
        else:
            train_idx.append(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset

def max_nodes_filter(dataset, max_nodes=100):
    idxs = []
    for i in range(len(dataset)):
        if dataset[i].x.size(0) <= max_nodes:
            idxs.append(i)
    return dataset[idxs]

def RandomScaffoldSplit(dataset, smiles_list, check_class=False, k=10, batch_size=64, start_seed=1234,
                        null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, **kargs):
    loader_tr = []
    loader_va = []
    loader_te = []
    for i in range(k):
        seed = i + start_seed
        while True:
            data_tr, data_va, data_te = random_scaffold_split(dataset, smiles_list,
                                                              null_value = null_value,
                                                              frac_train = frac_train,
                                                              frac_valid = frac_valid,
                                                              frac_test = frac_test,
                                                              seed = seed)
            # data_tr = max_nodes_filter(data_tr)
            # data_va = max_nodes_filter(data_va)
            # data_te = max_nodes_filter(data_te)
            
            
            loader_tr_n = DataLoader(data_tr, batch_size, shuffle=False, worker_init_fn=seed_worker, **kargs)
            loader_va_n = DataLoader(data_va, batch_size, shuffle=False, worker_init_fn=seed_worker, **kargs)
            loader_te_n = DataLoader(data_te, batch_size, shuffle=False, worker_init_fn=seed_worker, **kargs)
            if check_class:
                if auc_check(loader_tr_n) and auc_check(loader_va_n) and auc_check(loader_te_n):
                    break
                else:
                    print("label imba, start resplitting")
                    seed = np.random.randint(50, 999999)
            else:
                break
        loader_tr.append(loader_tr_n)
        loader_va.append(loader_va_n)
        loader_te.append(loader_te_n)
        print("\rSplitting, finish {:d}/{:d}  ".format(i + 1, k), end="", flush=True)
    print(end="\n")
    return loader_tr, loader_va, loader_te

def ScaffoldSplit(dataset, smiles_list, batch_size=64, task_idx=None, null_value=0,
                  frac_train=0.8, frac_valid=0.1, frac_test=0.1, **kargs):
    
    data_tr, data_va, data_te = scaffold_split(dataset, smiles_list, task_idx, null_value,
                                               frac_train, frac_valid, frac_test,
                                               return_smiles=False)
    loader_tr = DataLoader(data_tr, batch_size, shuffle=False, worker_init_fn=seed_worker, **kargs)
    loader_va = DataLoader(data_va, batch_size, shuffle=False, worker_init_fn=seed_worker, **kargs)
    loader_te = DataLoader(data_te, batch_size, shuffle=False, worker_init_fn=seed_worker, **kargs) 
    return loader_tr, loader_va, loader_te

def SmilesAugment(smiles_list, seed=1234, shuffle=False, aug_times=1,
                  rules_path='databuild/isostere_transformations_new.json'):
    '''
    this function is modified from https://github.com/illidanlab/MoCL-DK
    '''
    random.seed = seed
    if shuffle:
        random.shuffle(smiles_list)
    rules = json.load(open(rules_path))
    # rules = json.load(open('rules_carbon_drop.json'))
    print('# rules {:d}'.format(len(rules)))
    new_smiles = []
    for s in tqdm(smiles_list, desc="rule_indicator processing"):
        mol_obj = Chem.MolFromSmiles(s)
        mol_prev = mol_obj
        mol_next = None
        rule_indicator = np.zeros(len(rules), dtype=np.int)
        for i in range(len(rules)):
            rule = rules[i]
            rxn = AllChem.ReactionFromSmarts(rule['smarts'])
            products = rxn.RunReactants((mol_obj,))
            rule_indicator[i] = len(products)
            
        non_zero_idx = list(np.where(rule_indicator!=0)[0])
        cnt = -1
        while len(non_zero_idx)!=0:
            col_idx = random.choice(non_zero_idx)
            # calculate counts
            rule = rules[col_idx]
            rxn = AllChem.ReactionFromSmarts(rule['smarts'])
            products = rxn.RunReactants((mol_prev,))
            cnt = len(products)
            if cnt != 0:
                break
            else:
                non_zero_idx.remove(col_idx)
        if cnt >= 1:
            aug_idx = random.choice(range(cnt))
            mol = products[aug_idx][0]
            try:
                Chem.SanitizeMol(mol)
            except: # TODO: add detailed exception
                pass
            mol_next = mol
            mol_prev = mol
            #rule_indicator[row_idx, col_idx] -= 1
        else:
            mol_next = mol_prev
        new_smiles.append(Chem.MolToSmiles(mol_next))             
         
    return smiles_list, new_smiles


def set_seed(seed=1234, is_cpu=False):
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(is_cpu)
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.badatahmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False










































































































































































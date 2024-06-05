# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:16:02 2022

@author: Fanding Xu
"""

import os
import torch
from torch_geometric.data import Dataset, InMemoryDataset

class MolDataset(Dataset):
    def __init__(self, data_list, root='.\\datasets\\', filename='data.pt', transform=None):
        self.data_list = data_list
        self.filename = filename
        super().__init__(root, transform)
        
    @property
    def processed_file_names(self):
        return self.filename

    def process(self):
        for i in range(len(self.data_list)):
            data = self.data_list[i]
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
    def len(self):
        return len(self.data_list)
        
    def get(self, i):
        data = torch.load(os.path.join(self.processed_dir, f'data_{i}.pt'))
        return data
        
        
class MolDatasetInMemory(InMemoryDataset):
    def __init__(self, data_list=None, root='./datasets/', filename='data.pt', transform=None):
        self.filename = filename
        # if load_exist:
        #     assert os.path.exists('./datasets/processed/'+filename), "buffer file "+filename+"not found"
        #     self.data, self.slices = torch.load('./datasets/processed/'+filename)
        # else:
        self.data_list = data_list 
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return self.filename

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected, AddSelfLoops
from torch_geometric.utils import to_dense_adj
import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree


# In[2]:


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
import numpy as np
np.random.seed(42)
import random
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from sklearn.model_selection import KFold
from torch_geometric.nn import global_mean_pool
import psutil
import warnings
warnings.filterwarnings('ignore')
import statistics


# ## TopKPooling with HierarchicalGCN (2019)

# In[3]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TopKPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

class HierarchicalGCN_TOPK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_TOPK, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN and pooling layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        # Second GCN and pooling layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.bn3(x)

        # Mean pooling over the nodes
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)

        # Fully connected layers
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
max_nodes = 100
data_path = '/data/zeyu/Pooling/IMDB-MULTI'  # Adjust path as needed
dataset = TUDataset(data_path, 'IMDB-MULTI', transform=T.Constant(), use_node_attr=True, pre_filter=lambda data: data.num_nodes <= max_nodes,)

num_classes = dataset.num_classes
in_channels = dataset.num_features
dataset = dataset.shuffle()
def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 修改点：直接使用整个图的标签进行训练
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# In[4]:


import time

kf = KFold(n_splits=5)
results = []
patience = 150  # 早停的容忍度
min_delta = 0.00001  # 精度改进的最小变化

results = []
for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0  # 初始化总GPU内存使用

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = HierarchicalGCN_TOPK(
            in_channels=dataset.num_features,
            hidden_channels=64,
            out_channels=64,
            num_classes=dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        best_epoch = 0
        times = []

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, train_loader)
            val_acc = test(model, test_loader)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc = val_acc  # 保存最佳精度
            times.append(time.time() - start)

            # 检查早停条件
            if epoch - best_epoch >= patience:
                print(f'Seed: {seed}, Fold: {fold+1}, Early stopping at epoch {epoch}')
                break

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    accuracies = [x[0] for x in fold_results]
    std_accuracy = statistics.stdev(accuracies)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存
    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Std accuracy = {std_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


mean_accuracies = [x[0] for x in results]
std_accuracies = [x[1] for x in results]
total_times = [x[2] for x in results]
avg_memories = [x[3] for x in results]

overall_mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)
overall_std_accuracy = statistics.stdev(mean_accuracies)
overall_total_time = sum(total_times) / len(total_times)
overall_avg_memory = sum(avg_memories) / len(avg_memories)

if torch.cuda.is_available():
    avg_gpu_memories = [x[3] for x in results]
    overall_avg_gpu_memory = sum(avg_gpu_memories) / len(avg_gpu_memories)
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB, Average GPU Memory = {overall_avg_gpu_memory:.2f} MB')
else:
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB')


# ## SAGPooling with HierarchicalGCN (2019)

# In[40]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SAGPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm

class HierarchicalGCN_SAG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_SAG, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = SAGPooling(hidden_channels, ratio=0.5)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = SAGPooling(hidden_channels, ratio=0.5)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN and pooling layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        # Second GCN and pooling layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.bn3(x)

        # Mean pooling over the nodes
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)

        # Fully connected layers
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


max_nodes = 100
data_path = '/data/zeyu/Pooling/IMDB-MULTI'  # Adjust path as needed
dataset = TUDataset(data_path, 'IMDB-MULTI', transform=T.Constant(), pre_filter=lambda data: data.num_nodes <= max_nodes,)
dataset = dataset.shuffle()

num_classes = dataset.num_classes
in_channels = dataset.num_features

def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 修改点：直接使用整个图的标签进行训练
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# In[41]:


kf = KFold(n_splits=5)
results = []
patience = 150  # 早停的容忍度
min_delta = 0.00001  # 精度改进的最小变化

results = []
for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0  # 初始化总GPU内存使用

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = HierarchicalGCN_SAG(
            in_channels=dataset.num_features,
            hidden_channels=64,
            out_channels=64,
            num_classes=dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        best_epoch = 0
        times = []

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, train_loader)
            val_acc = test(model, test_loader)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc = val_acc  # 保存最佳精度
            times.append(time.time() - start)

            # 检查早停条件
            if epoch - best_epoch >= patience:
                print(f'Seed: {seed}, Fold: {fold+1}, Early stopping at epoch {epoch}')
                break

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    accuracies = [x[0] for x in fold_results]
    std_accuracy = statistics.stdev(accuracies)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存
    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Std accuracy = {std_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


mean_accuracies = [x[0] for x in results]
std_accuracies = [x[1] for x in results]
total_times = [x[2] for x in results]
avg_memories = [x[3] for x in results]

overall_mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)
overall_std_accuracy = statistics.stdev(mean_accuracies)
overall_total_time = sum(total_times) / len(total_times)
overall_avg_memory = sum(avg_memories) / len(avg_memories)

if torch.cuda.is_available():
    avg_gpu_memories = [x[3] for x in results]
    overall_avg_gpu_memory = sum(avg_gpu_memories) / len(avg_gpu_memories)
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB, Average GPU Memory = {overall_avg_gpu_memory:.2f} MB')
else:
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB')


# ## ASAPooling with HierarchicalGCN (2020)

# In[42]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, ASAPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm

class HierarchicalGCN_ASA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_ASA, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = ASAPooling(hidden_channels, ratio=0.5)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = ASAPooling(hidden_channels, ratio=0.5)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN and pooling layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.bn1(x)
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)

        # Second GCN and pooling layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.bn2(x)
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, batch=batch)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.bn3(x)

        # Mean pooling over the nodes
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)

        # Fully connected layers
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


max_nodes = 100
data_path = '/data/zeyu/Pooling/IMDB-MULTI'  # Adjust path as needed
dataset = TUDataset(data_path, 'IMDB-MULTI', transform=T.Constant(), pre_filter=lambda data: data.num_nodes <= max_nodes,)
dataset = dataset.shuffle()

num_classes = dataset.num_classes
in_channels = dataset.num_features

def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 修改点：直接使用整个图的标签进行训练
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# In[43]:


import time
kf = KFold(n_splits=5)
results = []
patience = 150  # 早停的容忍度
min_delta = 0.0001  # 精度改进的最小变化

results = []
for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0  # 初始化总GPU内存使用

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = HierarchicalGCN_ASA(
            in_channels=dataset.num_features,
            hidden_channels=64,
            out_channels=64,
            num_classes=dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        best_epoch = 0
        times = []

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, train_loader)
            val_acc = test(model, test_loader)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc = val_acc  # 保存最佳精度
            times.append(time.time() - start)

            # 检查早停条件
            if epoch - best_epoch >= patience:
                print(f'Seed: {seed}, Fold: {fold+1}, Early stopping at epoch {epoch}')
                break

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    accuracies = [x[0] for x in fold_results]
    std_accuracy = statistics.stdev(accuracies)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存
    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Std accuracy = {std_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


mean_accuracies = [x[0] for x in results]
std_accuracies = [x[1] for x in results]
total_times = [x[2] for x in results]
avg_memories = [x[3] for x in results]

overall_mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)
overall_std_accuracy = statistics.stdev(mean_accuracies)
overall_total_time = sum(total_times) / len(total_times)
overall_avg_memory = sum(avg_memories) / len(avg_memories)

if torch.cuda.is_available():
    avg_gpu_memories = [x[3] for x in results]
    overall_avg_gpu_memory = sum(avg_gpu_memories) / len(avg_gpu_memories)
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB, Average GPU Memory = {overall_avg_gpu_memory:.2f} MB')
else:
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB')


# ## PANPooling with HierarchicalGCN (2020)

# In[44]:


from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import spspmm
from torch_sparse import coalesce
from torch_sparse import eye
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_scatter import scatter_max

class PANPooling(torch.nn.Module):
    r""" General Graph pooling layer based on PAN, which can work with all layers.
    """
    def __init__(self, in_channels, ratio=0.5, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=3, panpool_filter_weight=None):
        super(PANPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.filter_size = filter_size
        if panpool_filter_weight is None:
            self.panpool_filter_weight = torch.nn.Parameter(0.5 * torch.ones(filter_size), requires_grad=True)

        self.transform = Parameter(torch.ones(in_channels), requires_grad=True)

        if pan_pool_weight is None:
            #self.weight = torch.tensor([0.7, 0.3], device=self.transform.device)
            self.pan_pool_weight = torch.nn.Parameter(0.5 * torch.ones(2), requires_grad=True)
        else:
            self.pan_pool_weight = pan_pool_weight

    def forward(self, x, edge_index, M=None, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # Path integral
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, edge_weight = self.panentropy_sparse(edge_index, num_nodes)

        # weighted degree
        num_nodes = x.size(0)
        degree = torch.zeros(num_nodes, device=edge_index.device)
        degree = scatter_add(edge_weight, edge_index[0], out=degree)

        # linear transform
        xtransform = torch.matmul(x, self.transform)

        # aggregate score
        x_transform_norm = xtransform #/ xtransform.norm(p=2, dim=-1)
        degree_norm = degree #/ degree.norm(p=2, dim=-1)
        score = self.pan_pool_weight[0] * x_transform_norm + self.pan_pool_weight[1] * degree_norm

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight

    def panentropy_sparse(self, edge_index, num_nodes):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panpool_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            #indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panpool_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')


# In[45]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, ASAPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm

class HierarchicalGCN_PAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_PAN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = PANPooling(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = PANPooling(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN and pooling layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.bn1(x)
        x, edge_index, _, batch, perm, score_perm = self.pool1(x, edge_index, batch=batch, M=None)

        # Second GCN and pooling layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.bn2(x)
        x, edge_index, _, batch, perm, score_perm = self.pool2(x, edge_index, batch=batch, M=None)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.bn3(x)

        # Mean pooling over the nodes
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)

        # Fully connected layers
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


max_nodes = 100
data_path = '/data/zeyu/Pooling/IMDB-MULTI'  # Adjust path as needed
dataset = TUDataset(data_path, 'IMDB-MULTI', transform=T.Constant(), pre_filter=lambda data: data.num_nodes <= max_nodes,)
dataset = dataset.shuffle()
num_classes = dataset.num_classes
in_channels = dataset.num_features

def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 修改点：直接使用整个图的标签进行训练
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# In[46]:


import time

kf = KFold(n_splits=5)
patience = 150  # 早停的容忍度
min_delta = 0.00001  # 精度改进的最小变化

results = []
for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0  # 初始化总GPU内存使用

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = HierarchicalGCN_PAN(
            in_channels=dataset.num_features,
            hidden_channels=64,
            out_channels=64,
            num_classes=dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        best_epoch = 0
        times = []

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, train_loader)
            val_acc = test(model, test_loader)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc = val_acc  # 保存最佳精度
            times.append(time.time() - start)

            # 检查早停条件
            if epoch - best_epoch >= patience:
                print(f'Seed: {seed}, Fold: {fold+1}, Early stopping at epoch {epoch}')
                break

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    accuracies = [x[0] for x in fold_results]
    std_accuracy = statistics.stdev(accuracies)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存
    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Std accuracy = {std_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


mean_accuracies = [x[0] for x in results]
std_accuracies = [x[1] for x in results]
total_times = [x[2] for x in results]
avg_memories = [x[3] for x in results]

overall_mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)
overall_std_accuracy = statistics.stdev(mean_accuracies)
overall_total_time = sum(total_times) / len(total_times)
overall_avg_memory = sum(avg_memories) / len(avg_memories)

if torch.cuda.is_available():
    avg_gpu_memories = [x[3] for x in results]
    overall_avg_gpu_memory = sum(avg_gpu_memories) / len(avg_gpu_memories)
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB, Average GPU Memory = {overall_avg_gpu_memory:.2f} MB')
else:
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB')


# ## CoPooling with HierarchicalGCN (2023)

# In[47]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, add_self_loops
from typing import Callable, Optional, Union
from torch_sparse import coalesce, transpose
from torch_scatter import scatter
from torch import Tensor
def cumsum(x: Tensor, dim: int = 0) -> Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`torch.cumsum`, prepends the output with zero.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    Example:
        >>> x = torch.tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])

    """
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))

    return out

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr

def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                           self.temp)


class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes) # in case all the edges are removed

        edge_index = edge_index.type(torch.long)
        row, col = edge_index
        # print(row, col)
        # print(edge_weight.shape, row.shape, num_nodes)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class graph_attention(torch.nn.Module):
    # reference: https://github.com/gordicaleksa/pytorch-GAT/blob/39c8f0ee634477033e8b1a6e9a6da3c7ed71bbd1/models/definitions/GAT.py#L324
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, dropout_prob=0.6, log_attention_weights=False):
        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

    def forward(self, x, edge_index):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features = x  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = scores_source_lifted + scores_target_lifted

        return torch.sigmoid(scores_per_edge)

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted



class CoPooling(torch.nn.Module):
    # reference for GAT code: https://github.com/PetarV-/GAT
    # reference for generalized pagerank code: https://github.com/jianhao2016/GPRGNN
    def __init__(self, ratio=0.5, K=0.05, edge_ratio=0.6, nhid=64, alpha=0.1, Init='Random', Gamma=None):
        super(CoPooling, self).__init__()
        self.ratio = ratio
        self.calc_information_score = NodeInformationScore()
        self.edge_ratio = edge_ratio

        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        score_dim = 32
        self.G_att = graph_attention(num_in_features=nhid, num_out_features=score_dim, num_of_heads=1)

        self.weight = Parameter(torch.Tensor(2*nhid, nhid))
        nn.init.xavier_uniform_(self.weight.data)
        self.bias = Parameter(torch.Tensor(nhid))
        nn.init.zeros_(self.bias.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.zeros_(self.bias.data)
        self.prop1.reset_parameters()
        self.G_att.init_params()

    def forward(self, x, edge_index, edge_attr, batch=None, nodes_index=None, node_attr=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        ori_batch = batch.clone()
        device = x.device
        num_nodes = x.shape[0]

        # cut edges based on scores
        x_cut = self.prop1(x, edge_index) # run generalized pagerank to update features

        attention = self.G_att(x_cut, edge_index) # get the attention weights after sigmoid
        attention = attention.sum(dim=1) #sum the weights on head dim
        edge_index, attention = add_self_loops(edge_index, attention, 1.0, num_nodes) # add self loops in case no edges

        # to get a systemitic adj matrix
        edge_index_t, attention_t = transpose(edge_index, attention, num_nodes, num_nodes)
        edge_tmp = torch.cat((edge_index, edge_index_t), 1)
        att_tmp = torch.cat((attention, attention_t),0)
        edge_index, attention = coalesce(edge_tmp, att_tmp, num_nodes, num_nodes, 'mean')

        attention_np = attention.cpu().data.numpy()
        cut_val = np.percentile(attention_np, int(100*(1-self.edge_ratio))) # this is for keep the top edge_ratio edges
        attention = attention * (attention >= cut_val) # keep the edge_ratio higher weights of edges

        kep_idx = attention > 0.0
        cut_edge_index, cut_edge_attr = edge_index[:, kep_idx], attention[kep_idx]

        # Graph Pooling based on nodes
        x_information_score = self.calc_information_score(x, cut_edge_index, cut_edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)
        perm = topk(score, self.ratio, batch)
        x_topk = x[perm]
        batch = batch[perm]
        if nodes_index is not None:
            nodes_index = nodes_index[perm]

        if node_attr is not None:
            node_attr = node_attr[perm]
        if cut_edge_index is not None or cut_edge_index.nelement() != 0:
            induced_edge_index, induced_edge_attr = filter_adj(cut_edge_index, cut_edge_attr, perm, num_nodes=num_nodes)
        else:
            print('All edges are cut!')
            induced_edge_index, induced_edge_attr = cut_edge_index, cut_edge_attr

        # update node features
        attention_dense = (to_dense_adj(cut_edge_index, edge_attr=cut_edge_attr, max_num_nodes=num_nodes)).squeeze()
        x = F.relu(torch.matmul(torch.cat((x_topk, torch.matmul(attention_dense[perm],x)), 1), self.weight) + self.bias)

        return x, induced_edge_index, perm, induced_edge_attr, batch, nodes_index, node_attr, attention_dense


# In[48]:


import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader,Data
from torch_geometric.nn import GCNConv
from torch.utils.data import random_split
import os
import os.path as osp
import argparse
import warnings

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, EdgePooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm


class HierarchicalGCN_CoPooling(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_CoPooling, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
        self.pool1 = CoPooling(ratio=0.5, K=1, edge_ratio=0.6, nhid=64, alpha=0.1, Init='Random', Gamma=1.0)
        self.pool2 = CoPooling(ratio=0.5, K=1, edge_ratio=0.6, nhid=64, alpha=0.1, Init='Random', Gamma=1.0)

    def forward(self, data):
        x, edge_index, batch, num_node, num_edge = data.x, data.edge_index, data.batch, data.num_node, data.num_edge
        batch_size = int(batch.max() + 1)

        # First GCN and pooling layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.bn1(x)
        x, edge_index, perm, _, batch, _, _, _ = self.pool1(x, edge_index, edge_attr=None, batch=batch)

        # Second GCN and pooling layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.bn2(x)
        x, edge_index, perm, _, batch, _, _, _ = self.pool2(x, edge_index, edge_attr=None, batch=batch)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.bn3(x)

        # Mean pooling over the nodes
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)

        # Fully connected layers
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


max_nodes = 100
data_path = '/data/zeyu/Pooling/IMDB-MULTI'  # Adjust path as needed
dataset = TUDataset(data_path, 'IMDB-MULTI', transform=T.Constant(), pre_filter=lambda data: data.num_nodes <= max_nodes,)
dataset = dataset.shuffle()
num_features = dataset.num_features
num_classes = dataset.num_classes
in_channels = dataset.num_features
dataset1 = list()
for i in range(len(dataset)):
    data1 = Data(x=dataset[i].x, edge_index =                  dataset[i].edge_index, y = dataset[i].y)
    data1.num_node = dataset[i].num_nodes
    data1.num_edge = dataset[i].edge_index.size(1)
    dataset1.append(data1)

dataset = dataset1


def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 修改点：直接使用整个图的标签进行训练
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# In[49]:


from torch.utils.data import Subset
patience = 150  # 早停的容忍度
min_delta = 0.00001  # 精度改进的最小变化
kf = KFold(n_splits=5)
results = []
from time import time
for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)


        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = HierarchicalGCN_CoPooling(in_channels=in_channels, hidden_channels=64, out_channels=64, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = test_acc = 0

        best_val_acc = 0
        best_epoch = 0
        times = []

        for epoch in range(1, 201):
            start = time()
            train_loss = train(model, train_loader)
            val_acc = test(model, test_loader)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc = val_acc  # 保存最佳精度
            times.append(time() - start)

            # 检查早停条件
            if epoch - best_epoch >= patience:
                print(f'Seed: {seed}, Fold: {fold+1}, Early stopping at epoch {epoch}')
                break


        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB, GPU Memory Usage: {gpu_memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存
    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


# ## CGIPooling with HierarchicalGCN (2021)

# In[50]:


from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import GCNConv, GATConv, LEConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(init=False)
class SelectOutput:
    r"""The output of the :class:`Select` method, which holds an assignment
    from selected nodes to their respective cluster(s).

    Args:
        node_index (torch.Tensor): The indices of the selected nodes.
        num_nodes (int): The number of nodes.
        cluster_index (torch.Tensor): The indices of the clusters each node in
            :obj:`node_index` is assigned to.
        num_clusters (int): The number of clusters.
        weight (torch.Tensor, optional): A weight vector, denoting the strength
            of the assignment of a node to its cluster. (default: :obj:`None`)
    """
    node_index: Tensor
    num_nodes: int
    cluster_index: Tensor
    num_clusters: int
    weight: Optional[Tensor] = None

    def __init__(
        self,
        node_index: Tensor,
        num_nodes: int,
        cluster_index: Tensor,
        num_clusters: int,
        weight: Optional[Tensor] = None,
    ):
        if node_index.dim() != 1:
            raise ValueError(f"Expected 'node_index' to be one-dimensional "
                             f"(got {node_index.dim()} dimensions)")

        if cluster_index.dim() != 1:
            raise ValueError(f"Expected 'cluster_index' to be one-dimensional "
                             f"(got {cluster_index.dim()} dimensions)")

        if node_index.numel() != cluster_index.numel():
            raise ValueError(f"Expected 'node_index' and 'cluster_index' to "
                             f"hold the same number of values (got "
                             f"{node_index.numel()} and "
                             f"{cluster_index.numel()} values)")

        if weight is not None and weight.dim() != 1:
            raise ValueError(f"Expected 'weight' vector to be one-dimensional "
                             f"(got {weight.dim()} dimensions)")

        if weight is not None and weight.numel() != node_index.numel():
            raise ValueError(f"Expected 'weight' to hold {node_index.numel()} "
                             f"values (got {weight.numel()} values)")

        self.node_index = node_index
        self.num_nodes = num_nodes
        self.cluster_index = cluster_index
        self.num_clusters = num_clusters
        self.weight = weight




class Select(torch.nn.Module):
    r"""An abstract base class for implementing custom node selections as
    described in the `"Understanding Pooling in Graph Neural Networks"
    <https://arxiv.org/abs/1905.05178>`_ paper, which maps the nodes of an
    input graph to supernodes in the coarsened graph.

    Specifically, :class:`Select` returns a :class:`SelectOutput` output, which
    holds a (sparse) mapping :math:`\mathbf{C} \in {[0, 1]}^{N \times C}` that
    assigns selected nodes to one or more of :math:`C` super nodes.
    """
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(self, *args, **kwargs) -> SelectOutput:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

def cumsum(x: Tensor, dim: int = 0) -> Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`torch.cumsum`, prepends the output with zero.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    Example:
        >>> x = torch.tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])

    """
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))

    return out

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr

def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels)
        self.fc2 = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.sigmoid(self.fc2(x))
        return x


class CGIPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5, non_lin=torch.tanh):
        super(CGIPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.non_lin = non_lin
        self.hidden_dim = in_channels
        self.transform = GraphConv(in_channels, self.hidden_dim)
        self.pp_conv = GraphConv(self.hidden_dim, self.hidden_dim)
        self.np_conv = GraphConv(self.hidden_dim, self.hidden_dim)

        self.positive_pooling = GraphConv(self.hidden_dim, 1)
        self.negative_pooling = GraphConv(self.hidden_dim, 1)

        self.discriminator = Discriminator(self.hidden_dim)
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        device = x.device  # 获取输入张量的设备信息

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_transform = F.leaky_relu(self.transform(x, edge_index), 0.2)
        x_tp = F.leaky_relu(self.pp_conv(x, edge_index), 0.2)
        x_tn = F.leaky_relu(self.np_conv(x, edge_index), 0.2)
        s_pp = self.positive_pooling(x_tp, edge_index).squeeze()
        s_np = self.negative_pooling(x_tn, edge_index).squeeze()

        perm_positive = topk(s_pp, 1, batch)
        perm_negative = topk(s_np, 1, batch)
        x_pp = x_transform[perm_positive] * self.non_lin(s_pp[perm_positive]).view(-1, 1)
        x_np = x_transform[perm_negative] * self.non_lin(s_np[perm_negative]).view(-1, 1)

        x_pp_readout = gap(x_pp, batch[perm_positive])
        x_np_readout = gap(x_np, batch[perm_negative])
        x_readout = gap(x_transform, batch)

        positive_pair = torch.cat([x_pp_readout, x_readout], dim=1)
        negative_pair = torch.cat([x_np_readout, x_readout], dim=1)

        real = torch.ones(positive_pair.shape[0], device=device)  # 将张量移动到相应设备
        fake = torch.zeros(negative_pair.shape[0], device=device)  # 将张量移动到相应设备
        #real_loss = self.loss_fn(self.discriminator(positive_pair), real)
        #fake_loss = self.loss_fn(self.discriminator(negative_pair), fake)
        #discrimination_loss = (real_loss + fake_loss) / 2

        score = (s_pp - s_np)

        perm = topk(score, self.ratio, batch)
        x = x_transform[perm] * self.non_lin(score[perm]).view(-1, 1)
        batch = batch[perm]

        filter_edge_index, filter_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, filter_edge_index, filter_edge_attr, batch, perm


# In[51]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SAGPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm

class HierarchicalGCN_CGI(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_CGI, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = CGIPool(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = CGIPool(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN and pooling layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.bn1(x)
        x, edge_index, _, batch, perm = self.pool1(x, edge_index, None, batch)

        # Second GCN and pooling layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.bn2(x)
        x, edge_index, _, batch, perm = self.pool2(x, edge_index, None, batch)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.bn3(x)

        # Mean pooling over the nodes
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)

        # Fully connected layers
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


max_nodes = 100
data_path = '/data/zeyu/Pooling/IMDB-MULTI'  # Adjust path as needed
dataset = TUDataset(data_path, 'IMDB-MULTI', transform=T.Constant(), pre_filter=lambda data: data.num_nodes <= max_nodes,)

num_classes = dataset.num_classes
in_channels = dataset.num_features
dataset = dataset.shuffle()

def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 修改点：直接使用整个图的标签进行训练
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# In[52]:


import time
kf = KFold(n_splits=5)
results = []
for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0  # 初始化总GPU内存使用

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = HierarchicalGCN_CGI(
            in_channels=dataset.num_features,
            hidden_channels=64,
            out_channels=64,
            num_classes=dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        best_epoch = 0
        times = []

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, train_loader)
            val_acc = test(model, test_loader)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc = val_acc  # 保存最佳精度
            times.append(time.time() - start)

            # 检查早停条件
            if epoch - best_epoch >= patience:
                print(f'Seed: {seed}, Fold: {fold+1}, Early stopping at epoch {epoch}')
                break

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    accuracies = [x[0] for x in fold_results]
    std_accuracy = statistics.stdev(accuracies)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存
    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Std accuracy = {std_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


mean_accuracies = [x[0] for x in results]
std_accuracies = [x[1] for x in results]
total_times = [x[2] for x in results]
avg_memories = [x[3] for x in results]

overall_mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)
overall_std_accuracy = statistics.stdev(mean_accuracies)
overall_total_time = sum(total_times) / len(total_times)
overall_avg_memory = sum(avg_memories) / len(avg_memories)

if torch.cuda.is_available():
    avg_gpu_memories = [x[3] for x in results]
    overall_avg_gpu_memory = sum(avg_gpu_memories) / len(avg_gpu_memories)
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB, Average GPU Memory = {overall_avg_gpu_memory:.2f} MB')
else:
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB')


# ## KMISPooling with HierarchicalGCN (2023)

# In[53]:


from typing import Callable, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch_scatter import scatter, scatter_add, scatter_min
from torch_sparse import SparseTensor, remove_diag

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.dense import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, Tensor

Scorer = Callable[[Tensor, Adj, OptTensor, OptTensor], Tensor]


import torch
from torch_scatter import scatter_max, scatter_min

from torch_geometric.typing import Adj, OptTensor, SparseTensor, Tensor


def maximal_independent_set(edge_index: Adj, k: int = 1,
                            perm: OptTensor = None) -> Tensor:
    r"""Returns a Maximal :math:`k`-Independent Set of a graph, i.e., a set of
    nodes (as a :class:`ByteTensor`) such that none of them are :math:`k`-hop
    neighbors, and any node in the graph has a :math:`k`-hop neighbor in the
    returned set.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method follows `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_ for :math:`k = 1`, and its
    generalization by `Bacciu et al. <https://arxiv.org/abs/2208.03523>`_ for
    higher values of :math:`k`.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: :class:`ByteTensor`
    """
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n = edge_index.size(0)
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n = edge_index.max().item() + 1

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()

    while not mask.all():
        for _ in range(k):
            min_neigh = torch.full_like(min_rank, fill_value=n)
            scatter_min(min_rank[row], col, out=min_neigh)
            torch.minimum(min_neigh, min_rank, out=min_rank)  # self-loops

        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()

        for _ in range(k):
            max_neigh = torch.full_like(mask, fill_value=0)
            scatter_max(mask[row], col, out=max_neigh)
            torch.maximum(max_neigh, mask, out=mask)  # self-loops

        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n

    return mis

def maximal_independent_set_cluster(edge_index: Adj, k: int = 1,
                                    perm: OptTensor = None) -> PairTensor:
    r"""Computes the Maximal :math:`k`-Independent Set (:math:`k`-MIS)
    clustering of a graph, as defined in `"Generalizing Downsampling from
    Regular Data to Graphs" <https://arxiv.org/abs/2208.03523>`_.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method returns both the :math:`k`-MIS and the clustering, where the
    :math:`c`-th cluster refers to the :math:`c`-th element of the
    :math:`k`-MIS.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    mis = maximal_independent_set(edge_index=edge_index, k=k, perm=perm)
    n, device = mis.size(0), mis.device

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index[0], edge_index[1]

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    min_rank = torch.full((n, ), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis

    for _ in range(k):
        min_neigh = torch.full_like(min_rank, fill_value=n)
        scatter_min(min_rank[row], col, out=min_neigh)
        torch.minimum(min_neigh, min_rank, out=min_rank)

    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)
    return mis, perm[clusters]


class KMISPooling(Module):

    _heuristics = {None, 'greedy', 'w-greedy'}
    _passthroughs = {None, 'before', 'after'}
    _scorers = {
        'linear',
        'random',
        'constant',
        'canonical',
        'first',
        'last',
    }

    def __init__(self, in_channels: Optional[int] = None, k: int = 1,
                 scorer: Union[Scorer, str] = 'linear',
                 score_heuristic: Optional[str] = 'greedy',
                 score_passthrough: Optional[str] = 'before',
                 aggr_x: Optional[Union[str, Aggregation]] = None,
                 aggr_edge: str = 'sum',
                 aggr_score: Callable[[Tensor, Tensor], Tensor] = torch.mul,
                 remove_self_loops: bool = True) -> None:
        super(KMISPooling, self).__init__()
        assert score_heuristic in self._heuristics,             "Unrecognized `score_heuristic` value."
        assert score_passthrough in self._passthroughs,             "Unrecognized `score_passthrough` value."

        if not callable(scorer):
            assert scorer in self._scorers,                 "Unrecognized `scorer` value."

        self.k = k
        self.scorer = scorer
        self.score_heuristic = score_heuristic
        self.score_passthrough = score_passthrough

        self.aggr_x = aggr_x
        self.aggr_edge = aggr_edge
        self.aggr_score = aggr_score
        self.remove_self_loops = remove_self_loops

        if scorer == 'linear':
            assert self.score_passthrough is not None,                 "`'score_passthrough'` must not be `None`"                 " when using `'linear'` scorer"

            self.lin = Linear(in_features=in_channels, out_features=1)

    def _apply_heuristic(self, x: Tensor, adj: SparseTensor) -> Tensor:
        if self.score_heuristic is None:
            return x

        row, col, _ = adj.coo()
        x = x.view(-1)

        if self.score_heuristic == 'greedy':
            k_sums = torch.ones_like(x)
        else:
            k_sums = x.clone()

        for _ in range(self.k):
            scatter_add(k_sums[row], col, out=k_sums)

        return x / k_sums

    def _scorer(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        if self.scorer == 'linear':
            return self.lin(x).sigmoid()

        if self.scorer == 'random':
            return torch.rand((x.size(0), 1), device=x.device)

        if self.scorer == 'constant':
            return torch.ones((x.size(0), 1), device=x.device)

        if self.scorer == 'canonical':
            return -torch.arange(x.size(0), device=x.device).view(-1, 1)

        if self.scorer == 'first':
            return x[..., [0]]

        if self.scorer == 'last':
            return x[..., [-1]]

        return self.scorer(x, edge_index, edge_attr, batch)


    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                batch: OptTensor = None) \
            -> Tuple[Tensor, Adj, OptTensor, OptTensor, Tensor, Tensor]:
        """"""
        edge_index = edge_index.long()
        adj, n = edge_index, x.size(0)

        if not isinstance(edge_index, SparseTensor):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, (n, n))

        score = self._scorer(x, edge_index, edge_attr, batch)
        updated_score = self._apply_heuristic(score, adj)
        perm = torch.argsort(updated_score.view(-1), 0, descending=True)

        mis, cluster = maximal_independent_set_cluster(adj, self.k, perm)

        row, col, val = adj.coo()
        c = mis.sum()

        if val is None:
            val = torch.ones_like(row, dtype=torch.float)

        adj = SparseTensor(row=cluster[row], col=cluster[col], value=val,
                           is_sorted=False,
                           sparse_sizes=(c, c)).coalesce(self.aggr_edge)

        if self.remove_self_loops:
            adj = remove_diag(adj)

        if self.score_passthrough == 'before':
            x = self.aggr_score(x, score)

        if self.aggr_x is None:
            x = x[mis]
        elif isinstance(self.aggr_x, str):
            x = scatter(x, cluster, dim=0, dim_size=mis.sum(),
                        reduce=self.aggr_x)
        else:
            x = self.aggr_x(x, cluster, dim_size=c)

        if self.score_passthrough == 'after':
            x = self.aggr_score(x, score[mis])

        if isinstance(edge_index, SparseTensor):
            edge_index, edge_attr = adj, None

        else:
            row, col, edge_attr = adj.coo()
            edge_index = torch.stack([row, col])

        if batch is not None:
            batch = batch[mis]


        return x, edge_index, edge_attr, batch, mis, cluster

    def __repr__(self):
        if self.scorer == 'linear':
            channels = f"in_channels={self.lin.in_channels}, "
        else:
            channels = ""

        return f'{self.__class__.__name__}({channels}k={self.k})'


# In[54]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, EdgePooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm

class HierarchicalGCN_KMIS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_KMIS, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)

        self.pool1 = KMISPooling(64, k=5, aggr_x='sum')
        self.pool2 = KMISPooling(64, k=5, aggr_x='sum')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN and pooling layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        # Second GCN and pooling layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.bn3(x)

        # Mean pooling over the nodes
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)

        # Fully connected layers
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


max_nodes = 100
data_path = '/data/zeyu/Pooling/IMDB-MULTI'  # Adjust path as needed
dataset = TUDataset(data_path, 'IMDB-MULTI', transform=T.Constant(), pre_filter=lambda data: data.num_nodes <= max_nodes,)
dataset = dataset.shuffle()
num_classes = dataset.num_classes
in_channels = dataset.num_features

def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 修改点：直接使用整个图的标签进行训练
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# In[55]:


import time

kf = KFold(n_splits=5)
results = []
patience = 150  # 早停的容忍度
min_delta = 0.00001  # 精度改进的最小变化

results = []
for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0  # 初始化总GPU内存使用

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = HierarchicalGCN_KMIS(
            in_channels=dataset.num_features,
            hidden_channels=64,
            out_channels=64,
            num_classes=dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        best_epoch = 0
        times = []

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, train_loader)
            val_acc = test(model, test_loader)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc = val_acc  # 保存最佳精度
            times.append(time.time() - start)

            # 检查早停条件
            if epoch - best_epoch >= patience:
                print(f'Seed: {seed}, Fold: {fold+1}, Early stopping at epoch {epoch}')
                break

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    accuracies = [x[0] for x in fold_results]
    std_accuracy = statistics.stdev(accuracies)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存
    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Std accuracy = {std_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


mean_accuracies = [x[0] for x in results]
std_accuracies = [x[1] for x in results]
total_times = [x[2] for x in results]
avg_memories = [x[3] for x in results]

overall_mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)
overall_std_accuracy = statistics.stdev(mean_accuracies)
overall_total_time = sum(total_times) / len(total_times)
overall_avg_memory = sum(avg_memories) / len(avg_memories)

if torch.cuda.is_available():
    avg_gpu_memories = [x[3] for x in results]
    overall_avg_gpu_memory = sum(avg_gpu_memories) / len(avg_gpu_memories)
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB, Average GPU Memory = {overall_avg_gpu_memory:.2f} MB')
else:
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB')


# ## GSAPooling with HierarchicalGCN (2021)

# In[56]:


from torch_geometric.nn import GCNConv
from torch.nn import Parameter
import torch

import torch.nn as nn
import numpy as np

from typing import Union, Optional, Callable
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import softmax

import math


import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GraphConv


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if isinstance(ratio, int):
            k = num_nodes.new_full((num_nodes.size(0), ), ratio)
            k = torch.min(k, num_nodes)
        else:
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class GSAPool(torch.nn.Module):

    def __init__(self, in_channels, pooling_ratio=0.5, alpha=0.6,
                        min_score=None, multiplier=1,
                        non_linearity=torch.tanh,
                        cus_drop_ratio =0):
        super(GSAPool,self).__init__()
        self.in_channels = in_channels

        self.ratio = pooling_ratio
        self.alpha = alpha

        self.sbtl_layer = GCNConv(in_channels,1)
        self.fbtl_layer = nn.Linear(in_channels, 1)
        self.fusion = GCNConv(in_channels,in_channels)

        self.min_score = min_score
        self.multiplier = multiplier
        self.fusion_flag = 0
        self.non_linearity = non_linearity

        self.dropout = torch.nn.Dropout(cus_drop_ratio)

    def conv_selection(self, conv, in_channels, conv_type=0):
        if(conv_type == 0):
            out_channels = 1
        elif(conv_type == 1):
            out_channels = in_channels
        if(conv == "GCNConv"):
            return GCNConv(in_channels,out_channels)
        elif(conv == "ChebConv"):
            return ChebConv(in_channels,out_channels,1)
        elif(conv == "SAGEConv"):
            return SAGEConv(in_channels,out_channels)
        elif(conv == "GATConv"):
            return GATConv(in_channels,out_channels, heads=1, concat=True)
        elif(conv == "GraphConv"):
            return GraphConv(in_channels,out_channels)
        else:
            raise ValueError

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        #SBTL
        score_s = self.sbtl_layer(x,edge_index).squeeze()
        #FBTL
        score_f = self.fbtl_layer(x).squeeze()
        #hyperparametr alpha
        score = score_s*self.alpha + score_f*(1-self.alpha)

        score = score.unsqueeze(-1) if score.dim()==0 else score

        if self.min_score is None:
            score = self.non_linearity(score)
        else:
            score = softmax(score, batch)

        sc = self.dropout(score)
        perm = topk(sc, self.ratio, batch)

        #fusion
        if(self.fusion_flag == 1):
            x = self.fusion(x, edge_index)
        x_ae = x[perm]
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, x_ae


# In[57]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, EdgePooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm

class HierarchicalGCN_GSA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_GSA, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)

        self.pool1 = GSAPool(64, pooling_ratio=0.5, alpha = 0.6, cus_drop_ratio = 0)
        self.pool2 = GSAPool(64, pooling_ratio=0.5, alpha = 0.6, cus_drop_ratio = 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN and pooling layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.bn1(x)
        x, edge_index, _, batch, perm_1, x_ae1 = self.pool1(x, edge_index, None, batch)

        # Second GCN and pooling layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.bn2(x)
        x, edge_index, _, batch, perm_1, x_ae1 = self.pool1(x, edge_index, None, batch)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.bn3(x)

        # Mean pooling over the nodes
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)

        # Fully connected layers
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


max_nodes = 100
data_path = '/data/zeyu/Pooling/IMDB-MULTI'  # Adjust path as needed
dataset = TUDataset(data_path, 'IMDB-MULTI', transform=T.Constant(), pre_filter=lambda data: data.num_nodes <= max_nodes,)
dataset = dataset.shuffle()

num_classes = dataset.num_classes
in_channels = dataset.num_features

def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 修改点：直接使用整个图的标签进行训练
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# In[58]:


kf = KFold(n_splits=5)
results = []
patience = 150  # 早停的容忍度
min_delta = 0.00001  # 精度改进的最小变化

results = []
for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0  # 初始化总GPU内存使用

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = HierarchicalGCN_GSA(
            in_channels=dataset.num_features,
            hidden_channels=64,
            out_channels=64,
            num_classes=dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        best_epoch = 0
        times = []

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, train_loader)
            val_acc = test(model, test_loader)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc = val_acc  # 保存最佳精度
            times.append(time.time() - start)

            # 检查早停条件
            if epoch - best_epoch >= patience:
                print(f'Seed: {seed}, Fold: {fold+1}, Early stopping at epoch {epoch}')
                break

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    accuracies = [x[0] for x in fold_results]
    std_accuracy = statistics.stdev(accuracies)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存
    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Std accuracy = {std_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


mean_accuracies = [x[0] for x in results]
std_accuracies = [x[1] for x in results]
total_times = [x[2] for x in results]
avg_memories = [x[3] for x in results]

overall_mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)
overall_std_accuracy = statistics.stdev(mean_accuracies)
overall_total_time = sum(total_times) / len(total_times)
overall_avg_memory = sum(avg_memories) / len(avg_memories)

if torch.cuda.is_available():
    avg_gpu_memories = [x[3] for x in results]
    overall_avg_gpu_memory = sum(avg_gpu_memories) / len(avg_gpu_memories)
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB, Average GPU Memory = {overall_avg_gpu_memory:.2f} MB')
else:
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB')


# ## HGPSLPooling with HierarchicalGCN (2019)

# In[59]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_scatter import scatter_add
from torch_sparse import spspmm, coalesce


import torch
import torch.nn as nn
from torch.autograd import Function
from torch_scatter import scatter_add, scatter_max

def topk(x, ratio, batch, min_score=None, tol=1e-7):

    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = torch.nonzero(x > scores_min).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
            num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm

def filter_adj(edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight

def scatter_sort(x, batch, fill_value=-1e16):
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,), fill_value)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    sorted_x, _ = dense_x.sort(dim=-1, descending=True)
    cumsum_sorted_x = sorted_x.cumsum(dim=-1)
    cumsum_sorted_x = cumsum_sorted_x.view(-1)

    sorted_x = sorted_x.view(-1)
    filled_index = sorted_x != fill_value

    sorted_x = sorted_x[filled_index]
    cumsum_sorted_x = cumsum_sorted_x[filled_index]

    return sorted_x, cumsum_sorted_x


def _make_ix_like(batch):
    num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0)
    idx = [torch.arange(1, i + 1, dtype=torch.long, device=batch.device) for i in num_nodes]
    idx = torch.cat(idx, dim=0)

    return idx


def _threshold_and_support(x, batch):
    """Sparsemax building block: compute the threshold
    Args:
        x: input tensor to apply the sparsemax
        batch: group indicators
    Returns:
        the threshold value
    """
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

    sorted_input, input_cumsum = scatter_sort(x, batch)
    input_cumsum = input_cumsum - 1.0
    rhos = _make_ix_like(batch).to(x.dtype)
    support = rhos * sorted_input > input_cumsum

    support_size = scatter_add(support.to(batch.dtype), batch)
    # mask invalid index, for example, if batch is not start from 0 or not continuous, it may result in negative index
    idx = support_size + cum_num_nodes - 1
    mask = idx < 0
    idx[mask] = 0
    tau = input_cumsum.gather(0, idx)
    tau /= support_size.to(x.dtype)

    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, x, batch):
        """sparsemax: normalizing sparse transform
        Parameters:
            ctx: context object
            x (Tensor): shape (N, )
            batch: group indicator
        Returns:
            output (Tensor): same shape as input
        """
        max_val, _ = scatter_max(x, batch)
        x -= max_val[batch]
        tau, supp_size = _threshold_and_support(x, batch)
        output = torch.clamp(x - tau[batch], min=0)
        ctx.save_for_backward(supp_size, output, batch)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output, batch = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = scatter_add(grad_input, batch) / supp_size.to(output.dtype)
        grad_input = torch.where(output != 0, grad_input - v_hat[batch], grad_input)

        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self):
        super(Sparsemax, self).__init__()

    def forward(self, x, batch):
        return sparsemax(x, batch)


if __name__ == '__main__':
    sparse_attention = Sparsemax()
    input_x = torch.tensor([1.7301, 0.6792, -1.0565, 1.6614, -0.3196, -0.7790, -0.3877, -0.4943, 0.1831, -0.0061])
    input_batch = torch.cat([torch.zeros(4, dtype=torch.long), torch.ones(6, dtype=torch.long)], dim=0)
    res = sparse_attention(input_x, input_batch)
    print(res)

class TwoHopNeighborhood(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        fill = 1e16
        value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, n, n, n, True)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op='min', fill_value=fill)
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, **kwargs):
        super(GCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class HGPSLPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2):
        super(HGPSLPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.negative_slop = negative_slop
        self.lamb = lamb

        self.att = Parameter(torch.Tensor(1, self.in_channels * 2))
        nn.init.xavier_uniform_(self.att.data)
        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = NodeInformationScore()

    def forward(self, x, edge_index, edge_attr, batch):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_information_score = self.calc_information_score(x, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)

        # Graph Pooling
        original_x = x
        perm = topk(score, self.ratio, batch)
        x = x[perm]
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        # Discard structure learning layer, directly return
        if self.sl is False:
            return x, induced_edge_index, induced_edge_attr, batch

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each pair of nodes is time consuming.
            # To accelerate this process, we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.
            k_hop = 3
            if edge_attr is None:
                edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float, device=edge_index.device)

            hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
            for _ in range(k_hop - 1):
                hop_data = self.neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            new_edge_index, new_edge_attr = filter_adj(hop_edge_index, hop_edge_attr, perm, num_nodes=score.size(0))

            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr * self.lamb
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            adj[row, col] = weights
            new_edge_index, weights = dense_to_sparse(adj)
            row, col = new_edge_index
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            del adj
            torch.cuda.empty_cache()
        else:
            # Learning the possible edge weights between each pair of nodes in the pooled subgraph, relative slower.
            if edge_attr is None:
                induced_edge_attr = torch.ones((induced_edge_index.size(1),), dtype=x.dtype,
                                               device=induced_edge_index.device)
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
            cum_num_nodes = num_nodes.cumsum(dim=0)
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            # Construct batch fully connected graph in block diagonal matirx format
            for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
                adj[idx_i:idx_j, idx_i:idx_j] = 1.0
            new_edge_index, _ = dense_to_sparse(adj)
            row, col = new_edge_index

            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            adj[row, col] = weights
            induced_row, induced_col = induced_edge_index

            adj[induced_row, induced_col] += induced_edge_attr * self.lamb
            weights = adj[row, col]
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, num_nodes=x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            del adj
            torch.cuda.empty_cache()

        return x, new_edge_index, new_edge_attr, batch


# In[60]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, EdgePooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm

class HierarchicalGCN_HGPSL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_HGPSL, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)

        self.pool1 = HGPSLPool(hidden_channels, ratio=0.5, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2)
        self.pool2 = HGPSLPool(hidden_channels, ratio=0.5, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr=None
        batch = data.batch

        # First GCN and pooling layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.bn1(x)
        x, edge_index, _, batch = self.pool1(x, edge_index, edge_attr, batch)

        # Second GCN and pooling layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.bn2(x)
        x, edge_index, _, batch = self.pool1(x, edge_index, edge_attr, batch)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.bn3(x)

        # Mean pooling over the nodes
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)

        # Fully connected layers
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


max_nodes = 100
data_path = '/data/zeyu/Pooling/IMDB-MULTI'  # Adjust path as needed
dataset = TUDataset(data_path, 'IMDB-MULTI', transform=T.Constant(), pre_filter=lambda data: data.num_nodes <= max_nodes,)

num_classes = dataset.num_classes
in_channels = dataset.num_features
dataset = dataset.shuffle()

def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 修改点：直接使用整个图的标签进行训练
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# In[35]:


import time

kf = KFold(n_splits=5)
results = []
patience = 150  # 早停的容忍度
min_delta = 0.00001  # 精度改进的最小变化

results = []
for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0  # 初始化总GPU内存使用

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = HierarchicalGCN_HGPSL(
            in_channels=dataset.num_features,
            hidden_channels=64,
            out_channels=64,
            num_classes=dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        best_epoch = 0
        times = []

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, train_loader)
            val_acc = test(model, test_loader)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc = val_acc  # 保存最佳精度
            times.append(time.time() - start)

            # 检查早停条件
            if epoch - best_epoch >= patience:
                print(f'Seed: {seed}, Fold: {fold+1}, Early stopping at epoch {epoch}')
                break

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    accuracies = [x[0] for x in fold_results]
    std_accuracy = statistics.stdev(accuracies)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存
    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Std accuracy = {std_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


mean_accuracies = [x[0] for x in results]
std_accuracies = [x[1] for x in results]
total_times = [x[2] for x in results]
avg_memories = [x[3] for x in results]

overall_mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)
overall_std_accuracy = statistics.stdev(mean_accuracies)
overall_total_time = sum(total_times) / len(total_times)
overall_avg_memory = sum(avg_memories) / len(avg_memories)

if torch.cuda.is_available():
    avg_gpu_memories = [x[3] for x in results]
    overall_avg_gpu_memory = sum(avg_gpu_memories) / len(avg_gpu_memories)
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB, Average GPU Memory = {overall_avg_gpu_memory:.2f} MB')
else:
    print(f'Overall: Mean accuracy = {overall_mean_accuracy:.4f}, Std accuracy = {overall_std_accuracy:.4f}, Total Time = {overall_total_time:.4f}s, Average Memory = {overall_avg_memory:.2f} MB')


# ## Dense AsymCheegerCutPooling with HierarchicalGCN (2023)

# In[69]:


import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset

# 数据集路径
data_path = '/data/zeyu/Pooling/IMDB-MULTI'

max_nodes =100
# 定义组合的transform操作
transform = T.Compose([T.Constant(), T.ToDense(max_nodes)])

# 创建数据集并应用transform
dataset_IMDB_MULTI = TUDataset(
    root=data_path,
    name='IMDB-MULTI',
    use_node_attr=True,
    transform=transform,
    pre_filter=lambda data: data.num_nodes <= max_nodes,
)

# 打印数据集信息
print(dataset_IMDB_MULTI)

dataset_IMDB_MULTI[1]


# In[62]:


from typing import List, Optional, Tuple, Union
import math
import torch
from torch import Tensor
from torch_geometric.nn.models.mlp import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import BatchNorm

class AsymCheegerCutPool(torch.nn.Module):
    r"""
    The asymmetric cheeger cut pooling layer from the `"Total Variation Graph Neural Networks"
    <https://arxiv.org/abs/2211.06218>`_ paper.

    Args:
        k (int):
            Number of clusters or output nodes
        mlp_channels (int, list of int):
            Number of hidden units for each hidden layer in the MLP used to
            compute cluster assignments. First integer must match the number
            of input channels.
        mlp_activation (any):
            Activation function between hidden layers of the MLP.
            Must be compatible with `torch_geometric.nn.resolver`.
        return_selection (bool):
            Whether to return selection matrix. Cannot not  be False
            if `return_pooled_graph` is False. (default: :obj:`False`)
        return_pooled_graph (bool):
            Whether to return pooled node features and adjacency.
            Cannot be False if `return_selection` is False. (default: :obj:`True`)
        bias (bool):
            whether to add a bias term to the MLP layers. (default: :obj:`True`)
        totvar_coeff (float):
            Coefficient for graph total variation loss component. (default: :obj:`1.0`)
        balance_coeff (float):
            Coefficient for asymmetric norm loss component. (default: :obj:`1.0`)
    """

    def __init__(self,
                 k: int,
                 mlp_channels: Union[int, List[int]],
                 mlp_activation="relu",
                 return_selection: bool = False,
                 return_pooled_graph: bool = True,
                 bias: bool = True,
                 totvar_coeff: float = 1.0,
                 balance_coeff: float = 1.0,
                 ):
        super().__init__()

        if not return_selection and not return_pooled_graph:
            raise ValueError("return_selection and return_pooled_graph can not both be False")

        if isinstance(mlp_channels, int):
            mlp_channels = [mlp_channels]

        act = activation_resolver(mlp_activation)
        in_channels = mlp_channels[0]
        self.mlp = torch.nn.Sequential()
        for channels in mlp_channels[1:]:
            self.mlp.append(Linear(in_channels, channels, bias=bias))
            in_channels = channels
            self.mlp.append(act)


        self.mlp.append(Linear(in_channels, k))
        self.k = k
        self.return_selection = return_selection
        self.return_pooled_graph = return_pooled_graph
        self.totvar_coeff = totvar_coeff
        self.balance_coeff = balance_coeff

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            x (Tensor):
                Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`
                with batch-size :math:`B`, (maximum) number of nodes :math:`N` for each graph,
                and feature dimension :math:`F`. Note that the cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
                being created within this method.
            adj (Tensor):
                Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (BoolTensor, optional):
                Mask matrix :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}`
                indicating the valid nodes for each graph. (default: :obj:`None`)

        :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
            :class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.mlp(x)
        s = torch.softmax(s, dim=-1)

        batch_size, n_nodes, _ = x.size()

        if mask is not None:
            mask = mask.view(batch_size, n_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        # Pooled features and adjacency
        if self.return_pooled_graph:
            x_pool = torch.matmul(s.transpose(1, 2), x)
            adj_pool = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # Total variation loss
        tv_loss = self.totvar_coeff*torch.mean(self.totvar_loss(adj, s))

        # Balance loss
        bal_loss = self.balance_coeff*torch.mean(self.balance_loss(s))

        if self.return_selection and self.return_pooled_graph:
            return s, x_pool, adj_pool, tv_loss, bal_loss
        elif self.return_selection and not self.return_pooled_graph:
            return s, tv_loss, bal_loss
        else:
            return x_pool, adj_pool, tv_loss, bal_loss

    def totvar_loss(self, adj, s):
        l1_norm = torch.sum(torch.abs(s[..., None, :] - s[:, None, ...]), dim=-1)

        loss = torch.sum(adj * l1_norm, dim=(-1, -2))

        # Normalize loss
        n_edges = torch.count_nonzero(adj, dim=(-1, -2))
        loss *= 1 / (2 * n_edges)

        return loss

    def balance_loss(self, s):
        n_nodes = s.size()[-2]

        # k-quantile
        idx = int(math.floor(n_nodes / self.k))
        quant = torch.sort(s, dim=-2, descending=True)[0][:, idx, :] # shape [B, K]

        # Asymmetric l1-norm
        loss = s - torch.unsqueeze(quant, dim=1)
        loss = (loss >= 0) * (self.k - 1) * loss + (loss < 0) * loss * -1
        loss = torch.sum(loss, dim=(-1, -2)) # shape [B]
        loss = 1 / (n_nodes * (self.k - 1)) * (n_nodes * (self.k - 1) - loss)

        return loss


# In[63]:



from torch_geometric.nn import SAGEConv

import os.path as osp
import time
from math import ceil

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
max_nodes = 1500
dataset = dataset_IMDB_MULTI
dataset = dataset.shuffle()
N = max(graph.num_nodes for graph in dataset)
mp_layers = 1
mp_channels = 64
mp_activation = "relu"
delta_coeff = 2.0

mlp_hidden_layers = 1
mlp_hidden_channels = 64
mlp_activation = "relu"
totvar_coeff = 0.5
balance_coeff = 0.5

epochs = 100
batch_size = 16
learning_rate = 5e-4
l2_reg_val = 0
patience = 10

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin:
            self.lin = torch.nn.Linear(out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class Net_AsymCheegerCut(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = 64
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = DenseSAGEConv(dataset.num_features, 64)

        num_nodes = 64
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = DenseSAGEConv(64, 64)

        self.gnn3_embed = DenseSAGEConv(64, 64)

        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, dataset.num_classes)

        self.pool1 = AsymCheegerCutPool(int(N//2),
                           mlp_channels=[mp_channels] +
                                [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=totvar_coeff,
                           balance_coeff=balance_coeff,
                           return_selection=False,
                           return_pooled_graph=True)
        self.pool2 = AsymCheegerCutPool(int(N//2),
                           mlp_channels=[mp_channels] +
                                [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=totvar_coeff,
                           balance_coeff=balance_coeff,
                           return_selection=False,
                           return_pooled_graph=True)


    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)

        x, adj, tv1, bal1 = self.pool1(x, adj, mask=None)
        #x = pool_output1.x_pool
        #adj = pool_output1.adj_pool

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)

        x, adj, tv1, bal1 = self.pool2(x, adj, mask=None)
        #x = pool_output1.x_pool
        #adj = pool_output1.adj_pool

        x = self.gnn3_embed(x, adj)
        x = F.relu(x)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = Net_AsymCheegerCut().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, optimizer, train_loader):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_loader.dataset)

@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.adj, data.mask)
        #print(output.shape)  # 检查输出的形状
        pred = output.max(dim=1)[1]  # 确保在正确的维度上进行 max 操作
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)


# In[64]:


import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import time
import psutil
import os
from sklearn.model_selection import KFold
from math import ceil


patience = 150  # 早停的容忍度
min_delta = 0.0001  # 精度改进的最小变化
kf = KFold(n_splits=5)
results = []

for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0 if torch.cuda.is_available() else None

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DenseDataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DenseDataLoader(test_dataset, batch_size=512, shuffle=False)

        model = Net_AsymCheegerCut().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        test_acc = 0
        times = []

        patience_counter = 0  # Initialize patience counter for early stopping

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, optimizer, train_loader)
            val_acc = test(model, test_loader)
            times.append(time.time() - start)

            # Early stopping logic
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                test_acc = val_acc
                patience_counter = 0  # Reset patience counter if improvement
            else:
                patience_counter += 1  # Increment patience counter if no improvement

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} for fold {fold+1}')
                break

            # Debugging print statement can be commented or uncommented
            # print(f'Seed: {seed}, Fold: {fold+1}, Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB, GPU Memory Usage: {gpu_memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存

    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


# ## Dense DifferencePooling with HierarchicalGCN (2018)

# In[70]:


import os.path as osp
import time
from math import ceil

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
max_nodes = 1500
dataset = dataset_IMDB_MULTI
dataset = dataset.shuffle()

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin:
            self.lin = torch.nn.Linear(out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class Net_Diff(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = 64
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = DenseSAGEConv(dataset.num_features, 64)

        num_nodes = 64
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = DenseSAGEConv(64, 64)

        self.gnn3_embed = DenseSAGEConv(64, 64)

        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, dataset.num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)
        x = F.relu(x)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = Net_Diff().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, optimizer, train_loader):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_loader.dataset)

@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)


# In[71]:


import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import time
import psutil
import os
from sklearn.model_selection import KFold
from math import ceil

patience = 150  # 早停的容忍度
min_delta = 0.0001  # 精度改进的最小变化
kf = KFold(n_splits=5)
results = []

for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0 if torch.cuda.is_available() else None

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DenseDataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DenseDataLoader(test_dataset, batch_size=512, shuffle=False)

        model = Net_Diff().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        test_acc = 0
        times = []

        patience_counter = 0  # Initialize patience counter for early stopping

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, optimizer, train_loader)
            val_acc = test(model, test_loader)
            times.append(time.time() - start)

            # Early stopping logic
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                test_acc = val_acc
                patience_counter = 0  # Reset patience counter if improvement
            else:
                patience_counter += 1  # Increment patience counter if no improvement

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} for fold {fold+1}')
                break

            # Debugging print statement can be commented or uncommented
            # print(f'Seed: {seed}, Fold: {fold+1}, Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB, GPU Memory Usage: {gpu_memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存

    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


# ## Dense Mincutpooling with HierarchicalGCN (2020)

# In[245]:


import os.path as osp
import time
from math import ceil

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_mincut_pool
max_nodes = 1500
dataset = dataset_IMDB_MULTI
dataset = dataset.shuffle()

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin:
            self.lin = torch.nn.Linear(out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())

        return x


class Net_mincut(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = 64
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = DenseSAGEConv(dataset.num_features, 64)

        num_nodes = 64
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = DenseSAGEConv(64, 64)

        self.gnn3_embed = DenseSAGEConv(64, 64)

        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, dataset.num_classes)


    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)

        x, adj, l1, e1 = dense_mincut_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)

        x, adj, l2, e2 = dense_mincut_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)
        x = F.relu(x)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = Net_mincut().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, optimizer, train_loader):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_loader.dataset)

@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)


# In[246]:


import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import time
import psutil
import os
from sklearn.model_selection import KFold
from math import ceil


patience = 150  # 早停的容忍度
min_delta = 0.0001  # 精度改进的最小变化
kf = KFold(n_splits=5)
results = []

for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0 if torch.cuda.is_available() else None

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DenseDataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DenseDataLoader(test_dataset, batch_size=512, shuffle=False)

        model = Net_mincut().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        test_acc = 0
        times = []

        patience_counter = 0  # Initialize patience counter for early stopping

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, optimizer, train_loader)
            val_acc = test(model, test_loader)
            times.append(time.time() - start)

            # Early stopping logic
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                test_acc = val_acc
                patience_counter = 0  # Reset patience counter if improvement
            else:
                patience_counter += 1  # Increment patience counter if no improvement

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} for fold {fold+1}')
                break

            # Debugging print statement can be commented or uncommented
            # print(f'Seed: {seed}, Fold: {fold+1}, Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB, GPU Memory Usage: {gpu_memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存

    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


# ## Dense DMoNPooling with HierarchicalGCN (2023)

# In[247]:


import os.path as osp
import time
from math import ceil

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, DMoNPooling
max_nodes = 1500
dataset = dataset_IMDB_MULTI
dataset = dataset.shuffle()

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin:
            self.lin = torch.nn.Linear(out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class Net_DMoN(torch.nn.Module):
    def __init__(self, in_channels=dataset.num_features, out_channels=64, hidden_channels=64):
        super().__init__()

        num_nodes = 64
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.pool1 = DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        num_nodes = 64
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.pool2 = DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.gnn1_embed = DenseSAGEConv(dataset.num_features, 64)
        self.gnn2_embed = DenseSAGEConv(64, 64)
        self.gnn3_embed = DenseSAGEConv(64, 64)

        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, dataset.num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)

        _, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)

        _, x, adj, sp2, o2, c2 = self.pool2(x, adj)

        x = self.gnn3_embed(x, adj)
        x = F.relu(x)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), sp1 + sp2 + o1 + o2 + c1 + c2

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = Net_DMoN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, optimizer, train_loader):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, = model(data.x, data.adj)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_loader.dataset)

@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj)[0].max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)


# In[248]:


import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import time
import psutil
import os
from sklearn.model_selection import KFold
from math import ceil


patience = 150  # 早停的容忍度
min_delta = 0.0001  # 精度改进的最小变化
kf = KFold(n_splits=5)
results = []

for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0 if torch.cuda.is_available() else None

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DenseDataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DenseDataLoader(test_dataset, batch_size=512, shuffle=False)

        model = Net_DMoN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        test_acc = 0
        times = []

        patience_counter = 0  # Initialize patience counter for early stopping

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, optimizer, train_loader)
            val_acc = test(model, test_loader)
            times.append(time.time() - start)

            # Early stopping logic
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                test_acc = val_acc
                patience_counter = 0  # Reset patience counter if improvement
            else:
                patience_counter += 1  # Increment patience counter if no improvement

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} for fold {fold+1}')
                break

            # Debugging print statement can be commented or uncommented
            # print(f'Seed: {seed}, Fold: {fold+1}, Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB, GPU Memory Usage: {gpu_memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存

    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


# ## Dense HoscPooling with HierarchicalGCN (2022)

# In[249]:


import torch

EPS = 1e-15


def dense_hoscpool(
    x,
    adj,
    s,
    mu=0.1,
    alpha=0.5,
    new_ortho=False,
    mask=None,
):
    r"""The highe-order pooling operator (HoscPool) from the paper
    `"Higher-order clustering and pooling for Graph Neural Networks"
    <http://arxiv.org/abs/2209.03473>`_. Based on motif spectral clustering,
    it captures and combines different levels of higher-order connectivity
    patterns when coarsening the graph.
    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}
        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})
    based on the learned cluster assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times K}`. This function returns the pooled feature matrix, the coarsened
    symmetrically normalised adjacency matrix, the motif spectral clustering loss :math:`\mathcal{L}_{mc}`
    and the orthogonality loss :math:`\mathcal{L}_{o}`.
    .. math::
        \mathcal{L}_{mc} &= - \frac{\alpha_1}{K} \cdot \text{Tr}\bigg(\frac{\mathbf{S}^\top \mathbf{A} \mathbf{S}}
            {\mathbf{S}^\top\mathbf{D}\mathbf{S}}\bigg) - \frac{\alpha_2}{K} \cdot \text{Tr}\bigg(
                \frac{\mathbf{S}^\top\mathbf{A}_{M}\mathbf{S}}{\mathbf{S}^\top\mathbf{D}_{M}\mathbf{S}}\bigg).
        \mathcal{L}_o &= \frac{1}{\sqrt{K}-1} \bigg( \sqrt{K} - \frac{1}{\sqrt{N}}\sum_{j=1}^K ||S_{*j}||_F\bigg)
    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): adjacency matrix :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): the learnable cluster assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times K}` with number of clusters :math:`K`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mu (Tensor, optional): scalar that controls the importance given to regularization loss
        alpha (Tensor, optional): scalar in [0,1] controlling the importance granted
            to higher-order information (in loss function).
        new_ortho (BoolTensor, optional): either to use new proposed loss or old one
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    # Output adjacency and feature matrices
    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Motif adj matrix - not sym. normalised
    motif_adj = torch.mul(torch.matmul(adj, adj), adj)
    motif_out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), motif_adj), s)

    mincut_loss = ho_mincut_loss = 0
    # 1st order MinCUT loss
    if alpha < 1:
        diag_SAS = torch.einsum("ijj->ij", out_adj.clone())
        d_flat = torch.einsum("ijk->ij", adj.clone())
        d = _rank3_diag(d_flat)
        sds = torch.matmul(torch.matmul(s.transpose(1, 2), d), s)
        diag_SDS = torch.einsum("ijk->ij", sds) + EPS
        mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        mincut_loss = 1 / k * torch.mean(mincut_loss)

    # Higher order cut
    if alpha > 0:
        diag_SAS = torch.einsum("ijj->ij", motif_out_adj)
        d_flat = torch.einsum("ijk->ij", motif_adj)
        d = _rank3_diag(d_flat)
        diag_SDS = (torch.einsum(
            "ijk->ij", torch.matmul(torch.matmul(s.transpose(1, 2), d), s)) +
                    EPS)
        ho_mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        ho_mincut_loss = 1 / k * torch.mean(ho_mincut_loss)

    # Combine ho and fo mincut loss.
    # We do not learn these coefficients yet
    hosc_loss = (1 - alpha) * mincut_loss + alpha * ho_mincut_loss

    # Orthogonality loss
    if mu == 0:
        ortho_loss = torch.tensor(0)
    else:
        if new_ortho:
            if s.shape[0] == 1:
                ortho_loss = ((-torch.sum(torch.norm(s, p="fro", dim=-2)) /
                               (num_nodes**0.5)) + k**0.5) / (k**0.5 - 1)
            elif mask != None:
                ortho_loss = sum([((-torch.sum(
                    torch.norm(
                        s[i][:mask[i].nonzero().shape[0]],
                        p="fro",
                        dim=-2,
                    )) / (mask[i].nonzero().shape[0]**0.5) + k**0.5) /
                                   (k**0.5 - 1)) for i in range(batch_size)
                                  ]) / float(batch_size)
            else:
                ortho_loss = sum(
                    [((-torch.sum(torch.norm(s[i], p="fro", dim=-2)) /
                       (num_nodes**0.5) + k**0.5) / (k**0.5 - 1))
                     for i in range(batch_size)]) / float(batch_size)
        else:
            # Orthogonality regularization.
            ss = torch.matmul(s.transpose(1, 2), s)
            i_s = torch.eye(k).type_as(ss)
            ortho_loss = torch.norm(
                ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
                i_s / torch.norm(i_s),
                dim=(-1, -2),
            )
            ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d + EPS)[:, None]
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, hosc_loss, mu * ortho_loss


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out


# In[250]:


import os.path as osp
import time
from math import ceil

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
max_nodes = 1500
dataset = dataset_IMDB_MULTI
dataset = dataset.shuffle()

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin:
            self.lin = torch.nn.Linear(out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class Net_Hosc(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = 64
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)

        num_nodes = 64
        self.gnn2_pool = GNN(64, 64, num_nodes)

        self.gnn1_embed = DenseSAGEConv(dataset.num_features, 64)
        self.gnn2_embed = DenseSAGEConv(64, 64)
        self.gnn3_embed = DenseSAGEConv(64, 64)

        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, dataset.num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)

        x, adj, mc, o = dense_hoscpool(x, adj, s, mu=0.1, alpha=0.5, new_ortho=False, mask=mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)

        x, adj, mc_aux, o_aux = dense_hoscpool(x, adj, s, mu=0.1, alpha=0.5, new_ortho=False)

        x = self.gnn3_embed(x, adj)
        x = F.relu(x)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        output = F.log_softmax(x, dim=-1)
        #print(f"Model output shape: {output.shape}")
        return output

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = Net_Hosc().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, optimizer, train_loader):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.adj, data.mask)  # Only unpack output
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        output = model(data.x, data.adj, data.mask)  # Model output
        pred = output.max(dim=1)[1]  # Find max along the correct dimension
        correct += int(pred.eq(data.y.view(-1)).sum())

    return correct / len(loader.dataset)


# In[251]:


import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import time
import psutil
import os
from sklearn.model_selection import KFold
from math import ceil


kf = KFold(n_splits=5)
results = []

patience = 150  # 早停的容忍度
min_delta = 0.0001  # 精度改进的最小变化
kf = KFold(n_splits=5)
results = []

for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0 if torch.cuda.is_available() else None

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DenseDataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DenseDataLoader(test_dataset, batch_size=512, shuffle=False)

        model = Net_Hosc().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        test_acc = 0
        times = []

        patience_counter = 0  # Initialize patience counter for early stopping

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, optimizer, train_loader)
            val_acc = test(model, test_loader)
            times.append(time.time() - start)

            # Early stopping logic
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                test_acc = val_acc
                patience_counter = 0  # Reset patience counter if improvement
            else:
                patience_counter += 1  # Increment patience counter if no improvement

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} for fold {fold+1}')
                break

            # Debugging print statement can be commented or uncommented
            # print(f'Seed: {seed}, Fold: {fold+1}, Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB, GPU Memory Usage: {gpu_memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存

    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


# ## Dense JustBalancePooling with HierarchicalGCN (2023)

# In[252]:


import torch

EPS = 1e-15


def just_balance_pool(x, adj, s, mask=None, normalize=True):
    r"""The Just Balance pooling operator from the `"Simplifying Clustering with
    Graph Neural Networks" <https://arxiv.org/abs/2207.08779>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened and symmetrically
    normalized adjacency matrix and the following auxiliary objective:

    .. math::
        \mathcal{L} = - {\mathrm{Tr}(\sqrt{\mathbf{S}^{\top} \mathbf{S}})}

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`
            with batch-size :math:`B`, (maximum) number of nodes :math:`N`
            for each graph, and feature dimension :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
            with number of clusters :math:`C`. The softmax does not have to be
            applied beforehand, since it is executed within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Loss
    ss = torch.matmul(s.transpose(1, 2), s)
    ss_sqrt = torch.sqrt(ss + EPS)
    loss = torch.mean(-_rank3_trace(ss_sqrt))
    if normalize:
        loss = loss / torch.sqrt(torch.tensor(num_nodes * k))

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, loss


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


# In[253]:


import os.path as osp
import time
from math import ceil

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
max_nodes = 1500
dataset = dataset_IMDB_MULTI
dataset = dataset.shuffle()

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin:
            self.lin = torch.nn.Linear(out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class Net_JustBalance(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = 64
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)

        num_nodes = 64
        self.gnn2_pool = GNN(64, 64, num_nodes)

        self.gnn1_embed = DenseSAGEConv(dataset.num_features, 64)
        self.gnn2_embed = DenseSAGEConv(64, 64)
        self.gnn3_embed = DenseSAGEConv(64, 64)

        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, dataset.num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)

        x, adj, b_loss = just_balance_pool(x, adj, s)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)

        x, adj, b_loss = just_balance_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)
        x = F.relu(x)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        output = F.log_softmax(x, dim=-1)
        #print(f"Model output shape: {output.shape}")
        return output

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = Net_JustBalance().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, optimizer, train_loader):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.adj, data.mask)  # Only unpack output
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        output = model(data.x, data.adj, data.mask)  # Model output
        pred = output.max(dim=1)[1]  # Find max along the correct dimension
        correct += int(pred.eq(data.y.view(-1)).sum())

    return correct / len(loader.dataset)


# In[254]:


import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import time
import psutil
import os
from sklearn.model_selection import KFold
from math import ceil


kf = KFold(n_splits=5)
results = []

for seed in range(1, 6):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_results = []
    total_time = 0
    total_memory_usage = 0
    total_gpu_memory_usage = 0 if torch.cuda.is_available() else None

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DenseDataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DenseDataLoader(test_dataset, batch_size=512, shuffle=False)

        model = Net_JustBalance().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        test_acc = 0
        times = []

        patience_counter = 0  # Initialize patience counter for early stopping

        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, optimizer, train_loader)
            val_acc = test(model, test_loader)
            times.append(time.time() - start)

            # Early stopping logic
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                test_acc = val_acc
                patience_counter = 0  # Reset patience counter if improvement
            else:
                patience_counter += 1  # Increment patience counter if no improvement

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} for fold {fold+1}')
                break

            # Debugging print statement can be commented or uncommented
            # print(f'Seed: {seed}, Fold: {fold+1}, Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        mean_time = torch.tensor(times).mean().item()
        total_time += sum(times)

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        total_memory_usage += memory_usage

        # Check if GPU is available and record GPU memory usage if it is
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            total_gpu_memory_usage += gpu_memory_usage
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB, GPU Memory Usage: {gpu_memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, gpu_memory_usage))
        else:
            print(f'Seed: {seed}, Fold: {fold+1}, Memory Usage: {memory_usage:.2f} MB')
            fold_results.append((best_val_acc, mean_time, memory_usage, None))

    mean_accuracy = sum([x[0] for x in fold_results]) / len(fold_results)
    tot_time = total_time  # 每个随机数种子5个fold的总时间
    avg_memory = total_memory_usage / 5  # 每个随机数种子5个fold的平均内存

    if torch.cuda.is_available():
        avg_gpu_memory = total_gpu_memory_usage / 5  # 每个随机数种子5个fold的平均显存
        results.append((mean_accuracy, tot_time, avg_memory, avg_gpu_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB, Average GPU Memory = {avg_gpu_memory:.2f} MB')
    else:
        results.append((mean_accuracy, tot_time, avg_memory))
        print(f'Seed {seed}: Mean accuracy = {mean_accuracy:.4f}, Total Time = {tot_time:.4f}s, Average Memory = {avg_memory:.2f} MB')


from datetime import datetime
import os
import h5py
import mat73
import scipy
import matplotlib.pyplot as plt
import numpy as np
import math, copy
import random
from tqdm import tqdm
import torch
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 0xD3
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
    torch.backends.cudnn.benchmark = False 
    device = "cuda:0" 
else:  
    device = "cpu"

def set_seed_device(seed=0xD3):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
        torch.backends.cudnn.benchmark = False 
        device = "cuda:0" 
    else:  
        device = "cpu"

    return device

def load_pde(name='SW', mode='Train'):
    if name == 'SW':
        if mode == 'Eval':
            TestArray = np.zeros((10, 101, 1, 128, 128))
            for i in range(10):
                trajectory = np.load('./datasets/mixed_data_test/SW/'+f'{i}'+'.npy')
                trajectory = trajectory[:,:1,:,:]
                TestArray[i]=trajectory
            TestTensor = torch.from_numpy(TestArray).float().to(device)

            print(TestTensor.shape)

            Trajectory_Length = TestTensor.shape[1]
            Test_Size = TestTensor.shape[0]

            print(Trajectory_Length, Test_Size)
            
            return TestTensor, Test_Size, Trajectory_Length
        TrainArray = np.zeros((810, 101, 1, 128, 128))
        for i in range(810):
            trajectory = np.load('./datasets/mixed_data_train/SW/'+f'{i}'+'.npy')
            trajectory = trajectory[:,:1,:,:]
            TrainArray[i]=trajectory

        TestArray = np.zeros((10, 101, 1, 128, 128))
        for i in range(10):
            trajectory = np.load('./datasets/mixed_data_test/SW/'+f'{i}'+'.npy')
            trajectory = trajectory[:,:1,:,:]
            TestArray[i]=trajectory

        print(TrainArray.shape)
        print(TestArray.shape)

        TrainTensor = torch.from_numpy(TrainArray).float().to(device)
        TestTensor = torch.from_numpy(TestArray).float().to(device)

        print(TrainTensor.shape, TestTensor.shape)

        Trajectory_Length = TrainTensor.shape[1]
        Train_Size = TrainTensor.shape[0]
        Test_Size = TestTensor.shape[0]

        print(Trajectory_Length, Test_Size)
        
        return TrainTensor, TestTensor, Train_Size, Test_Size, Trajectory_Length

    pass

def MOS(model, path=None, st=0, device=device, strict=True):
    if path:
        model.load_state_dict(torch.load(path), strict=strict)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) #5e-5
    def lr_lambda(step):
        return 1.0 / (1.0 + step * decay_rate)
    decay_rate = 1e-5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    for i in range(st):
        optimizer.step()
        scheduler.step()
    return model, optimizer, scheduler

def visualize_data_list(data_list,st=0):
    #data_list = np.array(data_list, dtype=np.float32)
    num_plots = st #len(data_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 1, 1))  # Adjust the figure size accordingly
    for i in range(num_plots):
        ax = axes[i] if num_plots > 1 else axes  # Handle single subplot case
        ax.imshow(data_list[i], cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('./plot.jpg', format='jpg', dpi=300)

if __name__=="__main__":
    Data = mat73.loadmat('./datasets/ns_V1e-3_N5000_T50.mat')
    Data_u = torch.from_numpy(Data['u'])
    Data_a = torch.from_numpy(Data['a'])
    Data = torch.cat([Data_a.unsqueeze(-1),Data_u],-1)
    Data = Data.permute(0, 3, 1, 2)
    print(Data.shape)

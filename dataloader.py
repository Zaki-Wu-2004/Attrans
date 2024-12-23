from torch.utils.data import DataLoader, Dataset
import numpy as np
import math, copy
import random
import torch
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PretrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone()
        mask = (torch.rand(x.shape) < 0.15).float()  #
        x[mask == 1] = 0  # 
        return x, mask, self.data[idx]  # 

class FinetuneDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @classmethod
    def split_dataset(cls, data, labels, test_ratio=0.3, random_seed=0xD3):

        data_np = data.numpy()
        labels_np = labels.numpy()

        train_data_np, test_data_np, train_labels_np, test_labels_np = train_test_split(
            data_np,
            labels_np,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=True
        )

        train_data = torch.tensor(train_data_np)
        test_data = torch.tensor(test_data_np)
        train_labels = torch.tensor(train_labels_np)
        test_labels = torch.tensor(test_labels_np)

        train_dataset = cls(train_data, train_labels)
        test_dataset = cls(test_data, test_labels)

        return train_dataset, test_dataset

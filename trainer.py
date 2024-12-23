from torch.utils.data import DataLoader, Dataset
import numpy as np
import math, copy
import random
import torch
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn_pretrain(batch):
    data, masks, targets = zip(*batch)
    data = torch.stack(data).to(device)
    masks = torch.stack(masks).to(device)
    targets = torch.stack(targets).to(device)
    return data, masks, targets

def collate_fn_finetune(batch):
    data, labels = zip(*batch)
    data = torch.stack(data).to(device)
    labels = torch.tensor(labels).to(device)
    return data, labels


def train_pretrain_epoch(model, dataloader, optimizer, criterion):
    batch_size = 16
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Pre-Training")  
    for data, masks, targets in progress_bar:
        optimizer.zero_grad()
        predictions = model(data, masks)
        #print(predictions.shape, data.shape, masks.shape)
        loss = criterion(predictions[masks == 1], targets[masks == 1])
        #wandb.log({"pretrain_loss":loss.item()})
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())  
    return total_loss / len(dataloader) / batch_size

def train_finetune_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Fine-Tuneing") 
    for data, labels in progress_bar:
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, labels)
        #wandb.log({"finetune_loss":loss.item()})
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())  #
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in dataloader:
            logits = model(data)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

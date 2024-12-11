import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import math, copy
import random

import torch
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import statistics
from utils import load_pde, set_seed_device, MOS
import argparse

from model import Attrans
from dataloader import PretrainDataset, FinetuneDataset
from trainer import (
    train_pretrain_epoch, 
    train_finetune_epoch, 
    evaluate, 
    collate_fn_pretrain, 
    collate_fn_finetune
)

#wandb.init(project="Att pretrain",name="default")
#Wconfig = wandb.config

device = set_seed_device(0xD3)
print(device)

class Config:
    model_name = "bert-base-uncased"  ##
    batch_size = 16
    num_classes = 3
    epochs_pretrain = 5
    epochs_finetune = 10
    learning_rate = 5e-5
    device = device
    hidden_size = 256
    channels = 4

config = Config()

pre_path = None #  
ft_path = "./saved_models/pretrain_2024-12-11_11:25/loss_1.0082.pth" # None


###########################################################################
path_predata = None
pretrain_data = None

path_ft = None
finetune_data = None
finetune_labels = None

if path_predata:
    pretrain_data = GetPre(path_predata)
else:
    num_samples_pretrain = 160
    pretrain_data = torch.randn(num_samples_pretrain, 4, 1000)

if path_ft:
    finetune_data, finetune_labels = GetFt(path_ft)
else:
    num_samples_finetune = 80
    finetune_data = torch.randn(num_samples_finetune, 4, 250)  
    finetune_labels = torch.randint(0, 3, (num_samples_finetune,)) 

if path_test:
    test_data, test_labels = GetTest(path_test)
else:
    num_samples_test = 16
    test_data = torch.randn(num_samples_test, 4, 250)  
    test_labels = torch.randint(0, 3, (num_samples_test,)) 
###########################################################################

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the model.")
    parser.add_argument("mode", type=str, choices=["pretrain", "finetune"], 
                        help="Mode to run the script: 'pretrain' or 'finetune'.")
    args = parser.parse_args()

    if args.mode == 'pretrain':

        pretrain_dataset = PretrainDataset(pretrain_data)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_pretrain)
        
        model = Attrans(config).to(config.device)
        if pre_path:
            model.load_state_dict(torch.load(pre_path))
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        pretrain_criterion = nn.MSELoss()
        finetune_criterion = nn.CrossEntropyLoss()

        current_time = datetime.now().strftime('%Y-%m-%d_%H:%M')
        os.makedirs(f'./saved_models/pretrain_{current_time}/', exist_ok=True)

        # 预训练
        best_loss = float('inf')
        print("Starting Pretraining...")
        for epoch in range(config.epochs_pretrain):
            pretrain_loss = train_pretrain_epoch(model, pretrain_loader, optimizer, pretrain_criterion)
            print(f"Epoch {epoch+1}/{config.epochs_pretrain}, Pretrain Loss: {pretrain_loss:.4f}")
            if best_loss > pretrain_loss:
                best_loss = pretrain_loss
                torch.save(model.state_dict(), f'./saved_models/pretrain_{current_time}/loss_{pretrain_loss:.4f}.pth')  # 保存模型
    
    elif args.mode == 'finetune':

        finetune_dataset = FinetuneDataset(finetune_data, finetune_labels)
        finetune_loader = DataLoader(finetune_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_finetune)

        test_dataset = FinetuneDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_finetune)

        model = Attrans(config).to(config.device)
        if ft_path:
            model.load_state_dict(torch.load(ft_path)) 
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        pretrain_criterion = nn.MSELoss()
        finetune_criterion = nn.CrossEntropyLoss()

        current_time = datetime.now().strftime('%Y-%m-%d_%H:%M')
        os.makedirs(f'./saved_models/finetune_{current_time}/', exist_ok=True)

        best_loss = float('inf')
        best_acc = 0
        print("Starting Fine-tuning...")
        for epoch in range(config.epochs_finetune):
            finetune_loss = train_finetune_epoch(model, finetune_loader, optimizer, finetune_criterion)
            print(f"Epoch {epoch+1}/{config.epochs_finetune}, Finetune Loss: {finetune_loss:.4f}")
            eval_loss, accuracy = evaluate(model, test_loader, finetune_criterion)
            if best_loss > eval_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), f'./saved_models/finetune_{current_time}/loss_{eval_loss:.4f}_acc_{accuracy:.4f}.pth')  # 保存模型
            elif best_acc < accuracy:
                best_acc = accuracy
                torch.save(model.state_dict(), f'./saved_models/finetune_{current_time}/loss_{eval_loss:.4f}_acc_{accuracy:.4f}.pth')  # 保存模型

        
        eval_loss, accuracy = evaluate(model, test_loader, finetune_criterion)
        print(f"Eval Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

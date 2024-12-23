import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
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
from process import assemble_data

wandb.init(project="Att pretrain",name="gpt2")
Wconfig = wandb.config

device = set_seed_device(0xD3)
print(device)

class Config:
    model_name = "gpt2"  ##roberta-base roberta-large
    batch_size = 32
    num_classes = 3
    epochs_pretrain = 50000
    epochs_finetune = 50000
    learning_rate = 5e-5
    device = device
    #hidden_size = 512
    channels = 4

config = Config()

pre_path = None #"./saved_models/pretrain_2024-12-23_02:41/loss_13.9233.pth"
ft_path = None #"./saved_models/pretrain_2024-12-23_02:41/loss_18.5481.pth" # None

###########################################################################

folder_path = './datasets/pretrain_data/' 
#pretrain_data, _ = assemble_data(folder_path, 'pretrain')
finetune_data, finetune_labels = assemble_data(folder_path, 'finetune', 'train')
test_data, test_labels = assemble_data(folder_path, 'finetune', 'test')
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
            wandb.log({"avg_pretrain_loss":pretrain_loss})
            print(f"Epoch {epoch+1}/{config.epochs_pretrain}, Pretrain Loss: {pretrain_loss:.4f}")
            if best_loss > pretrain_loss:
                best_loss = pretrain_loss
                torch.save(model.state_dict(), f'./saved_models/pretrain_{current_time}/loss_{pretrain_loss:.4f}.pth')  # 保存模型
    
    elif args.mode == 'finetune':
        #cconfig.batch_size = 1
        if test_data is not None and test_labels is not None:
            finetune_dataset = FinetuneDataset(finetune_data, finetune_labels)
            finetune_loader = DataLoader(finetune_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_finetune)
            test_dataset = FinetuneDataset(test_data, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_finetune)
        
        else:
            #print(finetune_dataset.shape, test_dataset.shape)
            finetune_dataset = FinetuneDataset(finetune_data, finetune_labels)
            finetune_loader = DataLoader(finetune_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_finetune)
            test_dataset = FinetuneDataset(test_data, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_finetune)

        model = Attrans(config).to(config.device)
        if ft_path:
            model.load_state_dict(torch.load(ft_path)) 
        #optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) #5e-5
        def lr_lambda(step):
            return 1.0 / (1.0 + step * decay_rate)
        decay_rate = 1e-5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        pretrain_criterion = nn.MSELoss()
        finetune_criterion = nn.CrossEntropyLoss()

        current_time = datetime.now().strftime('%Y-%m-%d_%H:%M')
        os.makedirs(f'./saved_models/finetune_{current_time}/', exist_ok=True)

        best_loss = float('inf')
        best_acc = 0
        print("Starting Fine-tuning...")
        for epoch in range(config.epochs_finetune):
            finetune_loss = train_finetune_epoch(model, finetune_loader, optimizer, scheduler, finetune_criterion)
            
            eval_loss, accuracy = evaluate(model, test_loader, finetune_criterion)
            wandb.log({"eval_loss":eval_loss})
            wandb.log({"accuracy":accuracy})
            print(f"Epoch {epoch+1}/{config.epochs_finetune}, Finetune Loss: {finetune_loss:.4f}, Eval Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")
            if best_loss > eval_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), f'./saved_models/finetune_{current_time}/acc_{accuracy:.4f}.pth')  # 保存模型
            elif best_acc < accuracy:
                best_acc = accuracy
                torch.save(model.state_dict(), f'./saved_models/finetune_{current_time}/acc_{accuracy:.4f}.pth')  # 保存模型

        
        eval_loss, accuracy = evaluate(model, test_loader, finetune_criterion)
        print(f"Eval Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import math, copy
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
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

ft_path = "./saved_models/finetune_2024-12-23_07:05/acc_0.9599.pth" # None

folder_path = './datasets/pretrain_data/' 
test_data, test_labels = assemble_data(folder_path, 'finetune', 'test')

if test_data is not None and test_labels is not None:    
    test_dataset = FinetuneDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_finetune)

model = Attrans(config).to(config.device)
if ft_path:
    model.load_state_dict(torch.load(ft_path)) 

finetune_criterion = nn.CrossEntropyLoss()


def evaluate(model, dataloader, criterion, class_names=None):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    # Ensure the model is on the correct device
    model.to(device)

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            logits = model(data)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())  # Collect predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())  # Collect true labels

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize confusion matrix (divide by row sums)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)  # Custom labels
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save confusion matrix as an image
    plt.savefig('normalized_confusion_matrix.png', dpi=300)
    plt.close()  # Close the plot to free up memory

    return avg_loss, accuracy

eval_loss, accuracy = evaluate(model, test_loader, finetune_criterion, ['Relaxed', 'Neutral', 'Concentrating'])
print(f"Eval Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")


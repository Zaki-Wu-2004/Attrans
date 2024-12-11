
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel
import os
import matplotlib.pyplot as plt
import numpy as np
import math, copy
import random
import torch
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from encoder import Encoder, Decoder

from utils import load_pde, set_seed_device, MOS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attrans(nn.Module):
    def __init__(self, config):
        super(Attrans, self).__init__()
        transformer_config = AutoConfig.from_pretrained(config.model_name)
        transformer_config.hidden_size = config.hidden_size
        transformer_config.num_attention_heads = 8
        transformer_config.num_hidden_layers = 4
        
        self.transformer = AutoModel.from_config(transformer_config)
        self.classifier = nn.Linear(transformer_config.hidden_size, config.num_classes)

        self.encoder = Encoder(config.channels, config.hidden_size)
        self.decoder = Decoder(config.hidden_size, config.channels)

    
    def forward(self, x, mask=None):

        x = self.encoder(x)

        batch_size, channels, width = x.shape
        x = x.view(batch_size, -1, width)
        transformer_output = self.transformer(inputs_embeds=x)
        hidden_states = transformer_output.last_hidden_state
        
        if mask is not None:
            predictions = self.decoder(hidden_states)
            return predictions
        else:
            logits = self.classifier(hidden_states[:, 0, :])
            return logits

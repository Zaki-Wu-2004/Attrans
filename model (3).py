
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
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm
from encoder import Encoder, Decoder, Classifier, Classifier1, Classifier2

from utils import load_pde, set_seed_device, MOS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    # Ensure embed_dim is even
    assert embed_dim % 2 == 0
    
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.view(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):  
    return torch.unsqueeze(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, torch.arange(length, dtype=torch.float32)
        ),
        0,
    )

t_emb_init = get_1d_sincos_pos_embed

class Attrans(nn.Module):
    def __init__(self, config):
        super(Attrans, self).__init__()
        transformer_config = AutoConfig.from_pretrained(config.model_name)
        
        self.transformer = AutoModel.from_config(transformer_config)
        self.classifier = Classifier2(transformer_config.hidden_size, config.num_classes)

        self.encoder = Encoder(config.channels, transformer_config.hidden_size)
        self.decoder = Decoder(transformer_config.hidden_size, config.channels)

        self.t_emb = nn.Parameter(t_emb_init(transformer_config.hidden_size, 31))

    
    def forward(self, x, mask=None):

        x = self.encoder(x)

        batch_size, channels, width = x.shape
        x = x.view(batch_size, -1, width)
        x = x + self.t_emb
        transformer_output = self.transformer(inputs_embeds=x)
        hidden_states = transformer_output.last_hidden_state

        #hidden_states = x + hidden_states
        
        if mask is not None:
            predictions = self.decoder(hidden_states)
            return predictions
        else:
            logits = self.classifier(hidden_states[:, :, :])
            return logits#, hidden_states



if __name__ == "__main__":
    
    x = torch.randn(1, 4, 250)
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
    model = Attrans(config)
    x, hid = model(x)
    print(x, hid.shape)



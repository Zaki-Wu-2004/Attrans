import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math, copy
import random
import torch
import torch, gc
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channel=4, embed_dim=256):
        super(Encoder, self).__init__()
        
        #self.conv1 = nn.Conv1d(in_channel, embed_dim//2, kernel_size=9, stride=4, padding=2)
        #self.relu = nn.GELU()
        #self.conv2 = nn.Conv1d(embed_dim//2, embed_dim, kernel_size=4, stride=2, padding=1)
        self.conv01 = nn.Conv1d(in_channel, in_channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.GELU()
        self.conv02 = nn.Conv1d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv03 = nn.Conv1d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, embed_dim//4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim//4, embed_dim//2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim//2, embed_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        #x = x + self.relu(self.conv01(x))
        #x = x + self.relu(self.conv02(x))
        #x = x + self.relu(self.conv03(x))
        x = self.conv(x)
        #x = self.conv1(x)
        #x = self.relu(x)
        #x = self.conv2(x)
        return x.transpose(-1,-2).contiguous()

class Decoder(nn.Module):
    def __init__(self, embed_dim=256, out_channel=4):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose1d(embed_dim, embed_dim//2, kernel_size=4, stride=2, padding=1)
        self.relu = nn.GELU()
        self.conv2 = nn.ConvTranspose1d(embed_dim//2, out_channel, kernel_size=9, stride=4, padding=2)

    def forward(self, x):
        x = x.transpose(-1, -2).contiguous()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, emb_dim=256, classes=3, Len=31):
        super(Classifier, self).__init__() #input:[bs, len, emb]
        self.emb_dim = emb_dim
        self.len = Len
        self.mlp1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, classes),
        )

    def forward(self, x):
        x = self.mlp1(x)
        return F.softmax(x, dim=-1)

class Classifier1(nn.Module):
    def __init__(self, emb_dim=256, classes=3, Len=31):
        super(Classifier1, self).__init__() #input:[bs, len, emb]
        self.emb_dim = emb_dim
        self.len = Len
        self.mlp1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(Len, Len//2),
            nn.ReLU(),
            nn.Linear(Len//2, classes),
        )

    def forward(self, x):
        x = x.reshape(-1, self.emb_dim)
        x = self.mlp1(x)
        x = x.reshape(-1, self.len)
        x = self.mlp2(x)
        return F.softmax(x, dim=-1)

class Classifier2(nn.Module):
    def __init__(self, emb_dim=256, classes=3, Len=31):
        super(Classifier2, self).__init__() #input:[bs, len, emb]
        self.emb_dim = emb_dim
        self.len = Len
        self.mlp1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, classes),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(Len, Len//2),
            nn.ReLU(),
            nn.Linear(Len//2, 1),
        )

    def forward(self, x):
        x = x.reshape(-1, self.len)
        x = self.mlp2(x)
        x = x.reshape(-1, self.emb_dim)
        x = self.mlp1(x)
        return F.softmax(x, dim=-1)


if __name__ == "__main__":
    x = torch.randn(16, 4, 250)

    model = Encoder(4, 256)

    x = model(x)

    print(x.shape)

    #model = Decoder(256, 4)

    #x = model(x)
    
    #print(x.shape)
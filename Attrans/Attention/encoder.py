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
        
        self.conv1 = nn.Conv1d(in_channel, embed_dim//2, kernel_size=9, stride=5, padding=2)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv1d(embed_dim//2, embed_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x.transpose(-1,-2).contiguous()

class Decoder(nn.Module):
    def __init__(self, embed_dim=256, out_channel=4):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose1d(embed_dim, embed_dim//2, kernel_size=4, stride=2, padding=1)
        self.relu = nn.GELU()
        self.conv2 = nn.ConvTranspose1d(embed_dim//2, out_channel, kernel_size=9, stride=5, padding=2)

    def forward(self, x):
        x = x.transpose(-1, -2).contiguous()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


if __name__ == "__main__":
    x = torch.randn(16, 4, 1000)

    model = Encoder(4, 256)

    x = model(x)

    print(x.shape)

    model = Decoder(256, 4)

    x = model(x)
    
    print(x.shape)
import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import torch 
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, token_dim, num_classes):
        super().__init__()
        self.layernorm = nn.LayerNorm(token_dim)
        self.mlp = nn.Linear(token_dim, num_classes)
    
    def forward(self, x):
        x = self.layernorm(x)
        x = self.mlp(x)
        return x
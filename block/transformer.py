import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import torch 
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, token_dim, num_heads, mlp_hidden_dim):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(token_dim)
        self.layernorm_2 = nn.LayerNorm(token_dim)
        self.multihead_attention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, token_dim),
        )

    def forward(self, x):
        residual_1 = x
        x = self.layernorm_1(x)
        x = self.multihead_attention(x, x, x)[0]
        x = residual_1 + x
        
        residual_2 = x
        x = self.layernorm_2(x)
        x = self.mlp(x)
        x = residual_2 + x
        
        return x
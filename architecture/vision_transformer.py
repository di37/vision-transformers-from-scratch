import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from block import PatchEmbedding, TransformerEncoder, MLPHead
import torch 
import torch.nn as nn

import json

class VisionTransformer(nn.Module):
    def __init__(self, img_size, num_channels, patch_size, token_dim, num_heads, transformer_blocks, num_classes, mlp_hidden_dim):
        super().__init__()
        
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embedding = PatchEmbedding(num_channels, token_dim, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, token_dim))
        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoder(token_dim, num_heads, mlp_hidden_dim) for _ in range(transformer_blocks)]
        )
        self.mlp_head = MLPHead(token_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.positional_embedding
        
        x = self.transformer_blocks(x)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x
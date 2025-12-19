import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, num_channels, token_dim, patch_size):
        super().__init__()
        self.projection = nn.Conv2d(num_channels, token_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
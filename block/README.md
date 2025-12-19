# Blocks

This directory contains the building blocks of the Vision Transformer.

## Files

### [patch_embedding.py](patch_embedding.py)
- **PatchEmbedding**: Converts input images into flattened patches and projects them into the token dimension using a Convolutional layer.

### [transformer.py](transformer.py)
- **TransformerEncoder**: A single layer of the Transformer, consisting of:
    - Multi-Head Self Attention (MSA)
    - LayerNorm
    - MLP (Feed Forward Network) with GELU activation
    - Residual connections

### [mlp_head.py](mlp_head.py)
- **MLPHead**: The final classification head that takes the CLS token output and projects it to the number of classes.

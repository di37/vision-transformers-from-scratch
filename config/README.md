# Configuration

This directory contains JSON configuration files to control the model architecture and training process.

## Files

### [model_config.json](model_config.json)
Defines the structure of the Vision Transformer.
- `img_size`: Input image size (height/width).
- `patch_size`: Size of each patch.
- `token_dim`: Embedding dimension size.
- `num_heads`: Number of attention heads.
- `transformer_blocks`: Number of transformer layers.
- `num_classes`: Number of output classes (e.g., 100 for CIFAR-100).
- `mlp_hidden_dim`: Hidden dimension size in the MLP blocks.

### [training_config.json](training_config.json)
Defines training hyperparameters.
- `epochs`: Number of training epochs.
- `batch_size`: Batch size for data loaders.
- `learning_rate`: Initial learning rate for the optimizer.

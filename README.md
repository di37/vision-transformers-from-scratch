# Vision Transformer (ViT) Implementation from Scratch

This project implements a Vision Transformer (ViT) model from scratch using PyTorch, capable of training on datasets like CIFAR-100.

## Project Structure

```
.
├── architecture/
│   └── vision_transformer.py  # Main ViT model assembly
├── block/
│   ├── patch_embedding.py     # Patch embedding layer
│   ├── transformer.py         # Transformer Encoder block
│   └── mlp_head.py            # MLP classification head
├── config/
│   ├── model_config.json      # Model architecture hyperparameters
│   └── training_config.json   # Training hyperparameters (LR, epochs, etc.)
├── data/                      # Dataset storage
├── pre_training_evaluation.ipynb # Notebook for training and evaluation
```

## Features

- **Modular Design**: Core components (`PatchEmbedding`, `TransformerEncoder`, `MLPHead`) are separated for clarity and reusability.
- **Configurable**: Model dimensions and training parameters are fully configurable via JSON files in the `config/` directory.
- **Data Augmentation**: Includes robust data augmentation (RandomCrop, HorizontalFlip) and normalization for better generalization.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy

### Installation

Clone the repository and install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

### Configuration

Modify `config/model_config.json` to change model size:
```json
{
    "img_size": 32,
    "patch_size": 4,
    "token_dim": 256,
    "transformer_blocks": 6,
    ...
}
```

Modify `config/training_config.json` for training parameters:
```json
{
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 3e-4
}
```

### Training

You can run the Jupyter Notebook `pre_training_evaluation.ipynb` for an interactive experience.

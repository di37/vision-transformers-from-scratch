# Architecture

This directory contains the high-level architecture of the Vision Transformer.

## Files

### [vision_transformer.py](vision_transformer.py)
Defines the `VisionTransformer` class, which assembles the model.
- **Initialization**: Loads configuration from `../config/model_config.json`.
- **Components**: Combines `PatchEmbedding`, `TransformerEncoder` blocks, and `MLPHead`.
- **Forward Pass**: Handles patch embedding, class token concatenation, position embedding addition, transformer encoding, and final classification.

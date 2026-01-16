# Models Directory - Student Implementation Guide

This directory is where you implement your face transformation models.

## Directory Structure (Recommended)

```
src/models/
├── __init__.py
├── README.md              # This file
├── encoders/              # GAN inversion encoders
│   ├── __init__.py
│   ├── psp.py             # Pixel2Style2Pixel encoder
│   └── e4e.py             # Encoder4Editing
├── generators/            # Image generators
│   ├── __init__.py
│   ├── stylegan2.py       # StyleGAN2 generator
│   └── stylegan3.py       # StyleGAN3 generator
├── discriminators/        # Discriminator networks
│   ├── __init__.py
│   └── stylegan2.py
├── losses/                # Loss functions
│   ├── __init__.py
│   ├── adversarial.py     # GAN losses
│   ├── perceptual.py      # LPIPS, VGG losses
│   ├── identity.py        # ArcFace identity loss
│   └── attribute.py       # Age/gender classification loss
├── directions/            # Latent direction finding
│   ├── __init__.py
│   ├── interfacegan.py    # SVM-based directions
│   └── ganspace.py        # PCA-based directions
└── lightning_modules/     # PyTorch Lightning modules
    ├── __init__.py
    └── transformer.py     # Main training module
```

## Quick Start

### 1. Create a Lightning Module

```python
# src/models/lightning_modules/transformer.py
import lightning as L
import torch
import torch.nn as nn

class FaceTransformer(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        generator: nn.Module,
        learning_rate: float = 1e-4,
        attribute_directions: dict[str, torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "generator"])
        
        self.encoder = encoder
        self.generator = generator
        self.directions = attribute_directions or {}
    
    def forward(self, x, target_attrs: dict[str, float] = None):
        # 1. Encode image to latent code
        latent = self.encoder(x)
        
        # 2. Apply attribute transformations
        if target_attrs:
            for attr, strength in target_attrs.items():
                if attr in self.directions:
                    latent = latent + strength * self.directions[attr]
        
        # 3. Decode to image
        output = self.generator(latent)
        return output, latent
    
    def training_step(self, batch, batch_idx):
        if "source_image" in batch:
            # Paired training
            source = batch["source_image"]
            target = batch["target_image"]
            # ... compute losses
        else:
            # Unpaired training
            image = batch["image"]
            attrs = batch["attributes"]
            # ... compute losses
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
```

### 2. Use with Training Script

Modify `scripts/train.py`:

```python
from src.models.lightning_modules.transformer import FaceTransformer
from src.models.encoders.e4e import E4EEncoder
from src.models.generators.stylegan2 import StyleGAN2Generator

# In main():
encoder = E4EEncoder(...)
generator = StyleGAN2Generator(...)
model = FaceTransformer(encoder, generator)

trainer = L.Trainer(
    max_epochs=cfg.training.max_epochs,
    accelerator=cfg.hardware.accelerator,
    devices=cfg.hardware.devices,
)
trainer.fit(model, datamodule)
```

## Key Components to Implement

### Encoders (GAN Inversion)

| Encoder | Paper | Key Feature |
|---------|-------|-------------|
| pSp | pixel2style2pixel | Direct W+ prediction |
| e4e | encoder4editing | Better editability |
| ReStyle | ReStyle | Iterative refinement |

### Loss Functions

| Loss | Purpose | Weight (typical) |
|------|---------|------------------|
| L2/L1 | Pixel reconstruction | 1.0 |
| LPIPS | Perceptual similarity | 0.8 |
| ID Loss | Identity preservation (ArcFace) | 0.1 |
| Adversarial | Realism | 0.1 |
| Attribute | Age/gender accuracy | 0.5 |

### Latent Directions

Methods to find attribute directions in W/W+ space:

1. **InterFaceGAN**: Train SVM on latent codes labeled with attributes
2. **GANSpace**: PCA on intermediate activations
3. **StyleCLIP**: CLIP-guided direction finding

## Configuration Integration

Your model config goes in `configs/model/`:

```yaml
# configs/model/stylegan_e4e.yaml
_target_: src.models.lightning_modules.transformer.FaceTransformer

encoder:
  _target_: src.models.encoders.e4e.E4EEncoder
  input_size: 256
  latent_dim: 512

generator:
  _target_: src.models.generators.stylegan2.StyleGAN2Generator
  output_size: 256
  latent_dim: 512
  pretrained: ${paths.pretrained_dir}/stylegan2-ffhq-256.pt

learning_rate: 1e-4
```

## Pretrained Weights

Download pretrained models to `pretrained/`:

- StyleGAN2 FFHQ: [NVIDIA](https://github.com/NVlabs/stylegan2-ada-pytorch)
- ArcFace: [InsightFace](https://github.com/deepinsight/insightface)
- LPIPS: Auto-downloaded by `lpips` package

## Tips

1. **Start simple**: Get basic reconstruction working first
2. **Use pretrained generator**: Freeze StyleGAN weights initially
3. **Progressive training**: Train encoder first, then fine-tune together
4. **Monitor identity**: Track ArcFace cosine similarity
5. **Visualize frequently**: Log image grids to W&B

## Example Training Command

```bash
python scripts/train.py \
    experiment=multi_attr \
    data=celeba_hq \
    model=stylegan_e4e \
    training.max_epochs=50
```

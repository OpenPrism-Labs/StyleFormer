# StyleFormer on Kaggle

This guide explains how to use StyleFormer for face transformation on Kaggle.

## Quick Start

### Option 1: Use Pre-made Notebooks

1. **Add StyleFormer as a Dataset**
   - Go to your Kaggle notebook
   - Click "Add Data" → "Your Datasets" or upload this repo as a dataset
   - Or use: `kaggle datasets create -p /path/to/StyleFormer`

2. **Open the Notebooks**
   - `notebooks/kaggle_training.ipynb` - Train models
   - `notebooks/kaggle_inference.ipynb` - Run inference

### Option 2: Manual Setup

```python
# Cell 1: Install dependencies
!pip install -q lpips einops pytorch-lightning hydra-core omegaconf

# Cell 2: Setup paths
import sys
sys.path.insert(0, '/kaggle/input/styleformer')

from kaggle.setup_kaggle import setup, KaggleConfig
config = KaggleConfig()
setup()

# Cell 3: Import and use
from src.data import FaceDataModule
from src.models.losses import L1Loss, PerceptualLoss
from src.evaluation import Evaluator
```

## Directory Structure

```
kaggle/
├── __init__.py              # Module exports
├── setup_kaggle.py          # Environment setup utilities
├── dataset-metadata.json    # For Kaggle dataset upload
└── README.md                # This file

notebooks/
├── kaggle_training.ipynb    # Training notebook
└── kaggle_inference.ipynb   # Inference notebook
```

## Recommended Datasets

Add these Kaggle datasets for face images:

| Dataset | Kaggle Path | Resolution |
|---------|-------------|------------|
| CelebA | `jessicali9530/celeba-dataset` | 178×218 |
| CelebA-HQ | `lamsimon/celebahq` | 256×256+ |
| FFHQ | `arnaud58/flickrfaceshq-dataset-ffhq` | 1024×1024 |

## Hardware Requirements

| Task | Recommended | Minimum |
|------|-------------|---------|
| Training | GPU T4 x2 or P100 | GPU T4 |
| Inference | GPU T4 | CPU (slow) |
| Batch Size | 8-16 (T4), 16-32 (P100) | 4 |

**Enable GPU:** Settings → Accelerator → GPU

## Environment Setup API

```python
from kaggle.setup_kaggle import (
    setup,              # Full environment setup
    KaggleConfig,       # Configuration helper
    is_kaggle,          # Check if running on Kaggle
    get_dataset_paths,  # Find mounted datasets
)

# Full setup
result = setup(
    install_packages=True,   # Install missing packages
    create_dirs=True,        # Create output directories
    verbose=True,            # Print status
)

# Configuration helper
config = KaggleConfig()
config.output_dir        # /kaggle/working/outputs
config.checkpoint_dir    # /kaggle/working/checkpoints
config.get_device()      # 'cuda' or 'cpu'
config.get_accelerator() # 'gpu' or 'cpu' (for PyTorch Lightning)
```

## Training on Kaggle

### Basic Training

```python
import pytorch_lightning as pl
from src.data import FaceDataModule
from src.models.lightning_modules.base import BaseTransformerModule

# Data
datamodule = FaceDataModule(
    name='celeba_hq',
    root='/kaggle/input/celebahq-resized-256x256',
    image_size=256,
    batch_size=8,
)

# Model (define your own or use provided)
model = YourTransformerModel(
    learning_rate=1e-4,
    w_dim=512,
    num_ws=14,  # for 256x256
)

# Train
trainer = pl.Trainer(
    max_epochs=50,
    accelerator='gpu',
    devices=1,
    precision='16-mixed',  # Faster training
)
trainer.fit(model, datamodule)
```

### Save Outputs

```python
# Save to Kaggle output
trainer.save_checkpoint('/kaggle/working/model.ckpt')

# Save as downloadable artifact
import shutil
shutil.copy('/kaggle/working/model.ckpt', '/kaggle/working/outputs/')
```

## Inference on Kaggle

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = YourModel.load_from_checkpoint('/kaggle/input/your-model/model.ckpt')
model.eval()
model.cuda()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Run inference
image = Image.open('/kaggle/input/test-images/face.jpg')
input_tensor = transform(image).unsqueeze(0).cuda()

with torch.no_grad():
    latent = model.encode(input_tensor)
    output = model.decode(latent)

# Save result
output_image = (output[0].cpu() + 1) / 2  # [-1,1] → [0,1]
transforms.ToPILImage()(output_image).save('/kaggle/working/result.png')
```

## Uploading StyleFormer as a Kaggle Dataset

### Using Kaggle CLI

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API (get token from kaggle.com/account)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Create dataset
cd /path/to/StyleFormer
kaggle datasets create -p . --dir-mode zip
```

### Using Web Interface

1. Go to kaggle.com/datasets
2. Click "New Dataset"
3. Upload the StyleFormer directory as a zip
4. Use metadata from `kaggle/dataset-metadata.json`

## Troubleshooting

### Import Errors

```python
# Make sure path is set correctly
import sys
print(sys.path)

# Verify StyleFormer location
!ls /kaggle/input/styleformer/src
```

### Out of Memory

```python
# Reduce batch size
batch_size = 4

# Use gradient accumulation
trainer = pl.Trainer(
    accumulate_grad_batches=4,  # Effective batch = 4 * 4 = 16
    ...
)

# Clear cache periodically
torch.cuda.empty_cache()
```

### Slow Training

```python
# Enable mixed precision
trainer = pl.Trainer(precision='16-mixed', ...)

# Use more workers (but not too many on Kaggle)
datamodule = FaceDataModule(num_workers=2, ...)

# Disable validation during training
trainer = pl.Trainer(limit_val_batches=0, ...)
```

## File Outputs

All outputs are saved to `/kaggle/working/`:

```
/kaggle/working/
├── outputs/           # Generated images, logs
├── checkpoints/       # Model checkpoints
├── generated/         # Inference outputs
└── logs/              # Training logs
```

Download from the "Output" tab after notebook completes.

## Example Notebooks

| Notebook | Description | GPU Required |
|----------|-------------|--------------|
| `kaggle_training.ipynb` | Train encoder-decoder model | Yes |
| `kaggle_inference.ipynb` | Run face transformation | Recommended |

## License

MIT License - See main repository for details.

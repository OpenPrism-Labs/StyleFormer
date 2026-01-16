# StyleFormer

**Face Transformation with Multi-Attribute Style Transfer**

A modular experimental framework for face transformation research, supporting simultaneous manipulation of multiple facial attributes (age, gender, etc.) using StyleGAN latent space editing.

---

## Features

- **Multi-Attribute Transformation**: Manipulate age and gender simultaneously with disentanglement constraints
- **Flexible Data Pipeline**: PyTorch Lightning DataModule with CelebA-HQ (labeled) and FFHQ (unlabeled) support
- **Paired Dataset Mode**: Automatic pairing of images with opposite attributes for style transfer training
- **Hydra Configuration**: Composable YAML configs for reproducible experiments
- **Modular Architecture**: Clean separation of infrastructure (provided) and models (student-implemented)

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Conda (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/OpenPrism-Labs/StyleFormer.git
cd StyleFormer

# Create conda environment
conda env create -f environment.yaml
conda activate styleformer

# Install package in development mode
pip install -e .

# (Optional) Install dev dependencies
pip install -e ".[dev]"
```

---

## Project Structure

```
StyleFormer/
├── configs/                      # Hydra configuration files
│   ├── base.yaml                 # Shared defaults
│   ├── attributes/default.yaml   # Multi-attribute definitions
│   ├── data/
│   │   ├── celeba_hq.yaml       # CelebA-HQ dataset config
│   │   └── ffhq.yaml            # FFHQ dataset config
│   ├── experiments/
│   │   └── multi_attr.yaml      # Multi-attribute experiment
│   └── paths/default.yaml       # Dataset and output paths
│
├── src/                          # Source code
│   ├── data/                     # Data loading infrastructure
│   │   ├── datasets/            # Dataset implementations
│   │   │   ├── base.py          # Abstract base classes
│   │   │   ├── celeba_hq.py     # CelebA-HQ with attributes
│   │   │   └── ffhq.py          # FFHQ dataset
│   │   ├── datamodule.py        # PyTorch Lightning DataModule
│   │   └── transforms.py        # Image augmentation pipelines
│   ├── config/schema.py         # Configuration dataclasses
│   ├── models/                  # Model implementations (TODO)
│   │   └── README.md            # Implementation guide
│   └── utils/                   # Utilities
│       ├── io.py                # Checkpoint I/O
│       ├── logging.py           # W&B and console logging
│       └── visualization.py     # Image grid generation
│
├── scripts/                      # Entry points
│   ├── train.py                 # Training script
│   └── prepare_data.py          # Dataset preparation
│
├── data/                         # Datasets (gitignored)
│   ├── celeba_hq/
│   ├── ffhq/
│   └── directions/              # Learned attribute directions
│
├── pretrained/                   # Pretrained weights (gitignored)
├── outputs/                      # Experiment outputs (gitignored)
└── notebooks/                    # Jupyter notebooks
```

---

## Quick Start

### 1. Prepare Datasets

```bash
# Show download instructions
python scripts/prepare_data.py --dataset celeba_hq

# Verify dataset after download
python scripts/prepare_data.py --verify
```

### 2. Test Data Loading

```bash
# Run with default config (verifies data pipeline works)
python scripts/train.py
```

### 3. Run Multi-Attribute Experiment

```bash
# Use the multi-attribute experiment config
python scripts/train.py experiment=multi_attr
```

---

## Configuration

StyleFormer uses [Hydra](https://hydra.cc/) for configuration management.

### Override Config via CLI

```bash
# Change batch size and learning rate
python scripts/train.py dataloader.batch_size=32 optimizer.lr=0.0002

# Use FFHQ dataset instead of CelebA-HQ
python scripts/train.py data=ffhq

# Enable paired training mode
python scripts/train.py data.pairing.enabled=true data.pairing.transfer_attr=Male
```

### Multi-Attribute Configuration

Edit `configs/attributes/default.yaml`:

```yaml
# Active attributes for transformation
active: ["age", "gender"]

# Attribute definitions
definitions:
  age:
    type: continuous
    range: [-3.0, 3.0]
    labels: ["young", "old"]
    celeba_attr: "Young"
    
  gender:
    type: continuous
    range: [-3.0, 3.0]
    labels: ["female", "male"]
    celeba_attr: "Male"

# Transformation settings
transform:
  mode: simultaneous          # Apply both at once
  null_space_projection: true # Prevent attribute leakage
  targets:
    age: 2.5                  # Strength towards "old"
    gender: -2.0              # Strength towards "female"
```

---

## Datasets

### CelebA-HQ

High-quality celebrity faces with 40 attribute labels.

**Required structure:**
```
data/celeba_hq/
├── images/                    # or img_align_celeba/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── list_attr_celeba.txt      # Attribute labels
└── list_eval_partition.txt   # Train/val/test splits (optional)
```

**Download options:**
- [Official CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)
- [Kaggle](https://www.kaggle.com/datasets/lamsimon/celebahq)

### FFHQ

70,000 high-quality face images without attribute labels.

**Required structure:**
```
data/ffhq/
├── 00000/
│   ├── 00000.png
│   └── ...
├── 00001/
│   └── ...
└── ...
```

**Download:** [NVIDIA FFHQ](https://github.com/NVlabs/ffhq-dataset)

---

## Usage Examples

### Loading Data Programmatically

```python
from src.data import FaceDataModule

# Create data module
dm = FaceDataModule(
    name="celeba_hq",
    root="./data/celeba_hq",
    image_size=256,
    batch_size=16,
    selected_attrs=["Male", "Young"],
    pairing_enabled=True,
    transfer_attr="Male",
)

# Setup and get loaders
dm.setup("fit")
train_loader = dm.train_dataloader()

# Iterate
for batch in train_loader:
    if dm.pairing_enabled:
        source_imgs = batch["source_image"]      # [B, 3, 256, 256]
        target_imgs = batch["target_image"]      # [B, 3, 256, 256]
        source_attrs = batch["source_attributes"] # [B, num_attrs]
    else:
        imgs = batch["image"]
        attrs = batch["attributes"]
```

### Filtering by Attributes

```python
# Load only young males
dm = FaceDataModule(
    name="celeba_hq",
    root="./data/celeba_hq",
    filter_attrs={"Male": 1, "Young": 1},
)
```

### Using Transforms

```python
from src.data.transforms import get_train_transforms, denormalize

# Get transforms
transform = get_train_transforms(
    image_size=256,
    horizontal_flip=True,
    color_jitter=True,
)

# Denormalize for visualization ([-1,1] -> [0,1])
img_display = denormalize(img_tensor)
```

---

## For Students

Model implementations go in `src/models/`. See [`src/models/README.md`](src/models/README.md) for detailed instructions.

### Recommended Implementation Order

1. **Encoder** (`src/models/encoders/`): Implement e4e or pSp for GAN inversion
2. **Generator** (`src/models/generators/`): Load pretrained StyleGAN2/3
3. **Losses** (`src/models/losses/`): LPIPS, identity (ArcFace), adversarial
4. **Directions** (`src/models/directions/`): InterFaceGAN or GANSpace
5. **Lightning Module** (`src/models/lightning_modules/`): Training logic

### Example Model Config

Create `configs/model/stylegan_e4e.yaml`:

```yaml
_target_: src.models.lightning_modules.transformer.FaceTransformer

encoder:
  _target_: src.models.encoders.e4e.E4EEncoder
  input_size: 256

generator:
  _target_: src.models.generators.stylegan2.StyleGAN2Generator
  pretrained: ${paths.pretrained_dir}/stylegan2-ffhq-256.pt

learning_rate: 1e-4
```

Then run:
```bash
python scripts/train.py model=stylegan_e4e experiment=multi_attr
```

---

## Development

```bash
# Run linter
make lint

# Format code
make format

# Run tests
make test

# Clean build artifacts
make clean
```

---

## References

### Papers

- [StyleGAN2](https://arxiv.org/abs/1912.04958) - Karras et al.
- [e4e](https://arxiv.org/abs/2102.02766) - Tov et al.
- [InterFaceGAN](https://arxiv.org/abs/1907.10786) - Shen et al.
- [GANSpace](https://arxiv.org/abs/2004.02546) - Härkönen et al.

### Datasets

- [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)
- [FFHQ](https://github.com/NVlabs/ffhq-dataset)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

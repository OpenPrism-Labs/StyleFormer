# StyleFormer

**Face Transformation with Multi-Attribute Style Transfer**

A modular experimental framework for face transformation research, supporting simultaneous manipulation of multiple facial attributes (age, gender, etc.) using StyleGAN latent space editing.

---

## Features

- **Multi-Attribute Transformation**: Manipulate age and gender simultaneously with disentanglement constraints
- **Flexible Data Pipeline**: PyTorch Lightning DataModule with CelebA-HQ (labeled) and FFHQ (unlabeled) support
- **Paired Dataset Mode**: Automatic pairing of images with opposite attributes for style transfer training
- **Evaluation Metrics**: FID, identity similarity (ArcFace), and attribute accuracy
- **Inference Pipeline**: Batch processing, visualization grids, and interpolation
- **Pretrained Model Loading**: Utilities for StyleGAN2, e4e, and ArcFace
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
├── configs/                          # Hydra configuration files
│   ├── base.yaml                     # Shared defaults
│   ├── evaluation.yaml               # Evaluation settings
│   ├── inference.yaml                # Inference settings
│   ├── attributes/default.yaml       # Multi-attribute definitions
│   ├── data/
│   │   ├── celeba_hq.yaml           # CelebA-HQ dataset config
│   │   └── ffhq.yaml                # FFHQ dataset config
│   ├── experiments/
│   │   └── multi_attr.yaml          # Multi-attribute experiment
│   └── paths/default.yaml           # Dataset and output paths
│
├── src/                              # Source code
│   ├── data/                         # Data loading infrastructure
│   │   ├── datasets/                # Dataset implementations
│   │   │   ├── base.py              # Abstract base classes
│   │   │   ├── celeba_hq.py         # CelebA-HQ with attributes
│   │   │   └── ffhq.py              # FFHQ dataset
│   │   ├── datamodule.py            # PyTorch Lightning DataModule
│   │   └── transforms.py            # Image augmentation pipelines
│   │
│   ├── evaluation/                   # Evaluation infrastructure
│   │   ├── evaluator.py             # Unified evaluation interface
│   │   └── metrics/
│   │       ├── fid.py               # Fréchet Inception Distance
│   │       ├── identity.py          # Identity similarity (ArcFace)
│   │       └── attribute.py         # Attribute classification accuracy
│   │
│   ├── inference/                    # Inference infrastructure
│   │   └── pipeline.py              # InferencePipeline, MultiAttributePipeline
│   │
│   ├── models/                       # Model implementations
│   │   ├── encoders/                # Image-to-latent encoders
│   │   │   └── base.py              # BaseEncoder, GradualStyleBlock
│   │   ├── generators/              # Latent-to-image generators
│   │   │   └── base.py              # BaseGenerator, SynthesisLayer
│   │   ├── discriminators/          # Discriminator networks
│   │   │   └── base.py              # BaseDiscriminator
│   │   ├── losses/                  # Loss functions
│   │   │   ├── base.py              # L1, L2, reconstruction losses
│   │   │   ├── perceptual.py        # VGG perceptual, LPIPS
│   │   │   ├── identity.py          # ArcFace identity loss
│   │   │   └── adversarial.py       # GAN losses, R1 regularization
│   │   ├── lightning_modules/       # PyTorch Lightning modules
│   │   │   └── base.py              # BaseTransformerModule
│   │   ├── pretrained/              # Pretrained model loading
│   │   │   ├── download.py          # Model download utilities
│   │   │   ├── stylegan2.py         # StyleGAN2 loader
│   │   │   ├── e4e.py               # e4e/pSp encoder loader
│   │   │   └── arcface.py           # ArcFace loader
│   │   └── README.md                # Implementation guide
│   │
│   ├── config/schema.py             # Configuration dataclasses
│   └── utils/                       # Utilities
│       ├── io.py                    # Checkpoint I/O
│       ├── logging.py               # W&B and console logging
│       └── visualization.py         # Image grid generation
│
├── scripts/                          # Entry points
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation script
│   ├── inference.py                 # Inference script
│   └── prepare_data.py              # Dataset preparation
│
├── data/                             # Datasets (gitignored)
├── pretrained/                       # Pretrained weights (gitignored)
├── outputs/                          # Experiment outputs (gitignored)
└── notebooks/                        # Jupyter notebooks
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

## Evaluation

Evaluate trained models using FID, identity similarity, and attribute accuracy metrics.

### Command Line

```bash
# Basic evaluation
python scripts/evaluate.py \
    --generated-dir outputs/generated \
    --source-dir data/celeba_hq/test \
    --real-dir data/celeba_hq/train

# With target attributes for accuracy measurement
python scripts/evaluate.py \
    --generated-dir outputs/generated \
    --source-dir data/celeba_hq/test \
    --target-age old \
    --target-gender female

# Skip certain metrics
python scripts/evaluate.py \
    --generated-dir outputs/generated \
    --source-dir data/celeba_hq/test \
    --no-fid \
    --no-attribute

# Save results to JSON
python scripts/evaluate.py \
    --generated-dir outputs/generated \
    --source-dir data/celeba_hq/test \
    --output-file results/evaluation.json
```

### Programmatic Usage

```python
from src.evaluation import Evaluator, FIDCalculator, IdentitySimilarity

# Full evaluation
evaluator = Evaluator(device="cuda")
results = evaluator.evaluate_full(
    source_loader=source_loader,
    generated_loader=generated_loader,
    real_loader=real_loader,
    target_attributes={"age": "old", "gender": "female"},
)
print(evaluator.format_results(results))

# Individual metrics
fid_calc = FIDCalculator(device="cuda")
fid_score = fid_calc.calculate_from_dataloaders(real_loader, fake_loader)

id_sim = IdentitySimilarity(device="cuda")
similarity = id_sim.compute_similarity(source_img, generated_img)
```

### Available Metrics

| Metric | Description | Lower/Higher is Better |
|--------|-------------|------------------------|
| **FID** | Fréchet Inception Distance - image quality & diversity | Lower |
| **Identity Similarity** | Cosine similarity of ArcFace embeddings | Higher |
| **Attribute Accuracy** | Classification accuracy on target attributes | Higher |

---

## Inference

Run inference with trained models for single images, batches, or directories.

### Command Line

```bash
# Single image transformation
python scripts/inference.py \
    --checkpoint checkpoints/model.ckpt \
    --model-class StyleGANTransformer \
    --input input.jpg \
    --output output.jpg \
    --target-age old \
    --target-gender female

# Directory processing
python scripts/inference.py \
    --checkpoint checkpoints/model.ckpt \
    --model-class StyleGANTransformer \
    --input-dir inputs/ \
    --output-dir outputs/ \
    --target-age old

# Create comparison grid
python scripts/inference.py \
    --checkpoint checkpoints/model.ckpt \
    --model-class StyleGANTransformer \
    --input-dir inputs/ \
    --output-grid grid.png \
    --target-age old

# Interpolation between attributes
python scripts/inference.py \
    --checkpoint checkpoints/model.ckpt \
    --model-class StyleGANTransformer \
    --input input.jpg \
    --output-dir interpolation/ \
    --target-age old \
    --interpolate \
    --n-steps 10

# Multi-attribute with custom strengths
python scripts/inference.py \
    --checkpoint checkpoints/model.ckpt \
    --model-class StyleGANTransformer \
    --input input.jpg \
    --output output.jpg \
    --target-age old --age-strength 1.0 \
    --target-gender female --gender-strength 0.5
```

### Programmatic Usage

```python
from src.inference import InferencePipeline
from src.inference.pipeline import MultiAttributePipeline

# Load pipeline from checkpoint
pipeline = InferencePipeline.from_checkpoint(
    "checkpoints/model.ckpt",
    model_class=MyTransformationModel,
)

# Single image transformation
result = pipeline.transform(
    "input.jpg",
    target_attributes={"age": "old", "gender": "female"},
)
pipeline.save_result(result, "output.jpg")

# Batch processing
results = pipeline.transform_batch(
    image_paths,
    target_attributes={"age": "old"},
)

# Directory processing
n_processed = pipeline.transform_directory(
    input_dir="inputs/",
    output_dir="outputs/",
    target_attributes={"age": "old"},
)

# Create visualization grid
grid = pipeline.create_grid(results, ncols=4, include_source=True)
grid.save("comparison_grid.png")

# Multi-attribute with per-attribute strengths
multi_pipeline = MultiAttributePipeline.from_checkpoint(...)
result = multi_pipeline.transform_multi(
    "input.jpg",
    attributes={
        "age": ("old", 1.0),      # (target, strength)
        "gender": ("female", 0.5),
    },
    use_null_space=True,
)
```

---

## Pretrained Model Loading

Utilities for loading pretrained models from various sources.

### Available Models

| Model | Description | Source |
|-------|-------------|--------|
| `stylegan2-ffhq-1024` | StyleGAN2 trained on FFHQ 1024x1024 | NVIDIA |
| `stylegan2-ffhq-256` | StyleGAN2 trained on FFHQ 256x256 | NVIDIA |
| `e4e-ffhq-1024` | e4e encoder for FFHQ | omertov/encoder4editing |
| `psp-ffhq-1024` | pSp encoder for FFHQ | eladrich/pixel2style2pixel |
| `arcface-r100` | ArcFace ResNet-100 | InsightFace |
| `arcface-r50` | ArcFace ResNet-50 | InsightFace |
| `interfacegan-age` | Age direction in W space | genforce/interfacegan |
| `interfacegan-gender` | Gender direction in W space | genforce/interfacegan |

### Usage

```python
from src.models.pretrained import (
    load_stylegan2,
    load_e4e_encoder,
    load_arcface,
    download_model,
    list_available_models,
)

# List available models
models = list_available_models()
for name, description in models.items():
    print(f"{name}: {description}")

# Load StyleGAN2 generator
generator = load_stylegan2("stylegan2-ffhq-1024", device="cuda")
z = torch.randn(1, 512, device="cuda")
image = generator(z)  # Generate image from Z
w = generator.mapping(z)  # Get W latent
image = generator.synthesis(w)  # Generate from W

# Load e4e encoder
encoder = load_e4e_encoder("e4e-ffhq-1024", device="cuda")
w_plus = encoder(image)  # Encode to W+ space

# Load ArcFace for identity
arcface = load_arcface("arcface-r100", device="cuda")
embedding = arcface(face_image)  # Get identity embedding
similarity = arcface.compute_similarity(face1, face2)

# Download models manually
path = download_model("stylegan2-ffhq-1024")
```

> **Note**: Model URLs in the registry are placeholders. Download models from official sources and place them in `~/.cache/styleformer/` or update the registry.

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

Model implementations go in `src/models/`. Base classes with detailed docstrings are provided as starting points.

### What's Provided (Infrastructure)

| Component | Location | Description |
|-----------|----------|-------------|
| Base Encoder | `src/models/encoders/base.py` | Abstract class with interface |
| Base Generator | `src/models/generators/base.py` | Abstract class with interface |
| Base Discriminator | `src/models/discriminators/base.py` | Abstract class with interface |
| Loss Functions | `src/models/losses/` | L1, L2, perceptual, identity, adversarial |
| Lightning Module | `src/models/lightning_modules/base.py` | Training loop skeleton |
| Pretrained Loaders | `src/models/pretrained/` | Loading utilities (architecture stubs) |

### What Students Implement

1. **Encoder Architecture** (`src/models/encoders/`):
   - Implement `forward()` method in a class inheriting `BaseEncoder`
   - Options: e4e, pSp, or custom encoder

2. **Generator Architecture** (`src/models/generators/`):
   - Implement `mapping()` and `synthesis()` in a class inheriting `BaseGenerator`
   - Or load from pretrained using `src/models/pretrained/stylegan2.py`

3. **Latent Editing** (`edit_latent()` method):
   - InterFaceGAN linear directions
   - GANSpace PCA directions
   - Or learned edit network

4. **Training Logic** (`src/models/lightning_modules/`):
   - Implement `training_step()` and `validation_step()`
   - Configure losses and optimizers

### Recommended Implementation Order

1. **Encoder** (`src/models/encoders/`): Implement e4e or pSp for GAN inversion
2. **Generator** (`src/models/generators/`): Load pretrained StyleGAN2/3
3. **Losses** (`src/models/losses/`): Implement VGG loading for perceptual loss
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
- [ArcFace](https://arxiv.org/abs/1801.07698) - Deng et al.

### Datasets

- [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)
- [FFHQ](https://github.com/NVlabs/ffhq-dataset)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

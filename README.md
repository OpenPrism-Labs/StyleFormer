# StyleFormer

Face-transformation **research infrastructure** built with PyTorch, Lightning and Hydra.

## Implemented status

- CelebA-HQ/CelebA-format labeled image loading and FFHQ loading.
- Attribute filtering, opposite-attribute pairing, image transforms and Lightning DataLoaders.
- Composable experiment configuration, dataset verification, checkpoint and visualization utilities.

**Not implemented:** face transformation, GAN inversion, generators, losses, latent-direction estimation or a training loop. `scripts/train.py` is a data-pipeline smoke check, despite its historical name. The multi-attribute config describes an experiment; it does not implement simultaneous editing or disentanglement. See [the model integration guide](src/models/README.md) and [research proposal](idea.md).

## Installation

Python 3.10+ is declared supported; Python 3.12 is the Conda environment default. The available PyTorch wheels may impose a newer Python minimum. CPU is sufficient for data checks; an NVIDIA GPU or CUDA toolkit is not required.

From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU example (Linux/Windows): install a matching pair from the same index.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

For CUDA, ROCm or other accelerators, replace the CPU command with the platform-specific command from [PyTorch's installation selector](https://pytorch.org/get-started/locally/). Do not mix independently chosen torch/torchvision versions or assume CUDA 11.8 is still the appropriate runtime. Binary wheels supply their runtime dependencies; local CUDA development tools are needed separately for extensions that compile CUDA code.

Alternatively, from the repository root:

```bash
conda env create -f environment.yaml
conda activate styleformer
```

This uses Conda for Python and pip for the package, not the discontinued official PyTorch Conda channel. It uses pip's default wheel selection; use the virtual-environment instructions when you want explicit accelerator selection before installing the project. See the [PyTorch 2.6 release announcement](https://pytorch.org/blog/pytorch2-6/) for the Conda distribution change.

W&B is optional:

```bash
python -m pip install -e '.[logging]'
```

OpenCV, pandas and einops are not used by the implemented code and are no longer mandatory dependencies. Model implementations may need their own dependencies later. Requirements are compatibility lower bounds, not an exact reproducible environment lock; record resolved versions with each experiment.

The distribution is named `styleformer`; the existing Python import namespace is `src`. Repository scripts and Hydra configs are checkout workflows, not installed console commands.

## Data preparation and smoke checks

Run commands from the repository root after installation:

```bash
python scripts/prepare_data.py --help
python scripts/prepare_data.py --dataset celeba_hq
python scripts/prepare_data.py --dataset ffhq

# After obtaining the datasets, verify the files.
python scripts/prepare_data.py --verify

# Load real batches; this does not train a model.
python scripts/train.py
python scripts/train.py experiments=multi_attr
python scripts/train.py data=ffhq dataloader.num_workers=0

# Inspect configuration without loading a dataset.
python scripts/train.py --cfg job --resolve
python scripts/train.py experiments=multi_attr --cfg job --resolve
```

Datasets and pretrained weights are not bundled or automatically downloaded. Download instructions do not imply permission to redistribute images or use models commercially.

## Datasets

### CelebA-HQ

[CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) contains 30,000 high-quality images derived from CelebA. Its numeric HQ index is **not** the original CelebA filename. Original annotation and partition files must be joined through the official HQ-to-CelebA mapping. Alternatively, use annotation files already keyed by the exact image filenames.

The loader uses `images/`, with `img_align_celeba/` as an alternative for original CelebA-format layouts. Keep annotation files under the dataset root. Original CelebA has 40 binary annotations; `Young` is not chronological age and `Male` is an annotation, not a person's gender identity.

Auto-detected annotation files, in preference order: `CelebAMask-HQ-attribute-anno.txt`, `list_attr_celeba.txt`, `attributes.txt`. These must use the CelebA count/header/`-1` or `1` text format. Mapping files are `CelebA-HQ-to-CelebA-mapping.txt` or `image_list.txt`, with `idx` and `orig_file` columns. An optional `list_eval_partition.txt` supplies official splits. With a mapping, partition keys refer to original filenames; missing entries are errors.

Override locations with `data.attribute_file=...`, `data.mapping_file=...`, and `data.partition_file=...` (relative to the dataset root), or the equivalent `FaceDataModule` keyword arguments. Without official partitions, seeded splits are formed before filtering so filtering cannot move images between splits.

### FFHQ

[FFHQ](https://github.com/NVlabs/ffhq-dataset) contains 70,000 images without CelebA-style labels. Point the root at an image directory (flat or nested), not the parent of several duplicate resolution sets:

```text
data/ffhq/
  00000/
    00000.png
    ...
  01000/
    01000.png
    ...
```

Unlabeled batches have an empty attribute dimension. The current FFHQ loader does not accept pseudo-labels or support paired mode. Images are recursively discovered as PNG/JPG/JPEG; archives and TFRecords are not supported. Point the loader at extracted files. Observe the dataset's terms and individual image licenses.

## Configuration and Python usage

```bash
# These change data loading; optimizer settings are not consumed by a training loop.
python scripts/train.py dataloader.batch_size=32
python scripts/train.py data.pairing.enabled=true data.pairing.transfer_attr=Male
```

Configuration groups live under `configs/`; root configuration is `configs/base.yaml`. Attribute targets and null-space-projection settings are reserved for a future model implementation.

Use **`experiments=multi_attr`** (plural) to select the config group; `experiment` (singular) contains metadata such as `experiment.seed`. W&B defaults to disabled (`logging.project=null`); set a project to enable it, and `logging.offline=false` only when you intend to upload a run.

Default data paths use `STYLEFORMER_ROOT` when set, otherwise the launch directory. Relative paths remain anchored to the launch directory even with Hydra directory changes. `--verify` exits nonzero for missing metadata, invalid mappings, missing images or corrupt images. Select one dataset with `--dataset ffhq` or `--dataset celeba_hq` if you do not have both.

```python
from src.data import FaceDataModule

dm = FaceDataModule(
    name="celeba_hq",
    root="./data/celeba_hq",
    image_size=256,
    batch_size=16,
    num_workers=0,
    selected_attrs=["Male", "Young"],
)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
images = batch["image"]
attributes = batch["attributes"]
```

Enable pairing with `pairing_enabled=True, transfer_attr="Male"`. Paired batches expose `source_image`, `target_image`, `source_attributes` and `target_attributes`. These are usually **different people** with opposite annotations, not identity-matched before/after images. A pixel loss against the paired target can therefore conflict with identity preservation.

Training transforms are stochastic; validation/test transforms are deterministic. Default face normalization maps image intensities to `[-1, 1]`; denormalize before displaying them.

## Utilities

`src.utils.io` saves checkpoints and loads tensor state dictionaries with restricted `weights_only=True` deserialization. For a trusted legacy object checkpoint only, explicitly pass `weights_only=False`; this can execute arbitrary pickle code. See [PyTorch serialization guidance](https://docs.pytorch.org/docs/stable/notes/serialization.html). Pretrained-weight loading accepts plain state dictionaries or dictionaries containing `state_dict`/`model`; it is not a loader for arbitrary upstream `.pkl` models.

`src.utils.visualization` creates/saves image grids and optionally logs images to an initialized W&B run. Calling W&B helpers without the optional dependency raises an actionable error rather than silently dropping images.

## Development

```bash
make lint
make format
make test
make test-cov
```

Tests use synthetic images; they do not require downloaded face datasets or pretrained weights. `make train` and `make train-multi-attr` run the historical smoke-check entry point, not training. `make clean-outputs` permanently removes experiment output directories; use deliberately.

### Verified modernization scope

Verified on Python 3.13 with CPU PyTorch 2.14.0, torchvision 0.29.0, Lightning 2.6.6, NumPy 2.5.2 and Hydra 1.3.6:

- 15 synthetic regression tests, Ruff and mypy.
- Editable installation, source/wheel builds and importing the built wheel outside the checkout.
- Default paired and multi-attribute data smoke runs, unlabeled FFHQ batches, and relative paths with Hydra directory changes.
- Dataset verification success and nonzero failure status.
- Checkpoint roundtrips, image saving, and gradient-bearing image logging to an offline W&B run.

`src/config/schema.py` contains dataclasses, not Pydantic models. Merging the base, FFHQ and multi-attribute configs into the structured schema was checked; the CLI does not automatically register that schema.

Full face datasets, pretrained models, GPU execution, Conda environment creation and every declared minimum dependency version were not exercised. Dataset verification establishes local loader usability, not official dataset completeness or cryptographic integrity. CPU runs with `pin_memory=true` may emit PyTorch's harmless no-accelerator warning; use `dataloader.pin_memory=false` when pinned GPU-transfer buffers are not needed.

## References and license

- [StyleGAN2](https://arxiv.org/abs/1912.04958)
- [e4e](https://arxiv.org/abs/2102.02766)
- [InterFaceGAN](https://arxiv.org/abs/1907.10786)
- [GANSpace](https://arxiv.org/abs/2004.02546)

Project code: [MIT](LICENSE). Dataset, image and external pretrained-model licenses apply separately.

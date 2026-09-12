# Model Implementation Guide

No encoder, generator, loss, latent-direction estimator or Lightning training module is implemented in this directory. `scripts/train.py` currently checks the data pipeline only; it does not train or transform faces. Attribute transformation settings are research configuration, not active model functionality.

## Suggested integration sequence

1. Choose a compatible pretrained generator and inversion encoder. Verify architecture, input normalization, resolution and latent shape; pSp/e4e and StyleGAN variants are not interchangeable simply by renaming a checkpoint.
2. Establish reconstruction quality before introducing attribute edits. Freeze pretrained components deliberately and optimize only intended trainable parameters.
3. Implement a real `lightning.LightningModule` with a differentiable loss returned from `training_step`, an optimizer, and validation metrics.
4. Integrate model construction and `lightning.Trainer.fit(model, datamodule=datamodule)` into an explicit training entry point. The current smoke script has neither operation.
5. Add latent-direction estimation and evaluate attribute changes, identity preservation and cross-attribute effects separately.

Possible subdirectories are `encoders/`, `generators/`, `losses/`, `directions/`, and `lightning_modules/`; these do not exist yet. Likewise, there is no `configs/model/` group or working `model=stylegan_e4e` command. Add a model config only after its `_target_` implementation exists and is integrated into a training entry point.

## Data contracts

Use `FaceDataModule` from `src.data`. Labeled samples expose image and attribute tensors; FFHQ has an empty attribute dimension and no pseudo-label input API. Paired mode supplies source and target images with opposite annotations, generally depicting **different identities**. Such pairs are not pixel-aligned reconstruction targets or ground-truth edits of the source person.

CelebA labels describe dataset annotations. `Young` is binary, not an age in years; `Male` is not a measurement of gender identity. Continuous latent strength needs a separately learned and evaluated interpretation.

## Checkpoints and resources

- [StyleGAN2-ADA PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [e4e](https://github.com/omertov/encoder4editing)
- [pSp](https://github.com/eladrich/pixel2style2pixel)
- [InsightFace](https://github.com/deepinsight/insightface)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)

These are external research dependencies, not bundled implementations. Check their code, model and data licenses independently of this repository's MIT license. LPIPS is not installed by the base package.

`src.utils.io.load_checkpoint` uses restricted `weights_only=True` loading. Tensor state dictionaries are recommended. `weights_only=False` is an explicit opt-in for trusted legacy objects and can execute arbitrary code; never use it to work around errors from untrusted downloads. NVIDIA `.pkl` generator files require their upstream loading code and are not interchangeable with plain PyTorch state dictionaries.

W&B is optional (`python -m pip install -e '.[logging]'`). Ensure a run is initialized before calling image logging utilities, and close it after use. Do not upload images without permission.

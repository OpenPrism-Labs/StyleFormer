# Face Transformation Research Proposal

This is a research direction, not a description of implemented models. The repository currently supplies dataset loading, configuration and utilities. See [README.md](README.md) for runnable functionality.

## Objective

Model editing as $y = f(x, a)$, where $x$ is an input face image, $a$ is the requested attribute change, and $y$ is the edited image. Evaluate identity preservation, requested appearance change and image quality separately; none is guaranteed by choosing a model family.

CelebA's `Male` and `Young` fields are binary dataset annotations, not measurements of gender identity or chronological age. They cannot supervise an exact age target such as “50 years old.” Use consented images and evaluate annotation bias, demographic performance and unintended changes.

## Candidate approaches

### Image-to-image translation

[CycleGAN](https://arxiv.org/abs/1703.10593) learns mappings between domains using cycle consistency. [StarGAN v2](https://arxiv.org/abs/1912.01865) uses a style encoder and mapping network for diverse multi-domain outputs. These architectures do not all condition by concatenating an attribute label to an image.

### StyleGAN latent editing

A candidate baseline is inversion followed by latent editing:

1. Encode a real image into $w$ or $w^+$ using an inversion method such as [pSp](https://arxiv.org/abs/2008.00951) or [e4e](https://arxiv.org/abs/2102.02766).
2. Edit a latent code along a learned direction: $w' = w + \alpha d$.
3. Decode with a compatible pretrained synthesis network.

[StyleGAN2](https://arxiv.org/abs/1912.04958) and [StyleGAN3](https://arxiv.org/abs/2106.12423) have different architectural goals and are not interchangeable checkpoint formats. Inversion fidelity and editability trade off; latent directions may alter correlated attributes. [InterFaceGAN](https://arxiv.org/abs/1907.10786) and [GANSpace](https://arxiv.org/abs/2004.02546) offer different direction-discovery methods.

### Diffusion-based editing

[Latent diffusion](https://arxiv.org/abs/2112.10752) is another candidate, not automatically a better or newer replacement for every GAN workflow. Structural conditioning such as [ControlNet](https://arxiv.org/abs/2302.05543) can guide geometry but does not guarantee identity preservation. Quality and speed depend on the model, sampler, resolution and hardware.

## Candidate constraints and losses

- Reconstruction and perceptual losses measure different aspects of similarity; LPIPS is not an identity guarantee.
- Face embeddings can provide an identity-similarity objective, with limitations and bias that require evaluation.
- Attribute supervision requires suitable annotations; binary labels cannot establish precise continuous control.
- Null-space projection can reduce interference relative to estimated linear directions. It does not ensure nonlinear disentanglement.
- Adversarial losses encourage distributional realism, not a guarantee that every output is realistic.

Loss weights must be calibrated to their scales and validated experimentally. There is no universal set of “typical” weights for this unimplemented architecture.

## Data and evaluation

- [FFHQ](https://github.com/NVlabs/ffhq-dataset): 70,000 face images, distributed without CelebA-style attribute labels. Observe the dataset and individual image licenses.
- [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans): 30,000 images derived from CelebA. Original CelebA annotations must be joined through the HQ-to-original mapping, not matched by numeric image index.
- Opposite-attribute pairs are generally different people, not ground-truth before/after edits. Pixel reconstruction against such a target can conflict with identity preservation.
- Keep train/validation/test samples disjoint. Report image quality, attribute success, identity similarity and unintended changes; examine performance across relevant subgroups.

Compare candidate methods on the same data, resolution and hardware before claiming a winner. The project does not currently implement or benchmark these methods.

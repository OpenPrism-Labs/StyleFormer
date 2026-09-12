"""Visualization utilities for image outputs."""

from pathlib import Path

import torch
from torchvision.utils import make_grid as tv_make_grid
from torchvision.utils import save_image as tv_save_image


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    std: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Denormalize tensor from normalized range to [0, 1].

    Args:
        tensor: Normalized image tensor [B, C, H, W] or [C, H, W].
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Denormalized tensor in [0, 1] range.
    """
    if tensor.dim() not in (3, 4):
        raise ValueError("Expected a CHW or BCHW image tensor")
    channels = tensor.shape[-3]
    if len(mean) != channels or len(std) != channels:
        raise ValueError("Normalization statistics must match the image channels")
    mean_t = tensor.new_tensor(mean).view(-1, 1, 1)
    std_t = tensor.new_tensor(std).view(-1, 1, 1)

    return tensor * std_t + mean_t


def make_grid(
    images: torch.Tensor | list[torch.Tensor],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: tuple[float, float] | None = None,
    denorm: bool = True,
) -> torch.Tensor:
    """Create a grid of images.

    Args:
        images: Tensor of images [B, C, H, W] or list of tensors.
        nrow: Number of images per row.
        padding: Padding between images.
        normalize: Normalize to [0, 1] range.
        value_range: Expected value range of input.
        denorm: Denormalize from [-1, 1] to [0, 1] first.

    Returns:
        Grid tensor [C, H, W].
    """
    if isinstance(images, list):
        images = torch.stack(images, dim=0)

    if denorm:
        images = denormalize(images)
        images = images.clamp(0, 1)
        normalize = False  # Already normalized

    if value_range is None and not denorm:
        value_range = (-1, 1)

    grid = tv_make_grid(
        images,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
    )

    return grid


def save_images(
    images: torch.Tensor | list[torch.Tensor],
    filepath: str | Path,
    nrow: int = 8,
    denorm: bool = True,
) -> Path:
    """Save images as a grid.

    Args:
        images: Tensor of images [B, C, H, W] or list of tensors.
        filepath: Path to save image.
        nrow: Number of images per row.
        denorm: Denormalize from [-1, 1] to [0, 1].

    Returns:
        Path to saved image.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(images, list):
        images = torch.stack(images, dim=0)

    if denorm:
        images = denormalize(images)
        images = images.clamp(0, 1)

    tv_save_image(images, filepath, nrow=nrow)

    return filepath


def create_comparison_grid(
    source: torch.Tensor,
    target: torch.Tensor,
    generated: torch.Tensor,
    denorm: bool = True,
) -> torch.Tensor:
    """Create a comparison grid: source | generated | target.

    Args:
        source: Source images [B, C, H, W].
        target: Target images [B, C, H, W].
        generated: Generated images [B, C, H, W].
        denorm: Denormalize images.

    Returns:
        Grid tensor showing comparison.
    """
    if denorm:
        source = denormalize(source).clamp(0, 1)
        target = denormalize(target).clamp(0, 1)
        generated = denormalize(generated).clamp(0, 1)

    # Interleave: [s1, g1, t1, s2, g2, t2, ...]
    comparison = torch.stack([source, generated, target], dim=1)
    comparison = comparison.view(-1, *source.shape[1:])

    grid = tv_make_grid(
        comparison,
        nrow=3,
        padding=2,
        normalize=False,
    )

    return grid


def log_images_to_wandb(
    images: dict[str, torch.Tensor],
    step: int,
    prefix: str = "images",
    denorm: bool = True,
) -> None:
    """Log images to Weights & Biases.

    Args:
        images: Dictionary of image tensors.
        step: Training step.
        prefix: Prefix for logged images.
        denorm: Denormalize images.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError('W&B logging requires: python -m pip install -e ".[logging]"') from None

    log_dict = {}

    for name, tensor in images.items():
        if denorm:
            tensor = denormalize(tensor).clamp(0, 1)

        # Convert to numpy [H, W, C]
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first in batch

        img_np = tensor.detach().float().permute(1, 2, 0).cpu().numpy()
        log_dict[f"{prefix}/{name}"] = wandb.Image(img_np)

    wandb.log(log_dict, step=step)

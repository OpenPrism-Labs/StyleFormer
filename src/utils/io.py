"""I/O utilities for checkpoints and file operations."""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    state: dict[str, Any],
    filepath: str | Path,
    filename: str = "checkpoint.pt",
) -> Path:
    """Save a checkpoint to disk.

    Args:
        state: State dictionary to save.
        filepath: Directory to save checkpoint in.
        filename: Checkpoint filename.

    Returns:
        Path to saved checkpoint.
    """
    filepath = Path(filepath)
    filepath.mkdir(parents=True, exist_ok=True)

    checkpoint_path = filepath / filename
    torch.save(state, checkpoint_path)

    return checkpoint_path


def load_checkpoint(
    filepath: str | Path,
    map_location: str | torch.device = "cpu",
    *,
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint from disk.

    Args:
        filepath: Path to checkpoint file.
        map_location: Device to load checkpoint to.
        weights_only: Restrict loading to tensors and primitive state by default.
            Set False only for trusted legacy checkpoints: pickle can execute code.

    Returns:
        Loaded state dictionary.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    return torch.load(filepath, map_location=map_location, weights_only=weights_only)


def load_pretrained_weights(
    model: torch.nn.Module,
    filepath: str | Path,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    """Load pretrained weights into a model.

    Args:
        model: Model to load weights into.
        filepath: Path to weights file.
        strict: Whether to strictly enforce matching keys.
        map_location: Device to load weights to.

    Returns:
        Tuple of (missing_keys, unexpected_keys).
    """
    checkpoint = load_checkpoint(filepath, map_location)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=strict)

    return result.missing_keys, result.unexpected_keys

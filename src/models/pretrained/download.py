"""Utilities for downloading pretrained model weights.

This module provides functions to download and manage pretrained weights
for various models used in face transformation.

Supported models:
- StyleGAN2-FFHQ: StyleGAN2 trained on FFHQ 1024x1024
- e4e-FFHQ: e4e encoder for FFHQ
- pSp-FFHQ: pSp encoder for FFHQ  
- ArcFace: Face recognition model for identity loss

Example:
    >>> from src.models.pretrained import download_model, get_model_path
    >>> # Download a model
    >>> download_model("stylegan2-ffhq-1024")
    >>> # Get path to downloaded model
    >>> path = get_model_path("stylegan2-ffhq-1024")
    >>> print(path)
    ~/.cache/styleformer/stylegan2-ffhq-1024.pt
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from tqdm import tqdm


# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "styleformer"

# Model registry with download URLs and checksums
# Note: These are placeholder URLs - students should update with actual sources
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "stylegan2-ffhq-1024": {
        "url": "https://example.com/stylegan2-ffhq-1024.pt",  # Placeholder
        "filename": "stylegan2-ffhq-1024.pt",
        "sha256": None,  # Add checksum for verification
        "description": "StyleGAN2 trained on FFHQ at 1024x1024 resolution",
        "source": "https://github.com/NVlabs/stylegan2-ada-pytorch",
    },
    "stylegan2-ffhq-256": {
        "url": "https://example.com/stylegan2-ffhq-256.pt",
        "filename": "stylegan2-ffhq-256.pt",
        "sha256": None,
        "description": "StyleGAN2 trained on FFHQ at 256x256 resolution",
        "source": "https://github.com/NVlabs/stylegan2-ada-pytorch",
    },
    "e4e-ffhq-1024": {
        "url": "https://example.com/e4e-ffhq-1024.pt",
        "filename": "e4e-ffhq-1024.pt",
        "sha256": None,
        "description": "e4e encoder for FFHQ at 1024x1024",
        "source": "https://github.com/omertov/encoder4editing",
    },
    "psp-ffhq-1024": {
        "url": "https://example.com/psp-ffhq-1024.pt",
        "filename": "psp-ffhq-1024.pt",
        "sha256": None,
        "description": "pSp encoder for FFHQ at 1024x1024",
        "source": "https://github.com/eladrich/pixel2style2pixel",
    },
    "arcface-r100": {
        "url": "https://example.com/arcface-r100.pt",
        "filename": "arcface-r100.pt",
        "sha256": None,
        "description": "ArcFace ResNet-100 for face recognition",
        "source": "https://github.com/deepinsight/insightface",
    },
    "arcface-r50": {
        "url": "https://example.com/arcface-r50.pt",
        "filename": "arcface-r50.pt",
        "sha256": None,
        "description": "ArcFace ResNet-50 for face recognition",
        "source": "https://github.com/deepinsight/insightface",
    },
    # Age/Gender attribute directions for latent editing
    "interfacegan-age": {
        "url": "https://example.com/interfacegan-age.npy",
        "filename": "interfacegan-age.npy",
        "sha256": None,
        "description": "InterFaceGAN age direction in W space",
        "source": "https://github.com/genforce/interfacegan",
    },
    "interfacegan-gender": {
        "url": "https://example.com/interfacegan-gender.npy",
        "filename": "interfacegan-gender.npy",
        "sha256": None,
        "description": "InterFaceGAN gender direction in W space",
        "source": "https://github.com/genforce/interfacegan",
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        """Update progress bar.

        Args:
            b: Number of blocks transferred.
            bsize: Size of each block.
            tsize: Total size.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_cache_dir() -> Path:
    """Get the cache directory for pretrained models.

    Returns:
        Path to cache directory.

    Note:
        Can be overridden with STYLEFORMER_CACHE_DIR environment variable.
    """
    cache_dir = os.environ.get("STYLEFORMER_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return DEFAULT_CACHE_DIR


def list_available_models() -> dict[str, str]:
    """List all available pretrained models.

    Returns:
        Dict mapping model names to descriptions.

    Example:
        >>> models = list_available_models()
        >>> for name, desc in models.items():
        ...     print(f"{name}: {desc}")
    """
    return {
        name: info["description"]
        for name, info in MODEL_REGISTRY.items()
    }


def get_model_path(model_name: str, cache_dir: Path | None = None) -> Path:
    """Get the local path to a pretrained model.

    Args:
        model_name: Name of the model.
        cache_dir: Cache directory (uses default if not specified).

    Returns:
        Path to the model file.

    Raises:
        ValueError: If model is not in registry.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available: {available}"
        )

    cache_dir = cache_dir or get_cache_dir()
    filename = MODEL_REGISTRY[model_name]["filename"]
    return cache_dir / filename


def verify_checksum(path: Path, expected_sha256: str | None) -> bool:
    """Verify file checksum.

    Args:
        path: Path to file.
        expected_sha256: Expected SHA256 hash.

    Returns:
        True if checksum matches or no checksum provided.
    """
    if expected_sha256 is None:
        return True

    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest() == expected_sha256


def download_model(
    model_name: str,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download a pretrained model.

    Args:
        model_name: Name of the model to download.
        cache_dir: Cache directory (uses default if not specified).
        force: Force re-download even if file exists.

    Returns:
        Path to downloaded model file.

    Raises:
        ValueError: If model is not in registry.
        RuntimeError: If download fails or checksum doesn't match.

    Example:
        >>> path = download_model("stylegan2-ffhq-1024")
        >>> print(f"Downloaded to: {path}")

    Note:
        The URLs in MODEL_REGISTRY are placeholders. Students should:
        1. Download models manually from official sources
        2. Update the URLs if hosting their own copies
        3. Add checksums for verification
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available: {available}"
        )

    model_info = MODEL_REGISTRY[model_name]
    cache_dir = cache_dir or get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / model_info["filename"]

    # Check if already downloaded
    if local_path.exists() and not force:
        if verify_checksum(local_path, model_info["sha256"]):
            print(f"Model already downloaded: {local_path}")
            return local_path
        else:
            print("Checksum mismatch, re-downloading...")

    # Check if URL is placeholder
    url = model_info["url"]
    if "example.com" in url:
        raise RuntimeError(
            f"Model '{model_name}' has a placeholder URL.\n"
            f"Please download manually from: {model_info['source']}\n"
            f"And place the file at: {local_path}\n"
            f"\nAlternatively, update MODEL_REGISTRY with actual URLs."
        )

    # Download with progress bar
    print(f"Downloading {model_name}...")
    try:
        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=model_info["filename"],
        ) as pbar:
            urlretrieve(url, local_path, reporthook=pbar.update_to)
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"Download failed: {e}") from e

    # Verify checksum
    if not verify_checksum(local_path, model_info["sha256"]):
        local_path.unlink()
        raise RuntimeError(
            f"Checksum verification failed for {model_name}. "
            "The download may be corrupted."
        )

    print(f"Downloaded to: {local_path}")
    return local_path


def download_all_models(cache_dir: Path | None = None) -> dict[str, Path]:
    """Download all available pretrained models.

    Args:
        cache_dir: Cache directory.

    Returns:
        Dict mapping model names to local paths.

    Note:
        This will fail for models with placeholder URLs.
        Use for testing or when all URLs are configured.
    """
    results = {}
    for model_name in MODEL_REGISTRY:
        try:
            path = download_model(model_name, cache_dir)
            results[model_name] = path
        except RuntimeError as e:
            print(f"Skipping {model_name}: {e}")
    return results


def get_model_info(model_name: str) -> dict[str, Any]:
    """Get detailed information about a model.

    Args:
        model_name: Name of the model.

    Returns:
        Dict with model information.

    Raises:
        ValueError: If model is not in registry.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available: {available}"
        )

    info = MODEL_REGISTRY[model_name].copy()
    info["name"] = model_name
    info["local_path"] = str(get_model_path(model_name))
    info["is_downloaded"] = get_model_path(model_name).exists()

    return info


def print_model_info(model_name: str) -> None:
    """Print formatted information about a model.

    Args:
        model_name: Name of the model.
    """
    info = get_model_info(model_name)
    print(f"Model: {info['name']}")
    print(f"  Description: {info['description']}")
    print(f"  Source: {info['source']}")
    print(f"  Local path: {info['local_path']}")
    print(f"  Downloaded: {'Yes' if info['is_downloaded'] else 'No'}")

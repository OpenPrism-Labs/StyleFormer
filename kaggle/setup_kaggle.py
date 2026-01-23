#!/usr/bin/env python3
"""Kaggle environment setup script for StyleFormer.

Run this at the start of your Kaggle notebook:
    !python /kaggle/input/styleformer/kaggle/setup_kaggle.py

Or import and call setup():
    import sys
    sys.path.insert(0, '/kaggle/input/styleformer')
    from kaggle.setup_kaggle import setup
    setup()
"""

import os
import subprocess
import sys
from pathlib import Path


def is_kaggle() -> bool:
    """Check if running on Kaggle."""
    return os.path.exists("/kaggle")


def get_styleformer_path() -> Path:
    """Get path to StyleFormer source."""
    # Check common Kaggle dataset mount points
    possible_paths = [
        Path("/kaggle/input/styleformer"),
        Path("/kaggle/input/styleformer-code"),
        Path("/kaggle/working/StyleFormer"),
        Path.cwd(),
    ]

    for path in possible_paths:
        if (path / "src").exists():
            return path

    # Fallback to current directory
    return Path.cwd()


def install_dependencies(extra_packages: list[str] | None = None) -> None:
    """Install required packages not available on Kaggle by default.

    Args:
        extra_packages: Additional packages to install.
    """
    # Packages that need to be installed on Kaggle
    packages = [
        "lpips",  # For LPIPS perceptual loss
        "einops",  # For tensor operations
        "hydra-core",  # For configuration management
        "omegaconf",  # For configuration
        "pytorch-lightning",  # For training
    ]

    if extra_packages:
        packages.extend(extra_packages)

    print("Installing required packages...")
    for pkg in packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  Installed: {pkg}")
        except subprocess.CalledProcessError:
            print(f"  Warning: Failed to install {pkg}")


def setup_paths() -> Path:
    """Add StyleFormer to Python path.

    Returns:
        Path to StyleFormer root directory.
    """
    styleformer_path = get_styleformer_path()

    # Add to Python path
    if str(styleformer_path) not in sys.path:
        sys.path.insert(0, str(styleformer_path))
        print(f"Added to path: {styleformer_path}")

    return styleformer_path


def setup_directories() -> dict[str, Path]:
    """Create working directories for Kaggle.

    Returns:
        Dictionary of directory paths.
    """
    working_dir = Path("/kaggle/working")

    dirs = {
        "outputs": working_dir / "outputs",
        "checkpoints": working_dir / "checkpoints",
        "pretrained": working_dir / "pretrained",
        "generated": working_dir / "generated",
        "logs": working_dir / "logs",
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")

    return dirs


def get_dataset_paths() -> dict[str, Path | None]:
    """Get paths to common face datasets on Kaggle.

    Returns:
        Dictionary of dataset paths (None if not found).
    """
    datasets = {
        "celeba": [
            "/kaggle/input/celeba-dataset",
            "/kaggle/input/celebahq-resized-256x256",
            "/kaggle/input/celeba-hq-resized",
        ],
        "ffhq": [
            "/kaggle/input/ffhq-face-data-set",
            "/kaggle/input/flickr-faces-hq-dataset-ffhq",
            "/kaggle/input/ffhq256",
        ],
    }

    result = {}
    for name, paths in datasets.items():
        result[name] = None
        for path in paths:
            if Path(path).exists():
                result[name] = Path(path)
                print(f"Found {name}: {path}")
                break

    return result


def download_pretrained_weights(model: str = "all") -> dict[str, Path]:
    """Download pretrained model weights.

    Args:
        model: Model to download ('stylegan2', 'arcface', 'e4e', or 'all').

    Returns:
        Dictionary of model paths.
    """
    pretrained_dir = Path("/kaggle/working/pretrained")
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # Model URLs (these are example URLs - replace with actual sources)
    models = {
        "stylegan2": {
            "url": "https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",
            "filename": "stylegan2-ffhq-256.pt",
        },
        "arcface": {
            "url": "https://drive.google.com/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn",
            "filename": "arcface-r100.pt",
        },
        "e4e": {
            "url": "https://drive.google.com/uc?id=1cUv_reLE6k3604or78EranS7XzuVMWeO",
            "filename": "e4e-ffhq-encode.pt",
        },
    }

    result = {}
    models_to_download = models.keys() if model == "all" else [model]

    for model_name in models_to_download:
        if model_name not in models:
            continue

        info = models[model_name]
        target_path = pretrained_dir / info["filename"]

        if target_path.exists():
            print(f"Already exists: {target_path}")
            result[model_name] = target_path
        else:
            print(f"Note: Download {model_name} manually from: {info['url']}")
            print(f"  Save to: {target_path}")
            result[model_name] = None

    return result


def setup(
    install_packages: bool = True,
    create_dirs: bool = True,
    verbose: bool = True,
) -> dict:
    """Full Kaggle environment setup.

    Args:
        install_packages: Whether to install missing packages.
        create_dirs: Whether to create working directories.
        verbose: Whether to print status messages.

    Returns:
        Dictionary with setup information.
    """
    if verbose:
        print("=" * 50)
        print("StyleFormer Kaggle Setup")
        print("=" * 50)

    result = {
        "is_kaggle": is_kaggle(),
        "styleformer_path": None,
        "directories": {},
        "datasets": {},
    }

    # Install dependencies
    if install_packages:
        install_dependencies()

    # Setup paths
    result["styleformer_path"] = setup_paths()

    # Create directories
    if create_dirs:
        result["directories"] = setup_directories()

    # Find datasets
    result["datasets"] = get_dataset_paths()

    if verbose:
        print("=" * 50)
        print("Setup complete!")
        print("=" * 50)
        print(f"\nStyleFormer path: {result['styleformer_path']}")
        print("\nNow you can import StyleFormer modules:")
        print("  from src.data import FaceDataModule")
        print("  from src.models import losses, pretrained")
        print("  from src.evaluation import Evaluator")

    return result


# Kaggle-specific configuration class
class KaggleConfig:
    """Configuration helper for Kaggle notebooks."""

    def __init__(self):
        self.working_dir = Path("/kaggle/working")
        self.input_dir = Path("/kaggle/input")
        self.styleformer_path = get_styleformer_path()

    @property
    def output_dir(self) -> Path:
        path = self.working_dir / "outputs"
        path.mkdir(exist_ok=True)
        return path

    @property
    def checkpoint_dir(self) -> Path:
        path = self.working_dir / "checkpoints"
        path.mkdir(exist_ok=True)
        return path

    @property
    def pretrained_dir(self) -> Path:
        path = self.working_dir / "pretrained"
        path.mkdir(exist_ok=True)
        return path

    def get_dataset_path(self, name: str) -> Path | None:
        """Get path to a dataset."""
        datasets = get_dataset_paths()
        return datasets.get(name)

    def get_accelerator(self) -> str:
        """Get PyTorch Lightning accelerator for Kaggle."""
        import torch

        if torch.cuda.is_available():
            return "gpu"
        return "cpu"

    def get_device(self) -> str:
        """Get torch device string."""
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    setup()

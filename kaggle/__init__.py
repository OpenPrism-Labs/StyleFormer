"""Kaggle utilities for StyleFormer.

This module provides helpers for running StyleFormer on Kaggle:
- Environment setup and path configuration
- Dataset discovery
- Pretrained model downloading
"""

from kaggle.setup_kaggle import (
    setup,
    KaggleConfig,
    is_kaggle,
    get_styleformer_path,
    install_dependencies,
    setup_paths,
    setup_directories,
    get_dataset_paths,
    download_pretrained_weights,
)

__all__ = [
    "setup",
    "KaggleConfig",
    "is_kaggle",
    "get_styleformer_path",
    "install_dependencies",
    "setup_paths",
    "setup_directories",
    "get_dataset_paths",
    "download_pretrained_weights",
]

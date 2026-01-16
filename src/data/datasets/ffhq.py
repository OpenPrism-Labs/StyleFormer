"""FFHQ dataset (no attribute labels)."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.datasets.base import BaseFaceDataset


class FFHQDataset(BaseFaceDataset):
    """Flickr-Faces-HQ (FFHQ) dataset.
    
    FFHQ contains 70,000 high-quality face images at 1024x1024 resolution.
    Unlike CelebA, it does not have attribute labels.
    
    Expected directory structure:
        root/
            00000/
                00000.png
                00001.png
                ...
            00001/
                01000.png
                ...
            ... or flat structure with all images
    """
    
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Any | None = None,
        target_transform: Any | None = None,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
    ) -> None:
        """Initialize FFHQ dataset.
        
        Args:
            root: Root directory containing images.
            split: Dataset split - "train", "val", or "test".
            transform: Image transforms.
            target_transform: Not used (FFHQ has no labels).
            train_ratio: Ratio of data for training.
            val_ratio: Ratio of data for validation.
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        self.image_paths: list[Path] = []
        
        super().__init__(root, split, transform, target_transform)
    
    def _find_images(self) -> list[Path]:
        """Find all image files in the dataset directory."""
        extensions = {".png", ".jpg", ".jpeg"}
        images = []
        
        # Check for nested structure (00000/, 00001/, ...)
        subdirs = [d for d in self.root.iterdir() if d.is_dir()]
        
        if subdirs and subdirs[0].name.isdigit():
            # Nested structure
            for subdir in sorted(subdirs):
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in extensions:
                        images.append(img_path)
        else:
            # Flat structure
            for img_path in sorted(self.root.iterdir()):
                if img_path.suffix.lower() in extensions:
                    images.append(img_path)
        
        return images
    
    def _load_data(self) -> None:
        """Load dataset file paths."""
        all_images = self._find_images()
        
        if not all_images:
            raise FileNotFoundError(
                f"No images found in {self.root}. "
                "Expected PNG or JPG files."
            )
        
        # Create splits
        n = len(all_images)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        # Use fixed seed for reproducibility
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        
        if self.split == "train":
            indices = perm[:n_train]
        elif self.split == "val":
            indices = perm[n_train:n_train + n_val]
        else:  # test
            indices = perm[n_train + n_val:]
        
        self.image_paths = [all_images[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample.
        
        Returns:
            Dictionary with:
                - image: Transformed image tensor
                - filename: Image filename
                - index: Sample index
        """
        img_path = self.image_paths[idx]
        image = self.load_image(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "filename": img_path.name,
            "index": idx,
            # FFHQ has no attributes, return empty tensor
            "attributes": torch.tensor([]),
        }
    
    @property
    def has_attributes(self) -> bool:
        return False
    
    @property
    def attribute_names(self) -> list[str]:
        return []

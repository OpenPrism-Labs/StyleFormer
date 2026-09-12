"""FFHQ dataset (no attribute labels)."""

from pathlib import Path
from typing import Any

import torch

from src.data.datasets.base import BaseFaceDataset, split_indices


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
            01000/
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
        seed: int = 42,
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
        self.seed = seed
        split_indices(0, split, train_ratio, val_ratio, seed)

        self.image_paths: list[Path] = []

        super().__init__(root, split, transform, target_transform)

    def _find_images(self) -> list[Path]:
        """Find all image files in the dataset directory."""
        extensions = {".png", ".jpg", ".jpeg"}
        return sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

    def _load_data(self) -> None:
        """Load dataset file paths."""
        all_images = self._find_images()

        if not all_images:
            raise FileNotFoundError(
                f"No images found in {self.root}. "
                "Expected extracted PNG or JPG files; archives and TFRecords are unsupported."
            )

        indices = split_indices(
            len(all_images), self.split, self.train_ratio, self.val_ratio, self.seed
        )

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
            "filename": str(img_path.relative_to(self.root)),
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

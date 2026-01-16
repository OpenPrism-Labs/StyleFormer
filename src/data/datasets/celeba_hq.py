"""CelebA-HQ dataset with multi-attribute support."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.datasets.base import BaseFaceDataset, BasePairedDataset


# CelebA attribute names (40 total)
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young"
]


class CelebAHQDataset(BaseFaceDataset):
    """CelebA-HQ dataset with attribute labels.
    
    Supports:
    - Multi-attribute selection
    - Attribute filtering
    - Train/val/test splits
    
    Expected directory structure:
        root/
            images/           # or img_align_celeba/
                000001.jpg
                000002.jpg
                ...
            list_attr_celeba.txt
            list_eval_partition.txt (optional)
    """
    
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Any | None = None,
        target_transform: Any | None = None,
        selected_attrs: list[str] | None = None,
        filter_attrs: dict[str, int] | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> None:
        """Initialize CelebA-HQ dataset.
        
        Args:
            root: Root directory containing images and attribute files.
            split: Dataset split - "train", "val", or "test".
            transform: Image transforms.
            target_transform: Attribute transforms.
            selected_attrs: List of attribute names to return. None = all.
            filter_attrs: Dict of {attr_name: value} to filter dataset.
                         Example: {"Male": 1, "Young": 1} for young males.
            train_ratio: Ratio of data for training (if no partition file).
            val_ratio: Ratio of data for validation.
        """
        self.selected_attrs = selected_attrs
        self.filter_attrs = filter_attrs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Will be populated by _load_data
        self.filenames: list[str] = []
        self.attrs: torch.Tensor = torch.tensor([])
        self._attr_names: list[str] = []
        self._attr_to_idx: dict[str, int] = {}
        self._selected_indices: list[int] = []
        
        super().__init__(root, split, transform, target_transform)
    
    def _find_image_dir(self) -> Path:
        """Find the image directory."""
        candidates = ["images", "img_align_celeba", "CelebA-HQ", "celeba_hq"]
        for name in candidates:
            path = self.root / name
            if path.exists():
                return path
        # Fallback to root if images are directly there
        return self.root
    
    def _load_attributes(self) -> tuple[list[str], np.ndarray, list[str]]:
        """Load attribute file.
        
        Returns:
            Tuple of (filenames, attributes_array, attribute_names)
        """
        attr_path = self.root / "list_attr_celeba.txt"
        
        if not attr_path.exists():
            # Try alternative locations
            for alt in ["CelebAMask-HQ-attribute-anno.txt", "attributes.txt"]:
                alt_path = self.root / alt
                if alt_path.exists():
                    attr_path = alt_path
                    break
        
        if not attr_path.exists():
            raise FileNotFoundError(
                f"Attribute file not found in {self.root}. "
                "Expected 'list_attr_celeba.txt'"
            )
        
        with open(attr_path, "r") as f:
            lines = f.readlines()
        
        # First line: number of images (skip)
        # Second line: attribute names
        # Rest: filename and attribute values
        num_images = int(lines[0].strip())
        attr_names = lines[1].strip().split()
        
        filenames = []
        attrs = []
        
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filenames.append(parts[0])
            # Convert {-1, 1} to {0, 1}
            attr_values = [(int(x) + 1) // 2 for x in parts[1:]]
            attrs.append(attr_values)
        
        return filenames, np.array(attrs, dtype=np.int64), attr_names
    
    def _load_partition(self) -> dict[str, int] | None:
        """Load partition file if exists.
        
        Returns:
            Dict mapping filename to split (0=train, 1=val, 2=test) or None.
        """
        partition_path = self.root / "list_eval_partition.txt"
        
        if not partition_path.exists():
            return None
        
        partition = {}
        with open(partition_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    partition[parts[0]] = int(parts[1])
        
        return partition
    
    def _load_data(self) -> None:
        """Load dataset metadata."""
        self.image_dir = self._find_image_dir()
        
        # Load attributes
        filenames, attrs, attr_names = self._load_attributes()
        
        self._attr_names = attr_names
        self._attr_to_idx = {name: idx for idx, name in enumerate(attr_names)}
        
        # Load or create partition
        partition = self._load_partition()
        split_map = {"train": 0, "val": 1, "test": 2}
        target_split = split_map[self.split]
        
        if partition is not None:
            # Use official partition
            indices = [
                i for i, fname in enumerate(filenames)
                if partition.get(fname, 0) == target_split
            ]
        else:
            # Create random split based on ratios
            n = len(filenames)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            # Use fixed seed for reproducibility
            rng = np.random.RandomState(42)
            perm = rng.permutation(n)
            
            if self.split == "train":
                indices = perm[:n_train].tolist()
            elif self.split == "val":
                indices = perm[n_train:n_train + n_val].tolist()
            else:  # test
                indices = perm[n_train + n_val:].tolist()
        
        # Apply attribute filters
        if self.filter_attrs:
            filtered_indices = []
            for idx in indices:
                match = True
                for attr_name, attr_value in self.filter_attrs.items():
                    attr_idx = self._attr_to_idx[attr_name]
                    if attrs[idx, attr_idx] != attr_value:
                        match = False
                        break
                if match:
                    filtered_indices.append(idx)
            indices = filtered_indices
        
        # Store filtered data
        self.filenames = [filenames[i] for i in indices]
        self.attrs = torch.from_numpy(attrs[indices])
        
        # Determine which attribute indices to return
        if self.selected_attrs:
            self._selected_indices = [
                self._attr_to_idx[name] for name in self.selected_attrs
            ]
        else:
            self._selected_indices = list(range(len(attr_names)))
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample.
        
        Returns:
            Dictionary with:
                - image: Transformed image tensor
                - attributes: Selected attribute tensor
                - filename: Image filename
                - all_attributes: All 40 attributes (optional)
        """
        filename = self.filenames[idx]
        
        # Load image
        img_path = self.image_dir / filename
        if not img_path.exists():
            # Try without extension or with different extension
            for ext in [".jpg", ".png", ".jpeg"]:
                alt_path = self.image_dir / (Path(filename).stem + ext)
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        image = self.load_image(img_path)
        
        # Get attributes
        all_attrs = self.attrs[idx]
        selected_attrs = all_attrs[self._selected_indices]
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            selected_attrs = self.target_transform(selected_attrs)
        
        return {
            "image": image,
            "attributes": selected_attrs.float(),
            "filename": filename,
            "index": idx,
        }
    
    @property
    def has_attributes(self) -> bool:
        return True
    
    @property
    def attribute_names(self) -> list[str]:
        if self.selected_attrs:
            return self.selected_attrs
        return self._attr_names
    
    def get_attribute_index(self, attr_name: str) -> int:
        """Get index of an attribute in the selected attributes."""
        if self.selected_attrs:
            return self.selected_attrs.index(attr_name)
        return self._attr_to_idx[attr_name]


class CelebAHQPairedDataset(BasePairedDataset):
    """CelebA-HQ dataset that returns paired images for style transfer.
    
    Creates pairs where:
    - Images have opposite values for transfer_attr (e.g., Male)
    - Images optionally match on preserve_attrs (e.g., same age)
    """
    
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Any | None = None,
        transfer_attr: str = "Male",
        preserve_attrs: list[str] | None = None,
        selected_attrs: list[str] | None = None,
        max_pairs: int | None = None,
    ) -> None:
        """Initialize paired dataset.
        
        Args:
            root: Root directory.
            split: Dataset split.
            transform: Image transforms (applied to both images).
            transfer_attr: Attribute to transfer (pairs have opposite values).
            preserve_attrs: Attributes that should match in pairs.
            selected_attrs: Attributes to include in output.
            max_pairs: Maximum number of pairs to create (None = all).
        """
        self.selected_attrs = selected_attrs or [transfer_attr]
        self.max_pairs = max_pairs
        
        # Internal storage
        self._base_dataset: CelebAHQDataset | None = None
        self.pairs: list[tuple[int, int]] = []
        
        super().__init__(root, split, transform, transfer_attr, preserve_attrs)
    
    def _load_data(self) -> None:
        """Load base dataset."""
        self._base_dataset = CelebAHQDataset(
            root=self.root,
            split=self.split,
            transform=None,  # Apply transform in __getitem__
            selected_attrs=self.selected_attrs,
        )
    
    def _create_pairs(self) -> None:
        """Create pairs with opposite transfer attribute values."""
        if self._base_dataset is None:
            return
        
        # Get transfer attribute index
        transfer_idx = self._base_dataset.get_attribute_index(self.transfer_attr)
        
        # Get preserve attribute indices
        preserve_indices = []
        if self.preserve_attrs:
            preserve_indices = [
                self._base_dataset.get_attribute_index(attr)
                for attr in self.preserve_attrs
            ]
        
        # Group images by (transfer_val, preserve_vals)
        groups: dict[tuple, list[int]] = {}
        
        for idx in range(len(self._base_dataset)):
            attrs = self._base_dataset.attrs[idx]
            transfer_val = attrs[self._base_dataset._attr_to_idx[self.transfer_attr]].item()
            
            if preserve_indices:
                # Get values from the full attribute tensor
                preserve_vals = tuple(
                    attrs[self._base_dataset._attr_to_idx[attr]].item()
                    for attr in self.preserve_attrs
                )
            else:
                preserve_vals = ()
            
            key = (transfer_val, preserve_vals)
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        
        # Create pairs: match images with opposite transfer_val, same preserve_vals
        self.pairs = []
        
        for (transfer_val, preserve_vals), indices_a in groups.items():
            opposite_key = (1 - transfer_val, preserve_vals)
            if opposite_key not in groups:
                continue
            
            indices_b = groups[opposite_key]
            
            # Create pairs (limit to avoid explosion)
            n_pairs = min(len(indices_a), len(indices_b))
            if self.max_pairs:
                n_pairs = min(n_pairs, self.max_pairs // 2)
            
            for i in range(n_pairs):
                self.pairs.append((indices_a[i], indices_b[i]))
                self.pairs.append((indices_b[i], indices_a[i]))  # Both directions
        
        # Shuffle pairs
        rng = np.random.RandomState(42)
        rng.shuffle(self.pairs)
        
        if self.max_pairs and len(self.pairs) > self.max_pairs:
            self.pairs = self.pairs[:self.max_pairs]
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a pair of images.
        
        Returns:
            Dictionary with source and target images/attributes.
        """
        source_idx, target_idx = self.pairs[idx]
        
        source_data = self._base_dataset[source_idx]
        target_data = self._base_dataset[target_idx]
        
        source_img = source_data["image"]
        target_img = target_data["image"]
        
        # Apply transform to PIL images
        if self.transform:
            # Need to reload as PIL for transform
            source_path = self._base_dataset.image_dir / self._base_dataset.filenames[source_idx]
            target_path = self._base_dataset.image_dir / self._base_dataset.filenames[target_idx]
            
            source_img = self.transform(Image.open(source_path).convert("RGB"))
            target_img = self.transform(Image.open(target_path).convert("RGB"))
        
        return {
            "source_image": source_img,
            "target_image": target_img,
            "source_attributes": source_data["attributes"],
            "target_attributes": target_data["attributes"],
            "source_filename": source_data["filename"],
            "target_filename": target_data["filename"],
        }

"""Abstract base class for face datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseFaceDataset(Dataset, ABC):
    """Abstract base class for face image datasets.
    
    All face datasets should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Any | None = None,
        target_transform: Any | None = None,
    ) -> None:
        """Initialize base dataset.
        
        Args:
            root: Root directory of the dataset.
            split: Dataset split - "train", "val", or "test".
            transform: Transform to apply to images.
            target_transform: Transform to apply to targets/attributes.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self._validate_split(split)
        self._load_data()
    
    def _validate_split(self, split: str) -> None:
        """Validate split name."""
        valid_splits = {"train", "val", "test"}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset metadata (file paths, attributes, etc.)."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with at least 'image' key. May also contain
            'attributes', 'filename', etc.
        """
        pass
    
    def load_image(self, path: str | Path) -> Image.Image:
        """Load and convert image to RGB.
        
        Args:
            path: Path to image file.
            
        Returns:
            PIL Image in RGB mode.
        """
        return Image.open(path).convert("RGB")
    
    @property
    def has_attributes(self) -> bool:
        """Whether dataset has attribute labels."""
        return False
    
    @property
    def attribute_names(self) -> list[str]:
        """List of attribute names (empty if no attributes)."""
        return []
    
    @property
    def num_attributes(self) -> int:
        """Number of attributes."""
        return len(self.attribute_names)


class BasePairedDataset(Dataset, ABC):
    """Abstract base class for paired face datasets.
    
    Used for style transfer where we need pairs of images
    with different attributes.
    """
    
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Any | None = None,
        transfer_attr: str = "gender",
        preserve_attrs: list[str] | None = None,
    ) -> None:
        """Initialize paired dataset.
        
        Args:
            root: Root directory of the dataset.
            split: Dataset split.
            transform: Transform to apply to both images.
            transfer_attr: Attribute to transfer (images will have opposite values).
            preserve_attrs: Attributes that should match between paired images.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.transfer_attr = transfer_attr
        self.preserve_attrs = preserve_attrs or []
        
        self._load_data()
        self._create_pairs()
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset metadata."""
        pass
    
    @abstractmethod
    def _create_pairs(self) -> None:
        """Create pairs of images with opposite transfer_attr."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of pairs."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a pair of images.
        
        Args:
            idx: Pair index.
            
        Returns:
            Dictionary with 'source_image', 'target_image', 
            'source_attrs', 'target_attrs'.
        """
        pass

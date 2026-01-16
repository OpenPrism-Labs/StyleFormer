"""PyTorch Lightning DataModule for face datasets."""

from pathlib import Path
from typing import Any

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.datasets import CelebAHQDataset, CelebAHQPairedDataset, FFHQDataset
from src.data.transforms import get_train_transforms, get_val_transforms


class FaceDataModule(L.LightningDataModule):
    """Lightning DataModule for face transformation experiments.
    
    Supports:
    - CelebA-HQ with multi-attribute labels
    - FFHQ (unconditional)
    - Paired datasets for style transfer
    - Configurable via Hydra
    """
    
    def __init__(
        self,
        name: str = "celeba_hq",
        root: str | Path = "./data/celeba_hq",
        image_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        persistent_workers: bool = True,
        # Attribute configuration
        selected_attrs: list[str] | None = None,
        filter_attrs: dict[str, int] | None = None,
        # Pairing configuration
        pairing_enabled: bool = False,
        pairing_mode: str = "opposite",
        transfer_attr: str = "Male",
        preserve_attrs: list[str] | None = None,
        # Split ratios
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        # Augmentation
        train_augmentation: dict[str, bool] | None = None,
        val_augmentation: dict[str, bool] | None = None,
    ) -> None:
        """Initialize FaceDataModule.
        
        Args:
            name: Dataset name ("celeba_hq" or "ffhq").
            root: Root directory of the dataset.
            image_size: Target image size.
            batch_size: Batch size for dataloaders.
            num_workers: Number of dataloader workers.
            pin_memory: Pin memory for faster GPU transfer.
            drop_last: Drop last incomplete batch.
            persistent_workers: Keep workers alive between epochs.
            selected_attrs: Attributes to include (CelebA-HQ only).
            filter_attrs: Filter dataset by attributes.
            pairing_enabled: Enable paired sampling for style transfer.
            pairing_mode: Pairing mode ("opposite", "random", "matched").
            transfer_attr: Attribute to transfer in pairs.
            preserve_attrs: Attributes to preserve in pairs.
            train_ratio: Training data ratio.
            val_ratio: Validation data ratio.
            train_augmentation: Training augmentation settings.
            val_augmentation: Validation augmentation settings.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.name = name
        self.root = Path(root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        
        self.selected_attrs = selected_attrs
        self.filter_attrs = filter_attrs
        
        self.pairing_enabled = pairing_enabled
        self.pairing_mode = pairing_mode
        self.transfer_attr = transfer_attr
        self.preserve_attrs = preserve_attrs
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Default augmentation settings
        self.train_augmentation = train_augmentation or {
            "horizontal_flip": True,
            "color_jitter": True,
            "random_crop": False,
        }
        self.val_augmentation = val_augmentation or {
            "horizontal_flip": False,
            "color_jitter": False,
            "random_crop": False,
        }
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "FaceDataModule":
        """Create DataModule from Hydra config.
        
        Args:
            cfg: Hydra configuration with 'data' section.
            
        Returns:
            Configured FaceDataModule instance.
        """
        data_cfg = cfg.data
        
        return cls(
            name=data_cfg.get("name", "celeba_hq"),
            root=data_cfg.get("root", "./data/celeba_hq"),
            image_size=data_cfg.get("image_size", 256),
            batch_size=cfg.dataloader.get("batch_size", 16),
            num_workers=cfg.dataloader.get("num_workers", 4),
            pin_memory=cfg.dataloader.get("pin_memory", True),
            drop_last=cfg.dataloader.get("drop_last", True),
            persistent_workers=cfg.dataloader.get("persistent_workers", True),
            selected_attrs=data_cfg.attributes.get("selected"),
            filter_attrs=data_cfg.attributes.get("filter"),
            pairing_enabled=data_cfg.pairing.get("enabled", False),
            pairing_mode=data_cfg.pairing.get("mode", "opposite"),
            transfer_attr=data_cfg.pairing.get("transfer_attr", "Male"),
            preserve_attrs=data_cfg.pairing.get("preserve_attrs"),
            train_ratio=data_cfg.splits.get("train", 0.8),
            val_ratio=data_cfg.splits.get("val", 0.1),
            train_augmentation=dict(data_cfg.augmentation.get("train", {})),
            val_augmentation=dict(data_cfg.augmentation.get("val", {})),
        )
    
    def prepare_data(self) -> None:
        """Download or prepare data (called on single process)."""
        # Check if dataset exists
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}. "
                f"Please download the {self.name} dataset first."
            )
    
    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.
        
        Args:
            stage: "fit", "validate", "test", or "predict".
        """
        # Build transforms
        train_transform = get_train_transforms(
            image_size=self.image_size,
            **self.train_augmentation,
        )
        val_transform = get_val_transforms(
            image_size=self.image_size,
        )
        
        if stage == "fit" or stage is None:
            if self.name == "celeba_hq":
                if self.pairing_enabled:
                    self.train_dataset = CelebAHQPairedDataset(
                        root=self.root,
                        split="train",
                        transform=train_transform,
                        transfer_attr=self.transfer_attr,
                        preserve_attrs=self.preserve_attrs,
                        selected_attrs=self.selected_attrs,
                    )
                    self.val_dataset = CelebAHQPairedDataset(
                        root=self.root,
                        split="val",
                        transform=val_transform,
                        transfer_attr=self.transfer_attr,
                        preserve_attrs=self.preserve_attrs,
                        selected_attrs=self.selected_attrs,
                    )
                else:
                    self.train_dataset = CelebAHQDataset(
                        root=self.root,
                        split="train",
                        transform=train_transform,
                        selected_attrs=self.selected_attrs,
                        filter_attrs=self.filter_attrs,
                        train_ratio=self.train_ratio,
                        val_ratio=self.val_ratio,
                    )
                    self.val_dataset = CelebAHQDataset(
                        root=self.root,
                        split="val",
                        transform=val_transform,
                        selected_attrs=self.selected_attrs,
                        filter_attrs=self.filter_attrs,
                        train_ratio=self.train_ratio,
                        val_ratio=self.val_ratio,
                    )
            elif self.name == "ffhq":
                self.train_dataset = FFHQDataset(
                    root=self.root,
                    split="train",
                    transform=train_transform,
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                )
                self.val_dataset = FFHQDataset(
                    root=self.root,
                    split="val",
                    transform=val_transform,
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                )
            else:
                raise ValueError(f"Unknown dataset: {self.name}")
        
        if stage == "test" or stage is None:
            if self.name == "celeba_hq":
                self.test_dataset = CelebAHQDataset(
                    root=self.root,
                    split="test",
                    transform=val_transform,
                    selected_attrs=self.selected_attrs,
                    filter_attrs=self.filter_attrs,
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                )
            elif self.name == "ffhq":
                self.test_dataset = FFHQDataset(
                    root=self.root,
                    split="test",
                    transform=val_transform,
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
    @property
    def num_attributes(self) -> int:
        """Number of selected attributes."""
        if self.selected_attrs:
            return len(self.selected_attrs)
        if self.name == "celeba_hq":
            return 40  # All CelebA attributes
        return 0
    
    @property
    def attribute_names(self) -> list[str]:
        """Names of selected attributes."""
        if self.selected_attrs:
            return self.selected_attrs
        return []

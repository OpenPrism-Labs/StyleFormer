"""Lightning data module for extracted face image datasets."""

from pathlib import Path
from typing import Any

import lightning as L
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.datasets import CelebAHQDataset, CelebAHQPairedDataset, FFHQDataset
from src.data.datasets.base import split_indices
from src.data.datasets.celeba_hq import CELEBA_ATTRIBUTES
from src.data.transforms import get_train_transforms


class FaceDataModule(L.LightningDataModule):
    """Create consistent train/validation/test partitions and predict on test data."""

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
        selected_attrs: list[str] | None = None,
        filter_attrs: dict[str, int] | None = None,
        pairing_enabled: bool = False,
        pairing_mode: str = "opposite",
        transfer_attr: str = "Male",
        preserve_attrs: list[str] | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        train_augmentation: dict[str, Any] | None = None,
        val_augmentation: dict[str, Any] | None = None,
        seed: int = 42,
        split_seed: int | None = None,
        attribute_file: str | Path | None = None,
        mapping_file: str | Path | None = None,
        partition_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        if name not in {"celeba_hq", "ffhq"}:
            raise ValueError(f"Unknown dataset: {name}")
        if pairing_mode not in {"opposite", "matched", "random"}:
            raise ValueError(f"Unknown pairing mode: {pairing_mode}")
        if name == "ffhq" and (
            pairing_enabled
            or selected_attrs
            or filter_attrs
            or attribute_file
            or mapping_file
            or partition_file
        ):
            raise ValueError(
                "FFHQ is unlabeled: attribute metadata, filtering, and pairing are unsupported"
            )
        if batch_size <= 0 or num_workers < 0 or image_size <= 0:
            raise ValueError(
                "batch_size and image_size must be positive; num_workers must be nonnegative"
            )
        self.save_hyperparameters()
        self.name, self.root, self.image_size = name, Path(root), image_size
        self.batch_size, self.num_workers = batch_size, num_workers
        self.pin_memory, self.drop_last = pin_memory, drop_last
        self.persistent_workers = persistent_workers
        self.selected_attrs = None if selected_attrs is None else list(selected_attrs)
        self.filter_attrs = filter_attrs
        self.pairing_enabled, self.pairing_mode = pairing_enabled, pairing_mode
        self.transfer_attr, self.preserve_attrs = transfer_attr, preserve_attrs
        self.train_ratio, self.val_ratio, self.seed = train_ratio, val_ratio, seed
        self.split_seed = seed if split_seed is None else split_seed
        split_indices(0, "train", train_ratio, val_ratio, self.split_seed)
        self.attribute_file, self.mapping_file, self.partition_file = (
            attribute_file,
            mapping_file,
            partition_file,
        )
        self.train_augmentation = (
            dict(train_augmentation)
            if train_augmentation is not None
            else {
                "horizontal_flip": True,
                "color_jitter": True,
                "random_crop": False,
            }
        )
        self.val_augmentation = dict(val_augmentation) if val_augmentation is not None else {}
        self.train_dataset = self.val_dataset = self.test_dataset = self.predict_dataset = None

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "FaceDataModule":
        """Construct from optional dataset sections and shared dataloader settings."""
        data = cfg.data
        loader = cfg.get("dataloader", {})
        attributes = data.get("attributes") or {}
        pairing = data.get("pairing") or {}
        splits = data.get("splits") or {}
        augmentation = data.get("augmentation") or {}
        definitions = cfg.get("attributes", {}).get("definitions", {})

        def annotation_name(name: str) -> str:
            return definitions.get(name, {}).get("celeba_attr", name)

        selected = attributes.get("selected")
        filters = attributes.get("filter")
        preserve = pairing.get("preserve_attrs")
        return cls(
            name=data.get("name", "celeba_hq"),
            root=to_absolute_path(data.get("root", "./data/celeba_hq")),
            image_size=data.get("image_size", 256),
            batch_size=loader.get("batch_size", 16),
            num_workers=loader.get("num_workers", 4),
            pin_memory=loader.get("pin_memory", True),
            drop_last=loader.get("drop_last", True),
            persistent_workers=loader.get("persistent_workers", True),
            selected_attrs=None
            if selected is None
            else [annotation_name(name) for name in selected],
            filter_attrs=None
            if filters is None
            else {annotation_name(name): value for name, value in filters.items()},
            pairing_enabled=pairing.get("enabled", False),
            pairing_mode=pairing.get("mode", "opposite"),
            transfer_attr=annotation_name(pairing.get("transfer_attr", "Male")),
            preserve_attrs=None
            if preserve is None
            else [annotation_name(name) for name in preserve],
            train_ratio=splits.get("train", 0.8),
            val_ratio=splits.get("val", 0.1),
            train_augmentation=augmentation.get("train"),
            val_augmentation=augmentation.get("val"),
            seed=data.get("seed", cfg.get("seed", cfg.get("experiment", {}).get("seed", 42))),
            split_seed=data.get("split_seed"),
            attribute_file=data.get("attribute_file"),
            mapping_file=data.get("mapping_file"),
            partition_file=data.get("partition_file"),
        )

    def prepare_data(self) -> None:
        """Datasets must be downloaded and extracted separately."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

    def _dataset(self, split: str):
        options = self.train_augmentation if split == "train" else self.val_augmentation
        # An explicit empty augmentation mapping means no random augmentation.
        augmentation = {
            "horizontal_flip": False,
            "color_jitter": False,
            "random_crop": False,
            **options,
        }
        common = dict(
            root=self.root,
            split=split,
            transform=get_train_transforms(self.image_size, **augmentation),
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.split_seed,
        )
        if self.name == "ffhq":
            return FFHQDataset(**common)
        common.update(
            selected_attrs=self.selected_attrs,
            filter_attrs=self.filter_attrs,
            attribute_file=self.attribute_file,
            mapping_file=self.mapping_file,
            partition_file=self.partition_file,
        )
        if self.pairing_enabled:
            return CelebAHQPairedDataset(
                **common,
                pairing_mode=self.pairing_mode,
                transfer_attr=self.transfer_attr,
                preserve_attrs=self.preserve_attrs,
            )
        return CelebAHQDataset(**common)

    def setup(self, stage: str | None = None) -> None:
        """Support independent Lightning stages without rebuilding existing datasets."""
        if stage not in {None, "fit", "validate", "test", "predict"}:
            raise ValueError(f"Unknown stage: {stage}")
        if stage in {None, "fit"} and self.train_dataset is None:
            self.train_dataset = self._dataset("train")
        if stage in {None, "fit", "validate"} and self.val_dataset is None:
            self.val_dataset = self._dataset("val")
        if stage in {None, "test", "predict"} and self.test_dataset is None:
            self.test_dataset = self._dataset("test")
        if stage in {None, "predict"}:
            self.predict_dataset = self.test_dataset

    def _loader(self, dataset, training: bool = False) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() for this dataloader's stage first")
        if training and not len(dataset):
            raise ValueError(
                "Training split has no eligible samples; check splits, filters, and pairing constraints"
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=training,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last if training else False,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, training=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_dataset)

    @property
    def num_attributes(self) -> int:
        return len(self.attribute_names)

    @property
    def attribute_names(self) -> list[str]:
        if self.name == "ffhq":
            return []
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None:
                return dataset.attribute_names
        if self.selected_attrs is not None:
            return self.selected_attrs
        return [self.transfer_attr] if self.pairing_enabled else list(CELEBA_ATTRIBUTES)

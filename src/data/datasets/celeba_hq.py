"""CelebA and CelebA-HQ image datasets with explicit annotation alignment."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.datasets.base import BaseFaceDataset, BasePairedDataset, split_indices
from src.data.transforms import PairedTransform

CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


class CelebAHQDataset(BaseFaceDataset):
    """Load extracted images with CelebA-format attribute text files.

    HQ-indexed annotations (CelebAMask-HQ-attribute-anno.txt) align directly.
    Original list_attr_celeba.txt requires the HQ-to-CelebA mapping for HQ
    images. Original six-digit CelebA filenames also work without a mapping.
    Relative metadata paths are resolved against root. Official partitions
    use original CelebA filenames; HQ annotations therefore also need the
    mapping when an original partition file is supplied.
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
        seed: int = 42,
        attribute_file: str | Path | None = None,
        mapping_file: str | Path | None = None,
        partition_file: str | Path | None = None,
    ) -> None:
        self.selected_attrs = None if selected_attrs is None else list(selected_attrs)
        self.filter_attrs = filter_attrs or {}
        self.train_ratio, self.val_ratio, self.seed = train_ratio, val_ratio, seed
        self.attribute_file, self.mapping_file = attribute_file, mapping_file
        self.partition_file = partition_file
        split_indices(0, split, train_ratio, val_ratio, seed)
        super().__init__(root, split, transform, target_transform)

    def _metadata_path(self, explicit: str | Path | None, names: tuple[str, ...]) -> Path | None:
        if explicit is not None:
            path = Path(explicit)
            path = path if path.is_absolute() else self.root / path
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        return next((self.root / name for name in names if (self.root / name).is_file()), None)

    def _find_image_dir(self) -> Path:
        for name in ("images", "CelebA-HQ-img", "img_align_celeba", "CelebA-HQ", "celeba_hq"):
            if (self.root / name).is_dir():
                return self.root / name
        return self.root

    def _load_attributes(self) -> tuple[dict[str, list[int]], list[str], bool]:
        path = self._metadata_path(
            self.attribute_file,
            (
                "CelebAMask-HQ-attribute-anno.txt",
                "list_attr_celeba.txt",
                "attributes.txt",
            ),
        )
        if path is None:
            raise FileNotFoundError(f"No CelebA-format attribute text file in {self.root}")
        lines = path.read_text().splitlines()
        if len(lines) < 2 or not lines[0].strip().isdigit():
            raise ValueError(
                f"Unsupported attribute format in {path}: expected count, names, then filename and -1/1 values"
            )
        names = lines[1].split()
        if not names or len(set(names)) != len(names):
            raise ValueError(f"Invalid attribute names in {path}")
        records = {}
        for line in lines[2:]:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != len(names) + 1 or fields[0] in records:
                raise ValueError(f"Invalid or duplicate attribute row: {line}")
            values = [int(value) for value in fields[1:]]
            if not set(values) <= {-1, 1}:
                raise ValueError(f"Unsupported attribute encoding in {path}: expected -1/1")
            records[fields[0]] = [(value + 1) // 2 for value in values]
        if len(records) != int(lines[0]):
            raise ValueError(f"Attribute count does not match rows in {path}")
        return records, names, path.name == "list_attr_celeba.txt"

    def _load_mapping(self) -> dict[int, str]:
        path = self._metadata_path(
            self.mapping_file,
            (
                "CelebA-HQ-to-CelebA-mapping.txt",
                "image_list.txt",
            ),
        )
        if path is None:
            return {}
        lines = path.read_text().splitlines()
        if not lines:
            raise ValueError(f"Empty mapping file: {path}")
        header = lines[0].split()
        if "idx" not in header or "orig_file" not in header:
            raise ValueError(
                f"Unsupported mapping format in {path}: expected idx and orig_file columns"
            )
        mapping = {}
        for line in lines[1:]:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != len(header):
                raise ValueError(f"Invalid mapping row: {line}")
            index = int(fields[header.index("idx")])
            if index in mapping:
                raise ValueError(f"Duplicate HQ index {index} in {path}")
            mapping[index] = fields[header.index("orig_file")]
        return mapping

    def _load_partition(self) -> dict[str, int] | None:
        path = self._metadata_path(self.partition_file, ("list_eval_partition.txt",))
        if path is None:
            return None
        partition = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2 or fields[1] not in {"0", "1", "2"} or fields[0] in partition:
                raise ValueError(f"Invalid partition row: {line}")
            partition[fields[0]] = int(fields[1])
        return partition

    def _load_data(self) -> None:
        self.image_dir = self._find_image_dir()
        records, self._attr_names, original_attrs = self._load_attributes()
        self._attr_to_idx = {name: idx for idx, name in enumerate(self._attr_names)}
        requested = self._attr_names if self.selected_attrs is None else self.selected_attrs
        unknown = (set(requested) | set(self.filter_attrs)) - self._attr_to_idx.keys()
        if unknown:
            raise ValueError(f"Unknown attributes: {sorted(unknown)}")
        if any(value not in (0, 1) for value in self.filter_attrs.values()):
            raise ValueError("Attribute filters must use binary 0/1 values")
        self._selected_indices = [self._attr_to_idx[name] for name in requested]
        mapping = self._load_mapping()
        partition = self._load_partition()
        paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not paths:
            raise FileNotFoundError(
                f"No extracted JPG/PNG images in {self.image_dir}; archives and TFRecords are unsupported"
            )
        # Match extensions independently, but never confuse HQ indices with original IDs.
        by_stem = {}
        for filename in records:
            stem = Path(filename).stem
            if stem in by_stem:
                raise ValueError(f"Ambiguous annotation stem: {stem}")
            by_stem[stem] = filename
        filenames, attrs, partition_keys = [], [], []
        for path in paths:
            original = mapping.get(int(path.stem)) if path.stem.isdigit() else None
            if original_attrs and mapping:
                if original is None:
                    raise ValueError(f"Missing HQ mapping for {path.name}")
                key = original
            else:
                if original_attrs and not (
                    len(path.stem) == 6 and path.stem.isdigit() and int(path.stem) > 0
                ):
                    raise ValueError(
                        "HQ images with original CelebA attributes require an HQ-to-CelebA mapping file"
                    )
                key = by_stem.get(path.stem, path.name)
            if key not in records:
                raise ValueError(f"No attributes for image {path.name} (annotation key {key})")
            filenames.append(str(path.relative_to(self.image_dir)))
            attrs.append(records[key])
            partition_keys.append(original or key)
        values = np.asarray(attrs, dtype=np.int64)
        if partition is None:
            indices = split_indices(
                len(filenames), self.split, self.train_ratio, self.val_ratio, self.seed
            ).tolist()
        else:
            missing = set(partition_keys) - partition.keys()
            if missing:
                raise ValueError(
                    f"Partition lacks {len(missing)} image IDs; HQ images need original-ID mapping for CelebA partitions"
                )
            target = {"train": 0, "val": 1, "test": 2}[self.split]
            indices = [i for i, key in enumerate(partition_keys) if partition[key] == target]
        indices = [
            i
            for i in indices
            if all(
                values[i, self._attr_to_idx[name]] == value
                for name, value in self.filter_attrs.items()
            )
        ]
        self.filenames = [filenames[i] for i in indices]
        self.attrs = torch.from_numpy(values[indices])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        filename = self.filenames[idx]
        image = self.load_image(self.image_dir / filename)
        attributes = self.attrs[idx, self._selected_indices].float()
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            attributes = self.target_transform(attributes)
        return {"image": image, "attributes": attributes, "filename": filename, "index": idx}

    @property
    def has_attributes(self) -> bool:
        return True

    @property
    def attribute_names(self) -> list[str]:
        return self._attr_names if self.selected_attrs is None else self.selected_attrs

    def get_attribute_index(self, attr_name: str) -> int:
        """Return the index in the selected output attributes."""
        return self.attribute_names.index(attr_name)


class CelebAHQPairedDataset(BasePairedDataset):
    """Deterministic distinct pairs within one split.

    opposite/matched both change transfer_attr and preserve preserve_attrs;
    random chooses distinct partners without attribute constraints.
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
        pairing_mode: str = "opposite",
        filter_attrs: dict[str, int] | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        attribute_file: str | Path | None = None,
        mapping_file: str | Path | None = None,
        partition_file: str | Path | None = None,
    ) -> None:
        if pairing_mode not in {"opposite", "matched", "random"}:
            raise ValueError(f"Unknown pairing mode: {pairing_mode}")
        if max_pairs is not None and max_pairs < 0:
            raise ValueError("max_pairs must be nonnegative")
        self.selected_attrs = [transfer_attr] if selected_attrs is None else list(selected_attrs)
        self.max_pairs, self.pairing_mode, self.seed = max_pairs, pairing_mode, seed
        self._base_options: dict[str, Any] = dict(
            filter_attrs=filter_attrs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            attribute_file=attribute_file,
            mapping_file=mapping_file,
            partition_file=partition_file,
        )
        super().__init__(root, split, transform, transfer_attr, preserve_attrs)

    def _load_data(self) -> None:
        self._base_dataset = CelebAHQDataset(
            self.root, self.split, selected_attrs=self.selected_attrs, **self._base_options
        )

    def _create_pairs(self) -> None:
        base = self._base_dataset
        rng = np.random.RandomState(self.seed)
        self.pairs = []
        if self.pairing_mode == "random":
            indices = rng.permutation(len(base)).tolist()
            if len(indices) > 1:
                self.pairs = list(zip(indices, indices[1:] + indices[:1], strict=True))
        else:
            unknown = {self.transfer_attr, *self.preserve_attrs} - base._attr_to_idx.keys()
            if unknown:
                raise ValueError(f"Unknown pairing attributes: {sorted(unknown)}")
            if self.transfer_attr in self.preserve_attrs:
                raise ValueError("The transfer attribute cannot also be preserved")
            groups: dict[tuple[int, ...], tuple[list[int], list[int]]] = {}
            for idx, row in enumerate(base.attrs.tolist()):
                key = tuple(row[base._attr_to_idx[name]] for name in self.preserve_attrs)
                groups.setdefault(key, ([], []))[row[base._attr_to_idx[self.transfer_attr]]].append(
                    idx
                )
            for negative, positive in groups.values():
                if not negative or not positive:
                    continue
                rng.shuffle(negative)
                rng.shuffle(positive)
                # Every eligible source participates, without duplicate directed pairs.
                self.pairs.extend(
                    (source, positive[i % len(positive)]) for i, source in enumerate(negative)
                )
                self.pairs.extend(
                    (source, negative[i % len(negative)]) for i, source in enumerate(positive)
                )
        rng.shuffle(self.pairs)
        if self.max_pairs is not None:
            self.pairs = self.pairs[: self.max_pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source_idx, target_idx = self.pairs[idx]
        source = self._base_dataset[source_idx]
        target = self._base_dataset[target_idx]
        source_img, target_img = source["image"], target["image"]
        if isinstance(self.transform, PairedTransform):
            source_img, target_img = self.transform(source_img, target_img)
        elif self.transform is not None:
            source_img, target_img = self.transform(source_img), self.transform(target_img)
        return {
            "source_image": source_img,
            "target_image": target_img,
            "source_attributes": source["attributes"],
            "target_attributes": target["attributes"],
            "source_filename": source["filename"],
            "target_filename": target["filename"],
        }

    @property
    def attribute_names(self) -> list[str]:
        return self._base_dataset.attribute_names

    @property
    def num_attributes(self) -> int:
        return len(self.attribute_names)

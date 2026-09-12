"""Regression coverage for annotation identity, partitioning, and paired loading."""

import pytest
import torch
from PIL import Image

from src.data.datamodule import FaceDataModule
from src.data.datasets import CelebAHQDataset, CelebAHQPairedDataset, FFHQDataset
from src.data.transforms import PairedTransform, denormalize, get_val_transforms


def _hq(root, count=6):
    images = root / "CelebA-HQ-img"
    images.mkdir()
    rows = []
    mapping = ["idx orig_idx orig_file"]
    partitions = []
    for idx in range(count):
        Image.new("RGB", (8, 8), (idx * 20, 0, 0)).save(images / f"{idx}.png")
        original = f"{count - idx:06d}.jpg"
        rows.append(f"{original} {1 if idx % 2 else -1} 1")
        mapping.append(f"{idx} {count - idx - 1} {original}")
        partitions.append(f"{original} {idx // 2 % 3}")
    (root / "list_attr_celeba.txt").write_text(f"{count}\nMale Young\n" + "\n".join(rows))
    (root / "CelebA-HQ-to-CelebA-mapping.txt").write_text("\n".join(mapping))
    (root / "list_eval_partition.txt").write_text("\n".join(partitions))


def test_hq_maps_labels_and_official_partitions_to_original_ids(tmp_path):
    _hq(tmp_path)
    for split, expected in [
        ("train", {"0.png", "1.png"}),
        ("val", {"2.png", "3.png"}),
        ("test", {"4.png", "5.png"}),
    ]:
        dataset = CelebAHQDataset(tmp_path, split, selected_attrs=["Male"])
        assert set(dataset.filenames) == expected
        for sample in dataset:
            assert sample["attributes"].item() == int(sample["filename"].split(".")[0]) % 2
    (tmp_path / "CelebA-HQ-to-CelebA-mapping.txt").unlink()
    with pytest.raises(ValueError, match="mapping"):
        CelebAHQDataset(tmp_path)


def test_missing_official_partition_is_not_silently_training(tmp_path):
    _hq(tmp_path)
    (tmp_path / "list_eval_partition.txt").write_text("000006.jpg 0\n")
    with pytest.raises(ValueError, match="Partition lacks"):
        CelebAHQDataset(tmp_path)


def test_pairs_use_unselected_preserved_attributes_and_decode_once(tmp_path):
    _hq(tmp_path)
    dataset = CelebAHQPairedDataset(
        tmp_path, selected_attrs=["Male"], preserve_attrs=["Young"], transform=get_val_transforms(8)
    )
    assert set(dataset.pairs) == {(0, 1), (1, 0)}
    sample = dataset[0]
    assert sample["source_attributes"].item() != sample["target_attributes"].item()
    assert sample["source_image"].shape == (3, 8, 8)
    assert len(CelebAHQPairedDataset(tmp_path, max_pairs=1)) == 1
    assert len(CelebAHQPairedDataset(tmp_path, max_pairs=0)) == 0


def test_recursive_ffhq_splits_are_reproducible_disjoint_and_collatable(tmp_path):
    for idx in range(20):
        directory = tmp_path / ("images/00000" if idx % 2 else "metadata-neighbor")
        directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(directory / f"{idx}.PNG")
    datasets = [
        FFHQDataset(tmp_path, split, train_ratio=0.6, val_ratio=0.2, seed=7)
        for split in ("train", "val", "test")
    ]
    paths = [set(dataset.image_paths) for dataset in datasets]
    assert [len(partition) for partition in paths] == [12, 4, 4]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    assert len(set.union(*paths)) == 20
    assert (
        datasets[0].image_paths
        == FFHQDataset(tmp_path, train_ratio=0.6, val_ratio=0.2, seed=7).image_paths
    )
    module = FaceDataModule(
        name="ffhq",
        root=tmp_path,
        image_size=8,
        batch_size=2,
        num_workers=0,
        train_ratio=0.6,
        val_ratio=0.2,
        train_augmentation={},
    )
    module.setup("validate")
    assert module.train_dataset is None
    assert next(iter(module.val_dataloader()))["attributes"].shape == (2, 0)
    module.setup("predict")
    assert next(iter(module.predict_dataloader()))["image"].shape == (2, 3, 8, 8)
    module.setup("fit")
    assert next(iter(module.train_dataloader()))["image"].shape == (2, 3, 8, 8)


def test_paired_flip_uses_shared_supported_random_sampling():
    image = Image.new("RGB", (8, 8))
    image.putpixel((0, 0), (255, 0, 0))
    transform = PairedTransform(8)
    for seed in (0, 1):
        torch.manual_seed(seed)
        first, second = transform(image, image)
        torch.testing.assert_close(first, second)
        assert denormalize(first).max().item() == 1

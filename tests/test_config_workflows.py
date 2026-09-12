"""Regression checks for user-facing composition and dataset verification."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import MissingConfigException
from PIL import Image

from scripts.prepare_data import main, verify_dataset

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.mark.parametrize("overrides", [[], ["experiments=multi_attr"]])
def test_composed_attributes_load_real_paired_batches(tmp_path, overrides):
    from src.data import FaceDataModule

    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for index in range(40):
        Image.new("RGB", (8, 8)).save(images / f"{index}.jpg")
        rows.append(f"{index}.jpg {1 if index % 2 else -1} {1 if index % 4 < 2 else -1}\n")
    (tmp_path / "CelebAMask-HQ-attribute-anno.txt").write_text("40\nMale Young\n" + "".join(rows))
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="base",
            overrides=overrides
            + [
                f"data.root={tmp_path}",
                "dataloader.num_workers=0",
                "dataloader.batch_size=2",
                "dataloader.pin_memory=false",
            ],
        )
        dm = FaceDataModule.from_config(cfg)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
    # Config uses age/gender names; batches must contain the corresponding labels.
    source, target = batch["source_attributes"], batch["target_attributes"]
    age, gender = dm.attribute_names.index("Young"), dm.attribute_names.index("Male")
    assert (source[:, age] == target[:, age]).all()
    assert (source[:, gender] != target[:, gender]).all()


def test_unknown_experiment_fails_instead_of_silently_using_baseline():
    with (
        initialize_config_dir(version_base=None, config_dir=CONFIG_DIR),
        pytest.raises(MissingConfigException),
    ):
        compose(config_name="base", overrides=["experiments=missing"])


def test_verify_empty_celeba_directory_fails(tmp_path):
    (tmp_path / "images").mkdir()
    assert not verify_dataset(tmp_path, "celeba_hq")


def test_verify_celeba_images_without_labels_fails(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8)).save(images / "0.jpg")
    assert not verify_dataset(tmp_path, "celeba_hq")


def test_verify_nested_ffhq_checks_image_decoding(tmp_path):
    images = tmp_path / "images1024x1024" / "00000"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(images / "00000.PNG")
    assert verify_dataset(tmp_path, "ffhq")
    (images / "00001.jpg").write_text("not an image")
    assert not verify_dataset(tmp_path, "ffhq")


def test_verify_cli_reports_failure_status(tmp_path):
    assert main(["--dataset", "all", "--data-dir", str(tmp_path), "--verify"]) == 1

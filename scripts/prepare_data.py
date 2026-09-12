"""Manual dataset setup instructions and loader-compatible integrity checks."""

import argparse
import logging
import os
import shlex
from pathlib import Path

from rich.logging import RichHandler

log = logging.getLogger(__name__)


def download_celeba_hq(output_dir: Path) -> None:
    """Show manual setup instructions; no data is downloaded."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("CelebA-HQ requires manual setup; no data has been downloaded.")
    log.info("Official source: https://github.com/tkarras/progressive_growing_of_gans")
    log.info("Extract decoded JPG/PNG images into %s/images/", output_dir)
    log.info("TFRecords and archives are not supported directly.")
    log.info("Provide CelebAMask-HQ-attribute-anno.txt for HQ-indexed labels, or")
    log.info("list_attr_celeba.txt plus CelebA-HQ-to-CelebA-mapping.txt (or image_list.txt)")
    log.info("to map HQ image indices to original CelebA filenames.")
    log.info("Original CelebA images in img_align_celeba/ use original labels directly.")
    log.info("Attribute annotations are required by the CelebA-HQ loader.")


def download_ffhq(output_dir: Path) -> None:
    """Show manual FFHQ download instructions; no data is downloaded."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("FFHQ requires manual setup; no data has been downloaded.")
    log.info("Official source: https://github.com/NVlabs/ffhq-dataset")
    log.info("Download the official script and its dependencies, then run:")
    log.info("  cd %s", shlex.quote(str(output_dir.resolve())))
    log.info("  python /absolute/path/to/download_ffhq.py --json --images")
    log.info("The official script downloads into the current directory (no -o flag).")
    log.info("Supported: flat or recursively nested PNG/JPG/JPEG images, including")
    log.info("images1024x1024/00000/00000.png. Extract ZIPs first; TFRecords are unsupported.")
    log.info("Keep only the desired aligned image collection under the dataset root.")


def verify_dataset(dataset_dir: Path, dataset_name: str) -> bool:
    """Check loader metadata and decode all images included across its splits.

    This checks local usability, not official dataset completeness or checksums.
    CelebA-HQ requires valid annotations and, for original labels, an HQ mapping.
    """
    if not dataset_dir.is_dir():
        log.error("Dataset directory not found: %s", dataset_dir)
        return False
    if dataset_name not in {"celeba_hq", "ffhq"}:
        log.error("Unsupported dataset: %s", dataset_name)
        return False

    from PIL import Image

    from src.data.datasets.celeba_hq import CelebAHQDataset
    from src.data.datasets.ffhq import FFHQDataset

    try:
        image_paths: set[Path] = set()
        for split in ("train", "val", "test"):
            if dataset_name == "celeba_hq":
                dataset = CelebAHQDataset(root=dataset_dir, split=split)
                image_paths.update(dataset.image_dir / name for name in dataset.filenames)
            else:
                dataset = FFHQDataset(root=dataset_dir, split=split)
                image_paths.update(dataset.image_paths)
        if not image_paths:
            raise ValueError("No usable images found")
        for path in sorted(image_paths):
            try:
                with Image.open(path) as image:
                    image.load()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot decode image {path}: {exc}") from exc
    except (OSError, ValueError, KeyError) as exc:
        log.error("%s verification failed: %s", dataset_name, exc)
        return False

    log.info(
        "Verified %d loader-visible images and dataset metadata in %s",
        len(image_paths),
        dataset_dir,
    )
    log.info("Local usability only: no official completeness or checksum guarantee.")
    return True


def main(argv: list[str] | None = None) -> int:
    """Show instructions or return a nonzero status for any failed verification."""
    parser = argparse.ArgumentParser(
        description="Show manual dataset setup instructions (no download), or verify local usability."
    )
    parser.add_argument("--dataset", choices=["celeba_hq", "ffhq", "all"], default="all")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("STYLEFORMER_ROOT", ".")) / "data",
        help="Data root; relative paths use the invocation directory (default: STYLEFORMER_ROOT/data or ./data)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Validate loader metadata and decode all loader-visible images; failures exit nonzero",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)]
    )
    data_dir = args.data_dir.expanduser().resolve()
    datasets = ["celeba_hq", "ffhq"] if args.dataset == "all" else [args.dataset]
    success = True
    for dataset in datasets:
        dataset_dir = data_dir / dataset
        if args.verify:
            valid = verify_dataset(dataset_dir, dataset)
            log.info("%s: %s", dataset, "OK" if valid else "FAILED")
            success = valid and success
        elif dataset == "celeba_hq":
            download_celeba_hq(dataset_dir)
        else:
            download_ffhq(dataset_dir)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

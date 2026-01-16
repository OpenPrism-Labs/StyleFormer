"""Dataset preparation and download utilities."""

import argparse
import logging
import os
from pathlib import Path

from rich.logging import RichHandler


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)


def download_celeba_hq(output_dir: Path) -> None:
    """Download CelebA-HQ dataset.
    
    Note: CelebA-HQ requires manual download due to licensing.
    This function provides instructions.
    """
    log.info("=" * 60)
    log.info("CelebA-HQ Dataset Setup")
    log.info("=" * 60)
    log.info("")
    log.info("CelebA-HQ requires manual download. Follow these steps:")
    log.info("")
    log.info("Option 1: From Google Drive (official)")
    log.info("  1. Go to: https://github.com/tkarras/progressive_growing_of_gans")
    log.info("  2. Download 'CelebA-HQ' from the Google Drive link")
    log.info(f"  3. Extract to: {output_dir}")
    log.info("")
    log.info("Option 2: Using kaggle CLI")
    log.info("  kaggle datasets download -d lamsimon/celebahq")
    log.info(f"  unzip celebahq.zip -d {output_dir}")
    log.info("")
    log.info("Required files:")
    log.info(f"  {output_dir}/")
    log.info("    images/")
    log.info("      000001.jpg")
    log.info("      000002.jpg")
    log.info("      ...")
    log.info("    list_attr_celeba.txt")
    log.info("")
    
    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)


def download_ffhq(output_dir: Path) -> None:
    """Download FFHQ dataset.
    
    Note: FFHQ requires manual download due to size.
    This function provides instructions.
    """
    log.info("=" * 60)
    log.info("FFHQ Dataset Setup")
    log.info("=" * 60)
    log.info("")
    log.info("FFHQ (70,000 images at 1024x1024) requires manual download.")
    log.info("")
    log.info("Option 1: Official download script")
    log.info("  git clone https://github.com/NVlabs/ffhq-dataset")
    log.info("  cd ffhq-dataset")
    log.info(f"  python download_ffhq.py --images -o {output_dir}")
    log.info("")
    log.info("Option 2: Kaggle (256x256 version)")
    log.info("  kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq")
    log.info(f"  unzip flickrfaceshq-dataset-ffhq.zip -d {output_dir}")
    log.info("")
    log.info("Required structure:")
    log.info(f"  {output_dir}/")
    log.info("    00000/")
    log.info("      00000.png")
    log.info("      00001.png")
    log.info("      ...")
    log.info("    00001/")
    log.info("      01000.png")
    log.info("      ...")
    log.info("")
    
    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)


def verify_dataset(dataset_dir: Path, dataset_name: str) -> bool:
    """Verify dataset is properly set up.
    
    Args:
        dataset_dir: Directory containing dataset.
        dataset_name: Name of dataset ("celeba_hq" or "ffhq").
        
    Returns:
        True if dataset is valid.
    """
    if not dataset_dir.exists():
        log.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    if dataset_name == "celeba_hq":
        # Check for images
        img_dirs = ["images", "img_align_celeba", "CelebA-HQ"]
        has_images = any((dataset_dir / d).exists() for d in img_dirs)
        
        if not has_images:
            log.error("No image directory found (expected 'images/' or 'img_align_celeba/')")
            return False
        
        # Check for attributes
        attr_file = dataset_dir / "list_attr_celeba.txt"
        if not attr_file.exists():
            log.warning("Attribute file not found: list_attr_celeba.txt")
            log.warning("Dataset will work but without attribute labels.")
        
        # Count images
        for d in img_dirs:
            img_dir = dataset_dir / d
            if img_dir.exists():
                n_images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
                log.info(f"Found {n_images} images in {img_dir}")
                break
        
        return True
    
    elif dataset_name == "ffhq":
        # Check for images (nested or flat)
        subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        
        if subdirs and subdirs[0].name.isdigit():
            # Nested structure
            n_images = sum(
                len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg")))
                for d in subdirs
            )
        else:
            # Flat structure
            n_images = len(list(dataset_dir.glob("*.png"))) + len(list(dataset_dir.glob("*.jpg")))
        
        log.info(f"Found {n_images} images")
        return n_images > 0
    
    return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare datasets for StyleFormer")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["celeba_hq", "ffhq", "all"],
        default="all",
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root data directory",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing datasets",
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    datasets = ["celeba_hq", "ffhq"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets:
        dataset_dir = data_dir / dataset
        
        if args.verify:
            log.info(f"Verifying {dataset}...")
            if verify_dataset(dataset_dir, dataset):
                log.info(f"{dataset}: OK")
            else:
                log.error(f"{dataset}: FAILED")
        else:
            if dataset == "celeba_hq":
                download_celeba_hq(dataset_dir)
            elif dataset == "ffhq":
                download_ffhq(dataset_dir)


if __name__ == "__main__":
    main()

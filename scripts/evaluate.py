#!/usr/bin/env python3
"""Evaluation script for face transformation models.

This script computes evaluation metrics on generated images:
- FID (Fréchet Inception Distance)
- Identity Similarity (using face recognition)
- Attribute Accuracy (using attribute classifiers)

Usage:
    # Basic evaluation with default config
    python scripts/evaluate.py \\
        --generated-dir outputs/generated \\
        --source-dir data/celeba_hq/test \\
        --real-dir data/celeba_hq/train

    # Specify target attributes for accuracy evaluation
    python scripts/evaluate.py \\
        --generated-dir outputs/generated \\
        --source-dir data/celeba_hq/test \\
        --target-age old \\
        --target-gender female

    # Skip certain metrics
    python scripts/evaluate.py \\
        --generated-dir outputs/generated \\
        --source-dir data/celeba_hq/test \\
        --no-fid \\
        --no-attribute

    # Save results to file
    python scripts/evaluate.py \\
        --generated-dir outputs/generated \\
        --source-dir data/celeba_hq/test \\
        --output-file results/evaluation.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.evaluation import Evaluator
from src.evaluation.evaluator import EvaluationConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate face transformation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="Directory containing generated images",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing source images (for identity similarity)",
    )

    # Optional directories
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=None,
        help="Directory containing real target domain images (for FID)",
    )

    # Target attributes
    parser.add_argument(
        "--target-age",
        type=str,
        choices=["young", "old"],
        default=None,
        help="Target age attribute",
    )
    parser.add_argument(
        "--target-gender",
        type=str,
        choices=["male", "female"],
        default=None,
        help="Target gender attribute",
    )
    parser.add_argument(
        "--target-smile",
        type=str,
        choices=["no_smile", "smile"],
        default=None,
        help="Target smile attribute",
    )
    parser.add_argument(
        "--target-glasses",
        type=str,
        choices=["no_glasses", "glasses"],
        default=None,
        help="Target glasses attribute",
    )

    # Classifier paths
    parser.add_argument(
        "--age-classifier",
        type=Path,
        default=None,
        help="Path to age classifier weights",
    )
    parser.add_argument(
        "--gender-classifier",
        type=Path,
        default=None,
        help="Path to gender classifier weights",
    )
    parser.add_argument(
        "--smile-classifier",
        type=Path,
        default=None,
        help="Path to smile classifier weights",
    )
    parser.add_argument(
        "--glasses-classifier",
        type=Path,
        default=None,
        help="Path to glasses classifier weights",
    )

    # Metric toggles
    parser.add_argument(
        "--no-fid",
        action="store_true",
        help="Skip FID computation",
    )
    parser.add_argument(
        "--no-identity",
        action="store_true",
        help="Skip identity similarity computation",
    )
    parser.add_argument(
        "--no-attribute",
        action="store_true",
        help="Skip attribute accuracy computation",
    )

    # Sampling
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--fid-samples",
        type=int,
        default=50000,
        help="Maximum samples for FID computation",
    )

    # Data loading
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for evaluation",
    )

    # Output
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to save results JSON",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )

    return parser.parse_args()


def create_dataloader(
    image_dir: Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> DataLoader:
    """Create a DataLoader for image directory.

    Args:
        image_dir: Directory containing images.
        batch_size: Batch size.
        image_size: Target image size.
        num_workers: Number of workers.

    Returns:
        DataLoader for the images.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Check if directory has subdirectories (ImageFolder format)
    subdirs = [d for d in image_dir.iterdir() if d.is_dir()]

    if subdirs:
        dataset = ImageFolder(image_dir, transform=transform)
    else:
        # Flat directory - create a simple dataset
        from torch.utils.data import Dataset
        from PIL import Image

        class FlatImageDataset(Dataset):
            def __init__(self, root: Path, transform):
                self.images = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = Image.open(self.images[idx]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img

        dataset = FlatImageDataset(image_dir, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main() -> None:
    """Run evaluation."""
    args = parse_args()

    print("=" * 50)
    print("Face Transformation Evaluation")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Generated images: {args.generated_dir}")
    print(f"Source images: {args.source_dir}")
    if args.real_dir:
        print(f"Real images: {args.real_dir}")
    print()

    # Validate directories
    if not args.generated_dir.exists():
        print(f"Error: Generated directory does not exist: {args.generated_dir}")
        sys.exit(1)
    if not args.source_dir.exists():
        print(f"Error: Source directory does not exist: {args.source_dir}")
        sys.exit(1)
    if args.real_dir and not args.real_dir.exists():
        print(f"Error: Real directory does not exist: {args.real_dir}")
        sys.exit(1)

    # Create config
    config = EvaluationConfig(
        compute_fid=not args.no_fid and args.real_dir is not None,
        compute_identity=not args.no_identity,
        compute_attribute=not args.no_attribute,
        fid_max_samples=args.fid_samples,
        identity_max_samples=args.max_samples,
        attribute_max_samples=args.max_samples,
        device=args.device,
    )

    # Create evaluator
    print("Initializing evaluator...")
    evaluator = Evaluator(config=config, device=args.device)

    # Load attribute classifiers if provided
    classifiers = {}
    if args.age_classifier:
        classifiers["age"] = args.age_classifier
    if args.gender_classifier:
        classifiers["gender"] = args.gender_classifier
    if args.smile_classifier:
        classifiers["smile"] = args.smile_classifier
    if args.glasses_classifier:
        classifiers["glasses"] = args.glasses_classifier

    if classifiers:
        print(f"Loading {len(classifiers)} attribute classifiers...")
        try:
            evaluator.load_attribute_classifiers(classifiers)
        except NotImplementedError:
            print("Warning: Attribute classifier loading not implemented yet")

    # Build target attributes
    target_attributes = {}
    if args.target_age:
        target_attributes["age"] = args.target_age
    if args.target_gender:
        target_attributes["gender"] = args.target_gender
    if args.target_smile:
        target_attributes["smile"] = args.target_smile
    if args.target_glasses:
        target_attributes["glasses"] = args.target_glasses

    if target_attributes:
        print(f"Target attributes: {target_attributes}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    generated_loader = create_dataloader(
        args.generated_dir,
        args.batch_size,
        args.image_size,
        args.num_workers,
    )
    source_loader = create_dataloader(
        args.source_dir,
        args.batch_size,
        args.image_size,
        args.num_workers,
    )

    real_loader = None
    if args.real_dir:
        real_loader = create_dataloader(
            args.real_dir,
            args.batch_size,
            args.image_size,
            args.num_workers,
        )

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate_full(
        source_loader=source_loader,
        generated_loader=generated_loader,
        real_loader=real_loader,
        target_attributes=target_attributes if target_attributes else None,
    )

    # Print results
    print()
    print(evaluator.format_results(results))

    # Save results if requested
    if args.output_file:
        evaluator.save_results(results, args.output_file)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

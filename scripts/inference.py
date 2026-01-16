#!/usr/bin/env python3
"""Inference script for face transformation.

Run inference on individual images or directories with a trained model.

Usage:
    # Single image
    python scripts/inference.py \\
        --checkpoint checkpoints/model.ckpt \\
        --input input.jpg \\
        --output output.jpg \\
        --target-age old \\
        --target-gender female

    # Directory processing
    python scripts/inference.py \\
        --checkpoint checkpoints/model.ckpt \\
        --input-dir inputs/ \\
        --output-dir outputs/ \\
        --target-age old

    # Create comparison grid
    python scripts/inference.py \\
        --checkpoint checkpoints/model.ckpt \\
        --input-dir inputs/ \\
        --output-grid grid.png \\
        --target-age old

    # Interpolation
    python scripts/inference.py \\
        --checkpoint checkpoints/model.ckpt \\
        --input input.jpg \\
        --output-dir interpolation/ \\
        --target-age old \\
        --interpolate \\
        --n-steps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image

from src.inference import InferencePipeline
from src.inference.pipeline import InferenceConfig, MultiAttributePipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run face transformation inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default=None,
        help="Model class name (if not specified, will try to infer from checkpoint)",
    )

    # Input/Output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        help="Path to single input image",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing input images",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save output image (for single image)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output images",
    )
    parser.add_argument(
        "--output-grid",
        type=Path,
        default=None,
        help="Path to save comparison grid",
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

    # Attribute strengths (for multi-attribute)
    parser.add_argument(
        "--age-strength",
        type=float,
        default=1.0,
        help="Strength for age transformation (0-1)",
    )
    parser.add_argument(
        "--gender-strength",
        type=float,
        default=1.0,
        help="Strength for gender transformation (0-1)",
    )
    parser.add_argument(
        "--smile-strength",
        type=float,
        default=1.0,
        help="Strength for smile transformation (0-1)",
    )
    parser.add_argument(
        "--glasses-strength",
        type=float,
        default=1.0,
        help="Strength for glasses transformation (0-1)",
    )

    # Interpolation
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Create interpolation instead of direct transformation",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10,
        help="Number of interpolation steps",
    )

    # Processing options
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for processing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for batch processing",
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save side-by-side comparison images",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Number of columns in output grid",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision (FP16)",
    )

    return parser.parse_args()


def get_model_class(class_name: str | None):
    """Get model class by name.

    Args:
        class_name: Name of the model class.

    Returns:
        Model class.

    Note:
        Students should add their model classes here.
    """
    if class_name is None:
        # Try to import a default model
        print("Warning: No model class specified.")
        print("Students: Implement your model and specify --model-class")
        return None

    # Example model class imports (students should add their own)
    model_registry = {
        # "StyleGANTransformer": lambda: StyleGANTransformer,
        # "E4EEncoder": lambda: E4EEncoder,
    }

    if class_name in model_registry:
        return model_registry[class_name]()

    raise ValueError(
        f"Unknown model class: {class_name}. "
        f"Available: {list(model_registry.keys())}"
    )


def build_target_attributes(args: argparse.Namespace) -> dict[str, str]:
    """Build target attributes dict from args."""
    attrs = {}
    if args.target_age:
        attrs["age"] = args.target_age
    if args.target_gender:
        attrs["gender"] = args.target_gender
    if args.target_smile:
        attrs["smile"] = args.target_smile
    if args.target_glasses:
        attrs["glasses"] = args.target_glasses
    return attrs


def build_multi_attributes(args: argparse.Namespace) -> dict[str, tuple[str, float]]:
    """Build multi-attribute dict with strengths."""
    attrs = {}
    if args.target_age:
        attrs["age"] = (args.target_age, args.age_strength)
    if args.target_gender:
        attrs["gender"] = (args.target_gender, args.gender_strength)
    if args.target_smile:
        attrs["smile"] = (args.target_smile, args.smile_strength)
    if args.target_glasses:
        attrs["glasses"] = (args.target_glasses, args.glasses_strength)
    return attrs


def main() -> None:
    """Run inference."""
    args = parse_args()

    print("=" * 50)
    print("Face Transformation Inference")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Validate checkpoint exists
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Get target attributes
    target_attributes = build_target_attributes(args)
    if not target_attributes:
        print("Error: No target attributes specified.")
        print("Use --target-age, --target-gender, etc.")
        sys.exit(1)

    print(f"Target attributes: {target_attributes}")

    # Get model class
    model_class = get_model_class(args.model_class)
    if model_class is None:
        print("\nError: Model class not available.")
        print("This is infrastructure only - students need to implement models.")
        print("\nExample usage after implementing a model:")
        print("  python scripts/inference.py \\")
        print("      --checkpoint checkpoints/model.ckpt \\")
        print("      --model-class StyleGANTransformer \\")
        print("      --input input.jpg \\")
        print("      --output output.jpg \\")
        print("      --target-age old")
        sys.exit(1)

    # Create config
    config = InferenceConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        mixed_precision=not args.no_fp16,
    )

    # Check if we need multi-attribute pipeline
    multi_attrs = build_multi_attributes(args)
    has_custom_strengths = any(
        s != 1.0 for _, s in multi_attrs.values()
    )

    # Create pipeline
    print("\nLoading model...")
    if has_custom_strengths:
        pipeline = MultiAttributePipeline.from_checkpoint(
            args.checkpoint,
            model_class=model_class,
            config=config,
        )
    else:
        pipeline = InferencePipeline.from_checkpoint(
            args.checkpoint,
            model_class=model_class,
            config=config,
        )

    # Process based on input type
    if args.input:
        # Single image
        print(f"\nProcessing: {args.input}")

        if args.interpolate:
            # Interpolation mode
            output_dir = args.output_dir or Path("interpolation")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Source attributes (neutral)
            source_attrs = {attr: "neutral" for attr in target_attributes}

            images = pipeline.create_interpolation(
                args.input,
                source_attributes=source_attrs,
                target_attributes=target_attributes,
                n_steps=args.n_steps,
            )

            for i, img in enumerate(images):
                output_path = output_dir / f"step_{i:03d}.png"
                pil_img = pipeline._to_pil(img)
                pil_img.save(output_path)

            print(f"Saved {len(images)} interpolation steps to {output_dir}")

        else:
            # Direct transformation
            if has_custom_strengths:
                result = pipeline.transform_multi(args.input, multi_attrs)
            else:
                result = pipeline.transform(args.input, target_attributes)

            # Save result
            output_path = args.output or Path("output.png")
            pipeline.save_result(
                result,
                output_path,
                save_comparison=args.save_comparison,
            )
            print(f"Saved result to {output_path}")

    else:
        # Directory processing
        print(f"\nProcessing directory: {args.input_dir}")

        if not args.input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)

        output_dir = args.output_dir or Path("outputs")

        n_processed = pipeline.transform_directory(
            args.input_dir,
            output_dir,
            target_attributes,
        )
        print(f"\nProcessed {n_processed} images")
        print(f"Results saved to {output_dir}")

        # Create grid if requested
        if args.output_grid:
            # Get all results for grid
            image_paths = list(args.input_dir.glob("*.jpg")) + \
                         list(args.input_dir.glob("*.png"))

            results = pipeline.transform_batch(
                image_paths[:16],  # Limit for grid
                target_attributes,
            )

            grid = pipeline.create_grid(
                results,
                ncols=args.grid_cols,
                include_source=True,
            )
            grid.save(args.output_grid)
            print(f"Grid saved to {args.output_grid}")


if __name__ == "__main__":
    main()

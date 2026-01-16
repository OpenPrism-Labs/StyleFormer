"""Inference pipeline for face transformation models.

Provides a unified interface for running inference on face images:
- Single image transformation
- Batch processing
- Multi-attribute editing
- Visualization generation

Example:
    >>> from src.inference import InferencePipeline
    >>> pipeline = InferencePipeline.from_checkpoint("checkpoints/model.ckpt")
    >>> result = pipeline.transform(image, target_attributes={"age": "old"})
    >>> pipeline.save_result(result, "output.png")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class TransformationModel(Protocol):
    """Protocol for face transformation models.

    Any model implementing this protocol can be used with InferencePipeline.
    Students should implement their models to conform to this interface.
    """

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            images: Input images (B, C, H, W).

        Returns:
            Latent codes.
        """
        ...

    def edit_latent(
        self,
        latent: torch.Tensor,
        target_attributes: dict[str, Any],
    ) -> torch.Tensor:
        """Edit latent codes to achieve target attributes.

        Args:
            latent: Original latent codes.
            target_attributes: Target attribute values.

        Returns:
            Edited latent codes.
        """
        ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to images.

        Args:
            latent: Latent codes.

        Returns:
            Generated images (B, C, H, W).
        """
        ...


@dataclass
class TransformationResult:
    """Container for transformation results.

    Attributes:
        source: Source image tensor.
        generated: Generated/transformed image tensor.
        latent_source: Source latent code (if available).
        latent_edited: Edited latent code (if available).
        target_attributes: Target attributes used.
        metadata: Additional metadata.
    """

    source: torch.Tensor
    generated: torch.Tensor
    latent_source: torch.Tensor | None = None
    latent_edited: torch.Tensor | None = None
    target_attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline.

    Attributes:
        image_size: Target image size for processing.
        batch_size: Batch size for batch processing.
        device: Device to run inference on.
        mixed_precision: Whether to use mixed precision (FP16).
        save_latents: Whether to save latent codes.
    """

    image_size: int = 256
    batch_size: int = 8
    device: str = "cuda"
    mixed_precision: bool = True
    save_latents: bool = False


class InferencePipeline:
    """Unified inference pipeline for face transformation.

    This class provides a consistent interface for running inference
    with any face transformation model that implements the
    TransformationModel protocol.

    Attributes:
        model: The transformation model.
        config: Inference configuration.
        device: Device to run on.
        transform: Image preprocessing transform.
        inverse_transform: Image postprocessing transform.

    Example:
        >>> # From checkpoint
        >>> pipeline = InferencePipeline.from_checkpoint(
        ...     "checkpoints/model.ckpt",
        ...     model_class=MyTransformationModel,
        ... )
        >>>
        >>> # Single image
        >>> result = pipeline.transform(
        ...     "input.jpg",
        ...     target_attributes={"age": "old", "gender": "female"},
        ... )
        >>> pipeline.save_result(result, "output.jpg")
        >>>
        >>> # Batch processing
        >>> results = pipeline.transform_batch(
        ...     input_dir="inputs/",
        ...     output_dir="outputs/",
        ...     target_attributes={"age": "old"},
        ... )

    Note:
        Students need to implement their model and pass it to the pipeline.
        The model should implement encode, edit_latent, and decode methods.
    """

    def __init__(
        self,
        model: nn.Module | TransformationModel,
        config: InferenceConfig | None = None,
    ) -> None:
        """Initialize inference pipeline.

        Args:
            model: Transformation model.
            config: Inference configuration.
        """
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.model.eval()

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.inverse_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-1, -1, -1],
                std=[2, 2, 2],
            ),  # Undo [-1, 1] normalization
            transforms.Lambda(lambda x: x.clamp(0, 1)),
            transforms.ToPILImage(),
        ])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        model_class: type[nn.Module],
        model_kwargs: dict[str, Any] | None = None,
        config: InferenceConfig | None = None,
    ) -> "InferencePipeline":
        """Create pipeline from a checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.
            model_class: Model class to instantiate.
            model_kwargs: Keyword arguments for model constructor.
            config: Inference configuration.

        Returns:
            Initialized inference pipeline.

        Example:
            >>> pipeline = InferencePipeline.from_checkpoint(
            ...     "checkpoints/stylegan_transformer.ckpt",
            ...     model_class=StyleGANTransformer,
            ...     model_kwargs={"latent_dim": 512},
            ... )
        """
        checkpoint_path = Path(checkpoint_path)
        model_kwargs = model_kwargs or {}

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle Lightning checkpoint format
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove 'model.' prefix if present (Lightning convention)
            state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict.items()
            }
        else:
            state_dict = checkpoint

        # Instantiate model
        model = model_class(**model_kwargs)
        model.load_state_dict(state_dict, strict=False)

        return cls(model=model, config=config)

    def _load_image(self, image: str | Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Load and preprocess an image.

        Args:
            image: Image path, PIL Image, or tensor.

        Returns:
            Preprocessed tensor (1, C, H, W).
        """
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def _to_pil(self, tensor: torch.Tensor) -> Image.Image | list[Image.Image]:
        """Convert tensor to PIL Image(s).

        Args:
            tensor: Image tensor (B, C, H, W) or (C, H, W).

        Returns:
            PIL Image or list of PIL Images.
        """
        if tensor.dim() == 3:
            return self.inverse_transform(tensor.cpu())

        images = [self.inverse_transform(t.cpu()) for t in tensor]
        return images if len(images) > 1 else images[0]

    @torch.no_grad()
    def transform(
        self,
        image: str | Path | Image.Image | torch.Tensor,
        target_attributes: dict[str, Any],
        return_latents: bool = False,
    ) -> TransformationResult:
        """Transform a single image.

        Args:
            image: Input image (path, PIL Image, or tensor).
            target_attributes: Target attribute values.
            return_latents: Whether to return latent codes.

        Returns:
            TransformationResult with source and generated images.

        Example:
            >>> result = pipeline.transform(
            ...     "input.jpg",
            ...     target_attributes={"age": "old", "gender": "female"},
            ... )
            >>> result.generated  # Transformed image tensor
        """
        # Load image
        source_tensor = self._load_image(image)

        # Use mixed precision if enabled
        autocast_ctx = (
            torch.cuda.amp.autocast()
            if self.config.mixed_precision and self.device.type == "cuda"
            else torch.inference_mode()
        )

        with autocast_ctx:
            # Encode
            latent_source = self.model.encode(source_tensor)

            # Edit
            latent_edited = self.model.edit_latent(latent_source, target_attributes)

            # Decode
            generated = self.model.decode(latent_edited)

        return TransformationResult(
            source=source_tensor.cpu(),
            generated=generated.cpu(),
            latent_source=latent_source.cpu() if return_latents else None,
            latent_edited=latent_edited.cpu() if return_latents else None,
            target_attributes=target_attributes,
        )

    @torch.no_grad()
    def transform_batch(
        self,
        images: list[str | Path | Image.Image | torch.Tensor],
        target_attributes: dict[str, Any],
        show_progress: bool = True,
    ) -> list[TransformationResult]:
        """Transform a batch of images.

        Args:
            images: List of input images.
            target_attributes: Target attribute values (same for all).
            show_progress: Whether to show progress bar.

        Returns:
            List of TransformationResult objects.
        """
        results = []

        iterator = tqdm(images, desc="Transforming") if show_progress else images

        for image in iterator:
            result = self.transform(image, target_attributes)
            results.append(result)

        return results

    def transform_directory(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        target_attributes: dict[str, Any],
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        show_progress: bool = True,
    ) -> int:
        """Transform all images in a directory.

        Args:
            input_dir: Input directory containing images.
            output_dir: Output directory for results.
            target_attributes: Target attribute values.
            image_extensions: Valid image file extensions.
            show_progress: Whether to show progress bar.

        Returns:
            Number of images processed.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_paths = [
            p for p in input_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]

        iterator = (
            tqdm(image_paths, desc="Processing")
            if show_progress else image_paths
        )

        for image_path in iterator:
            result = self.transform(image_path, target_attributes)
            output_path = output_dir / image_path.name
            self.save_result(result, output_path)

        return len(image_paths)

    def save_result(
        self,
        result: TransformationResult,
        path: Path | str,
        save_comparison: bool = False,
    ) -> None:
        """Save transformation result to file.

        Args:
            result: Transformation result to save.
            path: Output path.
            save_comparison: If True, save side-by-side comparison.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if save_comparison:
            # Create side-by-side comparison
            source_pil = self._to_pil(result.source)
            generated_pil = self._to_pil(result.generated)

            # Combine horizontally
            width = source_pil.width + generated_pil.width
            height = max(source_pil.height, generated_pil.height)
            comparison = Image.new("RGB", (width, height))
            comparison.paste(source_pil, (0, 0))
            comparison.paste(generated_pil, (source_pil.width, 0))
            comparison.save(path)
        else:
            generated_pil = self._to_pil(result.generated)
            generated_pil.save(path)

    def create_grid(
        self,
        results: list[TransformationResult],
        ncols: int = 4,
        include_source: bool = True,
        padding: int = 4,
    ) -> Image.Image:
        """Create a grid visualization of results.

        Args:
            results: List of transformation results.
            ncols: Number of columns in grid.
            include_source: Whether to include source images.
            padding: Padding between images.

        Returns:
            PIL Image with grid visualization.
        """
        if not results:
            raise ValueError("No results to visualize")

        # Get image size from first result
        sample = self._to_pil(results[0].generated)
        img_size = sample.size[0]

        # Calculate grid dimensions
        n_images_per_result = 2 if include_source else 1
        total_images = len(results) * n_images_per_result
        nrows = (total_images + ncols - 1) // ncols

        # Create grid
        grid_width = ncols * (img_size + padding) + padding
        grid_height = nrows * (img_size + padding) + padding
        grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

        # Place images
        idx = 0
        for result in results:
            if include_source:
                source_pil = self._to_pil(result.source)
                row, col = divmod(idx, ncols)
                x = col * (img_size + padding) + padding
                y = row * (img_size + padding) + padding
                grid.paste(source_pil, (x, y))
                idx += 1

            generated_pil = self._to_pil(result.generated)
            row, col = divmod(idx, ncols)
            x = col * (img_size + padding) + padding
            y = row * (img_size + padding) + padding
            grid.paste(generated_pil, (x, y))
            idx += 1

        return grid

    def create_interpolation(
        self,
        image: str | Path | Image.Image | torch.Tensor,
        source_attributes: dict[str, Any],
        target_attributes: dict[str, Any],
        n_steps: int = 10,
    ) -> list[torch.Tensor]:
        """Create interpolation between source and target attributes.

        Args:
            image: Input image.
            source_attributes: Starting attribute values.
            target_attributes: Ending attribute values.
            n_steps: Number of interpolation steps.

        Returns:
            List of interpolated images.

        Note:
            Requires model to support interpolation in latent space.
            Students should implement edit_latent to handle interpolation.
        """
        source_tensor = self._load_image(image)

        with torch.no_grad():
            latent_source = self.model.encode(source_tensor)

            # Get start and end latents
            latent_start = self.model.edit_latent(latent_source, source_attributes)
            latent_end = self.model.edit_latent(latent_source, target_attributes)

            # Interpolate
            interpolated_images = []
            for i in range(n_steps):
                alpha = i / (n_steps - 1)
                latent_interp = latent_start + alpha * (latent_end - latent_start)
                image = self.model.decode(latent_interp)
                interpolated_images.append(image.cpu())

        return interpolated_images


class MultiAttributePipeline(InferencePipeline):
    """Extended pipeline for multi-attribute transformation.

    Supports editing multiple attributes simultaneously with
    attribute-specific editing strengths.

    Example:
        >>> pipeline = MultiAttributePipeline.from_checkpoint(...)
        >>> result = pipeline.transform_multi(
        ...     image,
        ...     attributes={
        ...         "age": ("old", 1.0),      # (target, strength)
        ...         "gender": ("female", 0.8),
        ...     },
        ... )
    """

    @torch.no_grad()
    def transform_multi(
        self,
        image: str | Path | Image.Image | torch.Tensor,
        attributes: dict[str, tuple[Any, float]],
        use_null_space: bool = True,
    ) -> TransformationResult:
        """Transform with multiple attributes and per-attribute strengths.

        Args:
            image: Input image.
            attributes: Dict of attribute name to (target_value, strength) tuples.
            use_null_space: Whether to use null-space projection for
                attribute disentanglement.

        Returns:
            TransformationResult.

        Note:
            Students should implement null-space projection in their model
            to ensure attributes don't interfere with each other.
        """
        source_tensor = self._load_image(image)

        # Extract targets and strengths
        target_attributes = {
            attr: val for attr, (val, _) in attributes.items()
        }
        strengths = {
            attr: strength for attr, (_, strength) in attributes.items()
        }

        with torch.no_grad():
            latent_source = self.model.encode(source_tensor)

            # Edit with strengths (model should handle this)
            if hasattr(self.model, "edit_latent_with_strengths"):
                latent_edited = self.model.edit_latent_with_strengths(
                    latent_source,
                    target_attributes,
                    strengths,
                    use_null_space=use_null_space,
                )
            else:
                # Fallback to standard edit
                latent_edited = self.model.edit_latent(latent_source, target_attributes)

            generated = self.model.decode(latent_edited)

        return TransformationResult(
            source=source_tensor.cpu(),
            generated=generated.cpu(),
            target_attributes=target_attributes,
            metadata={"strengths": strengths, "use_null_space": use_null_space},
        )

"""StyleGAN2 pretrained model loading utilities.

This module provides utilities for loading pretrained StyleGAN2 generators
for use in face transformation.

Supported models:
- NVIDIA's official StyleGAN2-ADA-PyTorch checkpoints
- rosinality's StyleGAN2-PyTorch checkpoints

Example:
    >>> from src.models.pretrained import load_stylegan2
    >>> generator = load_stylegan2("stylegan2-ffhq-1024")
    >>> # Or from a local file
    >>> generator = load_stylegan2("/path/to/checkpoint.pt")

References:
    - StyleGAN2-ADA: https://github.com/NVlabs/stylegan2-ada-pytorch
    - rosinality: https://github.com/rosinality/stylegan2-pytorch
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.pretrained.download import download_model, get_model_path


class StyleGAN2GeneratorWrapper(nn.Module):
    """Wrapper for StyleGAN2 generator with consistent interface.

    This wrapper provides a consistent interface for StyleGAN2 generators
    regardless of the source (NVIDIA, rosinality, etc.).

    Attributes:
        generator: The underlying generator model.
        w_dim: Dimension of W latent space.
        img_resolution: Output image resolution.
        num_ws: Number of W vectors for style mixing.

    Example:
        >>> wrapper = StyleGAN2GeneratorWrapper(generator, w_dim=512)
        >>> # Generate from W latent
        >>> w = torch.randn(1, 18, 512)  # W+ space
        >>> image = wrapper.synthesis(w)
        >>> # Generate from Z latent
        >>> z = torch.randn(1, 512)
        >>> image = wrapper(z)

    Note:
        Students should implement the actual generator architecture
        or use one of the existing implementations.
    """

    def __init__(
        self,
        generator: nn.Module,
        w_dim: int = 512,
        img_resolution: int = 1024,
        num_ws: int = 18,
    ) -> None:
        """Initialize wrapper.

        Args:
            generator: The underlying generator module.
            w_dim: Dimension of W latent space.
            img_resolution: Output image resolution.
            num_ws: Number of W vectors (layers) for W+ space.
        """
        super().__init__()
        self.generator = generator
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.num_ws = num_ws

    def mapping(
        self,
        z: torch.Tensor,
        c: torch.Tensor | None = None,
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        """Map from Z space to W space.

        Args:
            z: Latent codes in Z space (B, z_dim).
            c: Class labels (optional, for conditional models).
            truncation_psi: Truncation factor (1.0 = no truncation).

        Returns:
            Latent codes in W space (B, num_ws, w_dim).
        """
        if hasattr(self.generator, "mapping"):
            return self.generator.mapping(z, c, truncation_psi=truncation_psi)
        elif hasattr(self.generator, "style"):
            # rosinality style
            w = self.generator.style(z)
            return w.unsqueeze(1).repeat(1, self.num_ws, 1)
        else:
            raise NotImplementedError(
                "Generator does not have mapping network. Students: Implement mapping function."
            )

    def synthesis(
        self,
        w: torch.Tensor,
        noise_mode: str = "const",
    ) -> torch.Tensor:
        """Synthesize images from W latents.

        Args:
            w: Latent codes in W/W+ space (B, num_ws, w_dim) or (B, w_dim).
            noise_mode: Noise mode ('const', 'random', 'none').

        Returns:
            Generated images (B, 3, H, W).
        """
        if hasattr(self.generator, "synthesis"):
            return self.generator.synthesis(w, noise_mode=noise_mode)
        elif hasattr(self.generator, "forward"):
            # rosinality style - expects list of styles
            if w.dim() == 2:
                w = w.unsqueeze(1).repeat(1, self.num_ws, 1)
            styles = [w[:, i] for i in range(w.shape[1])]
            return self.generator(styles, input_is_latent=True)[0]
        else:
            raise NotImplementedError(
                "Generator synthesis not implemented. Students: Implement synthesis function."
            )

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor | None = None,
        truncation_psi: float = 1.0,
        noise_mode: str = "const",
    ) -> torch.Tensor:
        """Generate images from Z latents.

        Args:
            z: Latent codes in Z space (B, z_dim).
            c: Class labels (optional).
            truncation_psi: Truncation factor.
            noise_mode: Noise mode.

        Returns:
            Generated images (B, 3, H, W).
        """
        w = self.mapping(z, c, truncation_psi)
        return self.synthesis(w, noise_mode)

    def get_w_avg(self) -> torch.Tensor | None:
        """Get the average W latent for truncation.

        Returns:
            Average W latent or None if not available.
        """
        if hasattr(self.generator, "mapping") and hasattr(self.generator.mapping, "w_avg"):
            return self.generator.mapping.w_avg
        elif hasattr(self.generator, "mean_latent"):
            return self.generator.mean_latent(4096)
        return None


def load_stylegan2(
    model_name_or_path: str | Path,
    device: str | torch.device = "cuda",
    **kwargs: Any,
) -> StyleGAN2GeneratorWrapper:
    """Load a pretrained StyleGAN2 generator.

    Args:
        model_name_or_path: Either a model name from registry
            (e.g., "stylegan2-ffhq-1024") or path to checkpoint.
        device: Device to load model on.
        **kwargs: Additional arguments passed to wrapper.

    Returns:
        Wrapped StyleGAN2 generator.

    Example:
        >>> # From registry
        >>> generator = load_stylegan2("stylegan2-ffhq-1024")
        >>>
        >>> # From local file
        >>> generator = load_stylegan2("/path/to/checkpoint.pt")
        >>>
        >>> # Generate an image
        >>> z = torch.randn(1, 512, device="cuda")
        >>> image = generator(z)

    Note:
        Students should implement the actual model architecture.
        This function provides the loading infrastructure.
    """
    # Determine if it's a path or model name
    path = Path(model_name_or_path)
    if path.exists():
        checkpoint_path = path
    else:
        # Try to get from registry
        try:
            checkpoint_path = get_model_path(str(model_name_or_path))
            if not checkpoint_path.exists():
                download_model(str(model_name_or_path))
        except ValueError:
            raise ValueError(
                f"'{model_name_or_path}' is not a valid path or model name. "
                f"Use list_available_models() to see available models."
            )

    # Load checkpoint
    print(f"Loading StyleGAN2 from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "g_ema" in checkpoint:
        # NVIDIA format
        state_dict = checkpoint["g_ema"]
        w_dim = 512
        img_resolution = _infer_resolution(state_dict)
    elif "generator" in checkpoint:
        # Our training format
        state_dict = checkpoint["generator"]
        w_dim = checkpoint.get("w_dim", 512)
        img_resolution = checkpoint.get("img_resolution", 1024)
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Lightning format
        state_dict = checkpoint["state_dict"]
        w_dim = 512
        img_resolution = 1024
    else:
        # Assume it's a raw state dict
        state_dict = checkpoint
        w_dim = 512
        img_resolution = _infer_resolution(state_dict)

    # Create generator
    # Note: Students need to implement the actual generator
    generator = _create_generator(
        state_dict,
        w_dim=w_dim,
        img_resolution=img_resolution,
    )
    generator.load_state_dict(state_dict, strict=False)
    generator = generator.to(device)
    generator.eval()

    # Calculate num_ws based on resolution
    # For 1024: 18, for 256: 14, etc.
    num_ws = 2 * int(torch.log2(torch.tensor(img_resolution)).item()) - 2

    wrapper = StyleGAN2GeneratorWrapper(
        generator,
        w_dim=w_dim,
        img_resolution=img_resolution,
        num_ws=num_ws,
        **kwargs,
    )

    return wrapper


def _infer_resolution(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer image resolution from state dict.

    Args:
        state_dict: Model state dict.

    Returns:
        Inferred image resolution.
    """
    # Look for synthesis layers to infer resolution
    for key in state_dict.keys():
        if "synthesis" in key and "torgb" in key.lower():
            # Parse resolution from layer name
            # e.g., "synthesis.b1024.torgb.weight" -> 1024
            parts = key.split(".")
            for part in parts:
                if part.startswith("b") and part[1:].isdigit():
                    return int(part[1:])

    # Default to 1024
    return 1024


def _create_generator(
    state_dict: dict[str, torch.Tensor],
    w_dim: int,
    img_resolution: int,
) -> nn.Module:
    """Create generator architecture matching state dict.

    Args:
        state_dict: State dict to match.
        w_dim: W latent dimension.
        img_resolution: Output resolution.

    Returns:
        Generator module (uninitialized weights).

    Note:
        This creates a minimal generator wrapper that can load
        various StyleGAN2 checkpoint formats. For full architecture,
        copy from rosinality or NVIDIA implementations.
    """
    # Try to use rosinality's implementation if available
    try:
        from models.stylegan2 import Generator

        return Generator(
            size=img_resolution,
            style_dim=w_dim,
            n_mlp=8,
        )
    except ImportError:
        pass

    # Create a minimal generator that can be loaded with state dict
    class MinimalStyleGAN2Generator(nn.Module):
        """Minimal StyleGAN2 generator for loading pretrained weights.

        This is a simplified implementation that provides the basic
        interface. For full functionality, use rosinality's implementation.
        """

        def __init__(self, size: int, style_dim: int, n_mlp: int = 8):
            super().__init__()
            self.size = size
            self.style_dim = style_dim
            self.n_mlp = n_mlp

            # Calculate number of style layers
            self.n_styles = 2 * int(torch.log2(torch.tensor(size)).item()) - 2

            # Mapping network
            layers = [nn.Linear(style_dim, style_dim), nn.LeakyReLU(0.2)]
            for _ in range(n_mlp - 1):
                layers.extend([nn.Linear(style_dim, style_dim), nn.LeakyReLU(0.2)])
            self.style = nn.Sequential(*layers)

            # Constant input
            self.input = nn.Parameter(torch.randn(1, 512, 4, 4))

            # Synthesis layers (simplified - just store as buffer for weight loading)
            self._synthesis_layers = nn.ModuleDict()
            resolutions = [
                4 * 2**i for i in range(int(torch.log2(torch.tensor(size // 4)).item()) + 1)
            ]

            channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256,
                128: 128,
                256: 64,
                512: 32,
                1024: 16,
            }

            in_ch = 512
            for res in resolutions:
                out_ch = channels.get(res, 512)
                # Create placeholder synthesis layers
                self._synthesis_layers[f"b{res}"] = nn.ModuleDict(
                    {
                        "conv0": nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                        "conv1": nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                        "torgb": nn.Conv2d(out_ch, 3, 1, 1, 0),
                    }
                )
                in_ch = out_ch

            self._w_avg = None

        def mapping(
            self,
            z: torch.Tensor,
            c: torch.Tensor | None = None,
            truncation_psi: float = 1.0,
        ) -> torch.Tensor:
            """Map Z to W space."""
            w = self.style(z)

            if truncation_psi < 1.0 and self._w_avg is not None:
                w = self._w_avg + truncation_psi * (w - self._w_avg)

            # Broadcast to W+
            return w.unsqueeze(1).repeat(1, self.n_styles, 1)

        def synthesis(
            self,
            w: torch.Tensor,
            noise_mode: str = "const",
        ) -> torch.Tensor:
            """Synthesize images from W latents.

            Note: This is a placeholder. For actual synthesis,
            use the full rosinality or NVIDIA implementation.
            """
            batch_size = w.shape[0]

            # Start from constant input
            x = self.input.repeat(batch_size, 1, 1, 1)

            # Simple forward through layers (placeholder)
            for name, block in self._synthesis_layers.items():
                if hasattr(block, "conv0"):
                    x = nn.functional.leaky_relu(block["conv0"](x), 0.2)
                if hasattr(block, "conv1"):
                    x = nn.functional.leaky_relu(block["conv1"](x), 0.2)
                # Upsample to next resolution
                res = int(name[1:])
                if res < self.size:
                    x = nn.functional.interpolate(
                        x, scale_factor=2, mode="bilinear", align_corners=False
                    )

            # Final to RGB
            last_block = list(self._synthesis_layers.values())[-1]
            return last_block["torgb"](x)

        def forward(
            self,
            z: torch.Tensor,
            c: torch.Tensor | None = None,
            truncation_psi: float = 1.0,
            noise_mode: str = "const",
        ) -> torch.Tensor:
            """Generate images from Z latents."""
            w = self.mapping(z, c, truncation_psi)
            return self.synthesis(w, noise_mode)

        def mean_latent(self, n_samples: int = 4096) -> torch.Tensor:
            """Compute mean W latent for truncation."""
            device = next(self.parameters()).device
            z = torch.randn(n_samples, self.style_dim, device=device)
            w = self.style(z)
            self._w_avg = w.mean(dim=0, keepdim=True)
            return self._w_avg

    return MinimalStyleGAN2Generator(
        size=img_resolution,
        style_dim=w_dim,
        n_mlp=8,
    )

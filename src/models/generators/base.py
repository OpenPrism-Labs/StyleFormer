"""Base generator interface for face synthesis.

Defines the abstract interface for generators.
All generator implementations should follow this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseGenerator(ABC, nn.Module):
    """Abstract base class for image generators.

    All generator implementations should inherit from this class
    and implement the required methods.

    Attributes:
        z_dim: Dimension of Z latent space.
        w_dim: Dimension of W latent space.
        img_resolution: Output image resolution.
        num_ws: Number of style vectors for W+.

    Example:
        >>> class MyGenerator(BaseGenerator):
        ...     def __init__(self, resolution=1024):
        ...         super().__init__(z_dim=512, w_dim=512, img_resolution=resolution)
        ...         self.mapping = MappingNetwork(...)
        ...         self.synthesis = SynthesisNetwork(...)
        ...
        ...     def forward(self, z, truncation_psi=1.0):
        ...         w = self.mapping(z, truncation_psi=truncation_psi)
        ...         return self.synthesis(w)
    """

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        img_resolution: int = 1024,
        img_channels: int = 3,
    ) -> None:
        """Initialize base generator.

        Args:
            z_dim: Dimension of Z latent space.
            w_dim: Dimension of W latent space.
            img_resolution: Output image resolution.
            img_channels: Number of output channels.
        """
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        # num_ws = 2 * log2(resolution) - 2
        self.num_ws = 2 * int(torch.log2(torch.tensor(img_resolution)).item()) - 2

    @abstractmethod
    def mapping(
        self,
        z: torch.Tensor,
        c: torch.Tensor | None = None,
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        """Map Z latents to W latents.

        Args:
            z: Latent codes in Z space (B, z_dim).
            c: Class labels (optional, for conditional generation).
            truncation_psi: Truncation factor (1.0 = no truncation).

        Returns:
            W latent codes (B, num_ws, w_dim).
        """
        pass

    @abstractmethod
    def synthesis(
        self,
        w: torch.Tensor,
        noise_mode: str = "const",
    ) -> torch.Tensor:
        """Synthesize images from W latents.

        Args:
            w: W latent codes (B, num_ws, w_dim) or (B, w_dim).
            noise_mode: Noise mode ('const', 'random', 'none').

        Returns:
            Generated images (B, 3, H, W) in range [-1, 1].
        """
        pass

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor | None = None,
        truncation_psi: float = 1.0,
        noise_mode: str = "const",
    ) -> torch.Tensor:
        """Generate images from Z latents.

        Args:
            z: Z latent codes (B, z_dim).
            c: Class labels (optional).
            truncation_psi: Truncation factor.
            noise_mode: Noise mode for synthesis.

        Returns:
            Generated images (B, 3, H, W).
        """
        w = self.mapping(z, c, truncation_psi)
        return self.synthesis(w, noise_mode)

    def get_w_avg(self) -> torch.Tensor | None:
        """Get average W latent for truncation.

        Returns:
            Average W latent (w_dim,) or None if not available.
        """
        return None

    def get_config(self) -> dict[str, Any]:
        """Get generator configuration.

        Returns:
            Configuration dictionary.
        """
        return {
            "z_dim": self.z_dim,
            "w_dim": self.w_dim,
            "img_resolution": self.img_resolution,
            "img_channels": self.img_channels,
            "num_ws": self.num_ws,
            "class_name": self.__class__.__name__,
        }


class SynthesisLayer(nn.Module):
    """Single synthesis layer in StyleGAN2.

    Each layer applies style modulation and convolution to
    progressively build up the output image.

    TODO (Students):
        Implement synthesis layer with:
        - Style modulation (modulated convolution)
        - Noise injection
        - Activation (LeakyReLU)

        See StyleGAN2 paper and rosinality implementation for details.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        resolution: int,
        kernel_size: int = 3,
    ) -> None:
        """Initialize synthesis layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            w_dim: Dimension of style vector.
            resolution: Output resolution of this layer.
            kernel_size: Convolution kernel size.
        """
        super().__init__()
        raise NotImplementedError(
            "Students: Implement SynthesisLayer. "
            "See StyleGAN2 paper for details."
        )

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply synthesis layer.

        Args:
            x: Input feature map.
            w: Style vector.
            noise: Noise tensor (optional).

        Returns:
            Output feature map.
        """
        raise NotImplementedError


class ToRGB(nn.Module):
    """Convert features to RGB image.

    Final layer in each resolution block that converts
    feature maps to RGB output.

    TODO (Students):
        Implement ToRGB layer:
        - Modulated 1x1 convolution
        - Style-based modulation
    """

    def __init__(self, in_channels: int, w_dim: int) -> None:
        """Initialize ToRGB.

        Args:
            in_channels: Number of input channels.
            w_dim: Style dimension.
        """
        super().__init__()
        raise NotImplementedError("Students: Implement ToRGB layer.")

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Convert features to RGB.

        Args:
            x: Feature map.
            w: Style vector.

        Returns:
            RGB image (B, 3, H, W).
        """
        raise NotImplementedError

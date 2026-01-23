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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Style modulation (affine transform from w to style)
        self.affine = nn.Linear(w_dim, in_channels)

        # Modulated convolution weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_gain = 1.0 / (in_channels * kernel_size**2) ** 0.5

        # Noise injection
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.register_buffer("noise_const", torch.randn(1, 1, resolution, resolution))

        # Bias and activation
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply synthesis layer.

        Args:
            x: Input feature map (B, C_in, H, W).
            w: Style vector (B, w_dim).
            noise: Noise tensor (optional).

        Returns:
            Output feature map (B, C_out, H, W).
        """
        batch_size = x.shape[0]

        # Get style modulation
        style = self.affine(w)  # (B, in_channels)

        # Modulate weights
        weight = self.weight * self.weight_gain
        # Modulate: weight * style (per-sample modulation)
        weight = weight.unsqueeze(0) * style.view(batch_size, 1, -1, 1, 1)

        # Demodulation (normalize)
        demod = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        weight = weight * demod.view(batch_size, -1, 1, 1, 1)

        # Group convolution (fused for efficiency)
        weight = weight.view(
            batch_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch_size * self.in_channels, x.shape[2], x.shape[3])
        x = nn.functional.conv2d(x, weight, padding=self.padding, groups=batch_size)
        x = x.view(batch_size, self.out_channels, x.shape[2], x.shape[3])

        # Noise injection
        if noise is None:
            noise = self.noise_const
            if noise.shape[2:] != x.shape[2:]:
                noise = torch.randn(batch_size, 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_strength * noise

        # Bias and activation
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.act(x)

        return x


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
        self.in_channels = in_channels
        self.w_dim = w_dim

        # Style modulation
        self.affine = nn.Linear(w_dim, in_channels)

        # Modulated 1x1 convolution to RGB
        self.weight = nn.Parameter(torch.randn(3, in_channels, 1, 1))
        self.weight_gain = 1.0 / in_channels**0.5

        # Bias
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Convert features to RGB.

        Args:
            x: Feature map (B, C, H, W).
            w: Style vector (B, w_dim).

        Returns:
            RGB image (B, 3, H, W).
        """
        batch_size = x.shape[0]

        # Get style modulation
        style = self.affine(w)  # (B, in_channels)

        # Modulate weights
        weight = self.weight * self.weight_gain
        weight = weight.unsqueeze(0) * style.view(batch_size, 1, -1, 1, 1)

        # Demodulation
        demod = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        weight = weight * demod.view(batch_size, -1, 1, 1, 1)

        # Group convolution
        weight = weight.view(batch_size * 3, self.in_channels, 1, 1)
        x = x.view(1, batch_size * self.in_channels, x.shape[2], x.shape[3])
        x = nn.functional.conv2d(x, weight, padding=0, groups=batch_size)
        x = x.view(batch_size, 3, x.shape[2], x.shape[3])

        # Add bias
        x = x + self.bias.view(1, -1, 1, 1)

        return x

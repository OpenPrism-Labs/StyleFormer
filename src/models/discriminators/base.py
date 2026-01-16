"""Base discriminator interface.

Defines the abstract interface for discriminators used in
GAN-based training of face transformation models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseDiscriminator(ABC, nn.Module):
    """Abstract base class for discriminators.

    Discriminators classify images as real or fake and can provide
    feature maps for perceptual losses.

    Attributes:
        img_resolution: Expected input image resolution.
        img_channels: Number of input channels.

    Example:
        >>> class MyDiscriminator(BaseDiscriminator):
        ...     def __init__(self, resolution=256):
        ...         super().__init__(img_resolution=resolution)
        ...         self.blocks = nn.ModuleList([...])
        ...
        ...     def forward(self, x):
        ...         for block in self.blocks:
        ...             x = block(x)
        ...         return x  # Real/fake logit

    Implementation Guide:
        Common discriminator architectures:
        1. PatchGAN: Multi-scale patch-based discrimination
        2. StyleGAN2 Discriminator: Residual blocks with downsampling
        3. Multi-scale: Multiple discriminators at different resolutions
    """

    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
    ) -> None:
        """Initialize base discriminator.

        Args:
            img_resolution: Expected input resolution.
            img_channels: Number of input channels.
        """
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Discriminate real vs fake images.

        Args:
            x: Input images (B, 3, H, W).
            c: Class labels (optional, for conditional discrimination).

        Returns:
            Discrimination scores (B, 1) or (B, H', W') for PatchGAN.
        """
        pass

    def get_features(
        self,
        x: torch.Tensor,
        layers: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """Extract intermediate features.

        Useful for perceptual/feature matching losses.

        Args:
            x: Input images.
            layers: Which layers to extract features from.
                If None, returns features from all layers.

        Returns:
            List of feature tensors from requested layers.
        """
        raise NotImplementedError(
            "Students: Implement feature extraction if using perceptual losses."
        )

    def get_config(self) -> dict[str, Any]:
        """Get discriminator configuration."""
        return {
            "img_resolution": self.img_resolution,
            "img_channels": self.img_channels,
            "class_name": self.__class__.__name__,
        }


class ResidualBlock(nn.Module):
    """Residual block for discriminator.

    Standard residual block with downsampling option.

    TODO (Students):
        Implement residual block:
        ```python
        def __init__(self, in_channels, out_channels, downsample=False):
            super().__init__()
            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride) if downsample or in_channels != out_channels else nn.Identity()
            self.act = nn.LeakyReLU(0.2)

        def forward(self, x):
            skip = self.skip(x)
            x = self.act(self.conv1(x))
            x = self.conv2(x)
            return self.act(x + skip)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
    ) -> None:
        """Initialize residual block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            downsample: Whether to downsample spatial dimensions.
        """
        super().__init__()
        raise NotImplementedError("Students: Implement ResidualBlock.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        raise NotImplementedError


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer.

    Computes standard deviation across minibatch and appends
    as an additional feature channel. Used in StyleGAN discriminator.

    TODO (Students):
        Implement minibatch std dev:
        ```python
        def forward(self, x):
            B, C, H, W = x.shape
            std = x.std(dim=0, keepdim=True).mean()
            std_map = std.expand(B, 1, H, W)
            return torch.cat([x, std_map], dim=1)
        ```
    """

    def __init__(self, group_size: int = 4) -> None:
        """Initialize minibatch std dev.

        Args:
            group_size: Size of groups for std computation.
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add minibatch std dev feature.

        Args:
            x: Input features (B, C, H, W).

        Returns:
            Features with std dev channel (B, C+1, H, W).
        """
        raise NotImplementedError("Students: Implement MinibatchStdDev.")

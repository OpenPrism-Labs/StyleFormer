"""Base encoder interface for face transformation.

Defines the abstract interface that all encoders must implement.
This ensures consistency across different encoder architectures
(e4e, pSp, ReStyle, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Abstract base class for image-to-latent encoders.

    All encoder implementations should inherit from this class and
    implement the required methods.

    Attributes:
        w_dim: Dimension of W latent space (typically 512).
        num_ws: Number of style vectors in W+ space (depends on resolution).
        input_size: Expected input image size.

    Example:
        >>> class MyEncoder(BaseEncoder):
        ...     def __init__(self):
        ...         super().__init__(w_dim=512, num_ws=18, input_size=256)
        ...         self.backbone = ResNet50()
        ...         self.style_layers = nn.ModuleList([...])
        ...
        ...     def forward(self, x):
        ...         features = self.backbone(x)
        ...         styles = [layer(feat) for layer, feat in zip(...)]
        ...         return torch.stack(styles, dim=1)

    Implementation Guide:
        1. The encoder should accept images of shape (B, 3, H, W)
        2. Images are normalized to [-1, 1] range
        3. Output should be W+ latents of shape (B, num_ws, w_dim)
        4. The backbone typically uses a pretrained network (ResNet, etc.)
        5. Style layers (map2style) convert features to W codes
    """

    def __init__(
        self,
        w_dim: int = 512,
        num_ws: int = 18,
        input_size: int = 256,
    ) -> None:
        """Initialize base encoder.

        Args:
            w_dim: Dimension of W latent space.
            num_ws: Number of style vectors for W+ space.
            input_size: Expected input image size.
        """
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.input_size = input_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to W+ latent space.

        Args:
            x: Input images (B, 3, H, W) normalized to [-1, 1].

        Returns:
            W+ latent codes (B, num_ws, w_dim).
        """
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward - encode images to latent space.

        Args:
            x: Input images.

        Returns:
            Latent codes.
        """
        return self.forward(x)

    def encode_to_w(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to W space (single vector per image).

        Takes the mean across style layers to produce a single W code.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            W latent codes (B, w_dim).
        """
        w_plus = self.forward(x)
        return w_plus.mean(dim=1)

    def get_config(self) -> dict[str, Any]:
        """Get encoder configuration.

        Returns:
            Dictionary with encoder configuration.
        """
        return {
            "w_dim": self.w_dim,
            "num_ws": self.num_ws,
            "input_size": self.input_size,
            "class_name": self.__class__.__name__,
        }


class GradualStyleBlock(nn.Module):
    """Gradual style block for progressive encoding.

    Converts feature maps to style codes at a specific resolution level.
    Used in e4e and pSp encoders.

    This block progressively refines style codes as resolution increases,
    allowing the encoder to capture both coarse and fine details.

    TODO (Students):
        Implement the gradual style block:
        ```python
        def __init__(self, in_channels, out_channels, spatial):
            super().__init__()
            self.convs = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(0.2),
            )
            self.pool = nn.AdaptiveAvgPool2d((spatial, spatial))
            self.fc = nn.Linear(out_channels * spatial * spatial, out_channels)

        def forward(self, x):
            x = self.convs(x)
            x = self.pool(x)
            x = x.flatten(1)
            return self.fc(x)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial: int,
    ) -> None:
        """Initialize gradual style block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (usually w_dim).
            spatial: Spatial size after pooling.
        """
        super().__init__()
        raise NotImplementedError(
            "Students: Implement GradualStyleBlock. See docstring."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert feature map to style code.

        Args:
            x: Feature map (B, C, H, W).

        Returns:
            Style code (B, out_channels).
        """
        raise NotImplementedError


class Map2Style(nn.Module):
    """Map feature to style code.

    Simple mapping from feature maps to W latent codes.
    Used in various encoder architectures.

    TODO (Students):
        Implement map2style layer:
        ```python
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(out_channels, out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)
        ```
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize map2style.

        Args:
            in_channels: Number of input channels.
            out_channels: Output dimension (w_dim).
        """
        super().__init__()
        raise NotImplementedError("Students: Implement Map2Style. See docstring.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map feature to style.

        Args:
            x: Feature map (B, C, H, W).

        Returns:
            Style code (B, out_channels).
        """
        raise NotImplementedError

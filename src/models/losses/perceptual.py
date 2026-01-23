"""Perceptual and LPIPS losses.

These losses compare images in feature space rather than pixel space,
leading to perceptually better results.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses.base import BaseLoss


class PerceptualLoss(BaseLoss):
    """Perceptual loss using VGG features.

    Compares images in the feature space of a pretrained VGG network.
    This encourages perceptually similar images rather than pixel-exact matches.

    Example:
        >>> perceptual_loss = PerceptualLoss(layers=[2, 7, 12, 21, 30])
        >>> loss = perceptual_loss(generated, target)

    TODO (Students):
        Implement VGG feature extraction:
        ```python
        from torchvision.models import vgg19, VGG19_Weights

        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        # Extract features at specified layers
        ```
    """

    # Default VGG layers for perceptual loss
    DEFAULT_LAYERS = [2, 7, 12, 21, 30]  # After ReLU in each block

    def __init__(
        self,
        layers: list[int] | None = None,
        layer_weights: list[float] | None = None,
        normalize_input: bool = True,
        weight: float = 1.0,
    ) -> None:
        """Initialize perceptual loss.

        Args:
            layers: VGG layer indices for feature extraction.
            layer_weights: Weights for each layer's loss.
            normalize_input: Whether to normalize input to VGG range.
            weight: Overall loss weight.
        """
        super().__init__(weight=weight, name="PerceptualLoss")
        self.layers = layers or self.DEFAULT_LAYERS
        self.layer_weights = layer_weights or [1.0] * len(self.layers)
        self.normalize_input = normalize_input

        # VGG normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        self.vgg: nn.Module | None = None

    def _load_vgg(self) -> nn.Module:
        """Load pretrained VGG network.

        Returns:
            VGG features network.
        """
        try:
            from torchvision.models import vgg19, VGG19_Weights

            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        except ImportError:
            # Fallback for older torchvision versions
            from torchvision.models import vgg19

            vgg = vgg19(pretrained=True).features

        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input from [-1, 1] to VGG range.

        Args:
            x: Input images in [-1, 1].

        Returns:
            Normalized images.
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2

        # Apply VGG normalization
        return (x - self.mean) / self.std

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features at specified layers.

        Args:
            x: Input images.

        Returns:
            List of feature tensors.
        """
        if self.vgg is None:
            self.vgg = self._load_vgg()
            self.vgg = self.vgg.to(x.device)

        if self.normalize_input:
            x = self._normalize(x)

        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)

        return features

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            pred: Predicted images (B, 3, H, W).
            target: Target images (B, 3, H, W).

        Returns:
            Perceptual loss.
        """
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = torch.tensor(0.0, device=pred.device)
        for pred_feat, target_feat, w in zip(pred_features, target_features, self.layer_weights):
            loss = loss + w * F.l1_loss(pred_feat, target_feat)

        return loss


class LPIPSLoss(BaseLoss):
    """LPIPS (Learned Perceptual Image Patch Similarity) loss.

    LPIPS uses learned weights on VGG features that better correlate
    with human perception than hand-designed weights.

    Example:
        >>> lpips_loss = LPIPSLoss(net='vgg')
        >>> loss = lpips_loss(generated, target)

    TODO (Students):
        Option 1: Use the lpips package
        ```python
        import lpips
        self.lpips = lpips.LPIPS(net='vgg')
        ```

        Option 2: Implement from scratch following the paper
        "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
    """

    def __init__(
        self,
        net: str = "vgg",
        weight: float = 1.0,
    ) -> None:
        """Initialize LPIPS loss.

        Args:
            net: Network type ('vgg', 'alex', 'squeeze').
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="LPIPSLoss")
        self.net = net
        self.lpips: nn.Module | None = None

    def _load_lpips(self) -> nn.Module:
        """Load LPIPS network.

        Returns:
            LPIPS model.
        """
        try:
            import lpips

            model = lpips.LPIPS(net=self.net)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return model
        except ImportError:
            raise ImportError("LPIPS package not found. Please install it with: pip install lpips")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute LPIPS loss.

        Args:
            pred: Predicted images (B, 3, H, W) in [-1, 1].
            target: Target images (B, 3, H, W) in [-1, 1].

        Returns:
            LPIPS loss (mean over batch).
        """
        if self.lpips is None:
            self.lpips = self._load_lpips()
            self.lpips = self.lpips.to(pred.device)

        return self.lpips(pred, target).mean()

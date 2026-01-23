"""Identity preservation loss using face recognition."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses.base import BaseLoss


class IdentityLoss(BaseLoss):
    """Identity preservation loss using ArcFace.

    Encourages generated images to preserve the identity of the source
    by maximizing cosine similarity between face embeddings.

    Loss = 1 - cosine_similarity(embed(source), embed(generated))

    Example:
        >>> id_loss = IdentityLoss(model_path="weights/arcface.pt")
        >>> loss = id_loss(generated, source)
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        weight: float = 1.0,
    ) -> None:
        """Initialize identity loss.

        Args:
            model_path: Path to ArcFace weights.
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="IdentityLoss")
        self.model_path = model_path
        self.arcface: nn.Module | None = None
        self.input_size = 112

    def _load_arcface(self) -> nn.Module:
        """Load ArcFace model.

        Returns:
            ArcFace model.
        """
        if self.model_path is not None:
            from src.models.pretrained import load_arcface

            wrapper = load_arcface(self.model_path)
            return wrapper.model
        else:
            # Try to use insightface if available
            try:
                from insightface.recognition.arcface_torch import iresnet100

                model = iresnet100(num_features=512)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                return model
            except ImportError:
                raise ImportError(
                    "ArcFace model path not provided and insightface not available. "
                    "Please either provide a model_path or install insightface: "
                    "pip install insightface"
                )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for ArcFace.

        Args:
            x: Images (B, 3, H, W) in [-1, 1].

        Returns:
            Preprocessed images.
        """
        # Resize to ArcFace input size
        if x.shape[-1] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def _extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract identity embedding.

        Args:
            x: Face images.

        Returns:
            Normalized embeddings (B, 512).
        """
        if self.arcface is None:
            self.arcface = self._load_arcface()
            self.arcface = self.arcface.to(x.device)
            self.arcface.eval()

        x = self._preprocess(x)
        embedding = self.arcface(x)
        return F.normalize(embedding, p=2, dim=1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute identity loss.

        Note: pred is generated image, target is source image.
        We want the generated image to have the same identity as source.

        Args:
            pred: Generated images (B, 3, H, W).
            target: Source images (B, 3, H, W).

        Returns:
            Identity loss (1 - cosine_similarity).
        """
        pred_embed = self._extract_embedding(pred)
        target_embed = self._extract_embedding(target)

        # Cosine similarity
        similarity = F.cosine_similarity(pred_embed, target_embed, dim=1)

        # Loss is 1 - similarity (want to maximize similarity)
        return (1 - similarity).mean()


class MultiScaleIdentityLoss(BaseLoss):
    """Multi-scale identity loss.

    Computes identity loss at multiple resolutions for
    more robust identity preservation.
    """

    def __init__(
        self,
        scales: list[int] | None = None,
        model_path: Path | str | None = None,
        weight: float = 1.0,
    ) -> None:
        """Initialize multi-scale identity loss.

        Args:
            scales: List of scales to compute loss at.
            model_path: Path to ArcFace weights.
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="MultiScaleIdentityLoss")
        self.scales = scales or [1, 2, 4]  # 1x, 0.5x, 0.25x
        self.id_loss = IdentityLoss(model_path=model_path)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute multi-scale identity loss.

        Args:
            pred: Generated images.
            target: Source images.

        Returns:
            Multi-scale identity loss.
        """
        loss = torch.tensor(0.0, device=pred.device)

        for scale in self.scales:
            if scale == 1:
                pred_scaled = pred
                target_scaled = target
            else:
                size = pred.shape[-1] // scale
                pred_scaled = F.interpolate(pred, size=(size, size), mode="bilinear")
                target_scaled = F.interpolate(target, size=(size, size), mode="bilinear")

            loss = loss + self.id_loss(pred_scaled, target_scaled)

        return loss / len(self.scales)

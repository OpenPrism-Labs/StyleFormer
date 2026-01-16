"""Base loss classes and reconstruction losses."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(ABC, nn.Module):
    """Abstract base class for all losses.

    All loss implementations should inherit from this class.

    Attributes:
        weight: Loss weight for combining multiple losses.
        name: Name identifier for logging.
    """

    def __init__(self, weight: float = 1.0, name: str | None = None) -> None:
        """Initialize base loss.

        Args:
            weight: Loss weight.
            name: Loss name for logging.
        """
        super().__init__()
        self.weight = weight
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            pred: Predicted/generated tensor.
            target: Target/ground truth tensor.
            **kwargs: Additional arguments.

        Returns:
            Scalar loss value.
        """
        pass

    def weighted_forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute weighted loss.

        Args:
            pred: Predicted tensor.
            target: Target tensor.
            **kwargs: Additional arguments.

        Returns:
            Weighted loss value.
        """
        return self.weight * self.forward(pred, target, **kwargs)


class ReconstructionLoss(BaseLoss):
    """Combined reconstruction loss.

    Combines L1 and L2 losses with configurable weights.

    Example:
        >>> loss_fn = ReconstructionLoss(l1_weight=1.0, l2_weight=0.5)
        >>> loss = loss_fn(generated, target)
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.0,
        weight: float = 1.0,
    ) -> None:
        """Initialize reconstruction loss.

        Args:
            l1_weight: Weight for L1 loss.
            l2_weight: Weight for L2 loss.
            weight: Overall loss weight.
        """
        super().__init__(weight=weight, name="ReconstructionLoss")
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            pred: Predicted images (B, C, H, W).
            target: Target images (B, C, H, W).

        Returns:
            Reconstruction loss.
        """
        loss = torch.tensor(0.0, device=pred.device)

        if self.l1_weight > 0:
            loss = loss + self.l1_weight * F.l1_loss(pred, target)

        if self.l2_weight > 0:
            loss = loss + self.l2_weight * F.mse_loss(pred, target)

        return loss


class L1Loss(BaseLoss):
    """L1 (Mean Absolute Error) loss.

    Simple pixel-wise L1 loss for image reconstruction.
    """

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize L1 loss.

        Args:
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="L1Loss")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute L1 loss.

        Args:
            pred: Predicted tensor.
            target: Target tensor.

        Returns:
            L1 loss value.
        """
        return F.l1_loss(pred, target)


class L2Loss(BaseLoss):
    """L2 (Mean Squared Error) loss.

    Pixel-wise L2 loss for image reconstruction.
    """

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize L2 loss.

        Args:
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="L2Loss")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute L2 loss.

        Args:
            pred: Predicted tensor.
            target: Target tensor.

        Returns:
            L2 loss value.
        """
        return F.mse_loss(pred, target)


class LatentLoss(BaseLoss):
    """Latent space regularization loss.

    Encourages latent codes to stay close to the average latent
    or to follow certain distributions.

    Example:
        >>> latent_loss = LatentLoss(w_avg=model.w_avg)
        >>> loss = latent_loss(predicted_w, w_avg)
    """

    def __init__(
        self,
        w_avg: torch.Tensor | None = None,
        weight: float = 1.0,
    ) -> None:
        """Initialize latent loss.

        Args:
            w_avg: Average W latent for regularization.
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="LatentLoss")
        if w_avg is not None:
            self.register_buffer("w_avg", w_avg)
        else:
            self.w_avg = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute latent regularization loss.

        Args:
            pred: Predicted latent codes (B, num_ws, w_dim) or (B, w_dim).
            target: Target latent (optional, uses w_avg if None).

        Returns:
            Latent loss.
        """
        if target is None:
            if self.w_avg is None:
                raise ValueError("No target provided and w_avg not set")
            target = self.w_avg

        # Expand target to match pred shape if needed
        if target.dim() < pred.dim():
            target = target.unsqueeze(0).expand_as(pred)

        return F.mse_loss(pred, target)


class DeltaRegularization(BaseLoss):
    """Delta regularization for latent edits.

    Encourages small, sparse edits in latent space rather than
    large global changes.

    Used to ensure edits are minimal and targeted.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        weight: float = 1.0,
    ) -> None:
        """Initialize delta regularization.

        Args:
            threshold: Threshold below which deltas are not penalized.
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="DeltaRegularization")
        self.threshold = threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute delta regularization.

        Args:
            pred: Edited latent codes.
            target: Original latent codes.

        Returns:
            Regularization loss.
        """
        delta = pred - target
        delta_norm = delta.norm(dim=-1)

        # Apply threshold
        if self.threshold > 0:
            delta_norm = F.relu(delta_norm - self.threshold)

        return delta_norm.mean()

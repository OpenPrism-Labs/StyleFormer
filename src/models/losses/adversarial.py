"""Adversarial losses for GAN training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses.base import BaseLoss


class AdversarialLoss(BaseLoss):
    """Base adversarial loss.

    Can be configured for different GAN loss types:
    - 'vanilla': Original GAN loss (BCE)
    - 'lsgan': Least squares GAN
    - 'wgan': Wasserstein GAN
    - 'hinge': Hinge loss
    - 'nonsaturating': Non-saturating logistic loss
    """

    def __init__(
        self,
        loss_type: str = "nonsaturating",
        weight: float = 1.0,
    ) -> None:
        """Initialize adversarial loss.

        Args:
            loss_type: Type of GAN loss.
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="AdversarialLoss")
        self.loss_type = loss_type

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | bool,
        **kwargs,
    ) -> torch.Tensor:
        """Compute adversarial loss.

        Args:
            pred: Discriminator output.
            target: True for real, False for fake (or target tensor).

        Returns:
            Adversarial loss.
        """
        # Convert bool target to tensor
        if isinstance(target, bool):
            target_val = 1.0 if target else 0.0
            target = torch.full_like(pred, target_val)

        if self.loss_type == "vanilla":
            return F.binary_cross_entropy_with_logits(pred, target)

        elif self.loss_type == "lsgan":
            return F.mse_loss(pred, target)

        elif self.loss_type == "wgan":
            return -pred.mean() if target.mean() > 0.5 else pred.mean()

        elif self.loss_type == "hinge":
            if target.mean() > 0.5:  # Real
                return F.relu(1 - pred).mean()
            else:  # Fake
                return F.relu(1 + pred).mean()

        elif self.loss_type == "nonsaturating":
            if target.mean() > 0.5:  # Real
                return F.softplus(-pred).mean()
            else:  # Fake
                return F.softplus(pred).mean()

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class NonSaturatingLoss(BaseLoss):
    """Non-saturating logistic GAN loss.

    The standard loss used in StyleGAN2:
    - D_real: softplus(-D(real))
    - D_fake: softplus(D(fake))
    - G: softplus(-D(fake))
    """

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize non-saturating loss.

        Args:
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="NonSaturatingLoss")

    def forward(
        self,
        pred: torch.Tensor,
        target: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Compute non-saturating loss.

        Args:
            pred: Discriminator logits.
            target: True for real, False for fake.

        Returns:
            Loss value.
        """
        if target:  # Real
            return F.softplus(-pred).mean()
        else:  # Fake
            return F.softplus(pred).mean()

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute generator loss.

        Args:
            fake_pred: Discriminator output for fake images.

        Returns:
            Generator loss.
        """
        return F.softplus(-fake_pred).mean()

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator loss.

        Args:
            real_pred: Discriminator output for real images.
            fake_pred: Discriminator output for fake images.

        Returns:
            Discriminator loss.
        """
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()
        return real_loss + fake_loss


class HingeLoss(BaseLoss):
    """Hinge loss for GAN training.

    Used in SAGAN and other modern GANs:
    - D_real: max(0, 1 - D(real))
    - D_fake: max(0, 1 + D(fake))
    - G: -D(fake)
    """

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize hinge loss.

        Args:
            weight: Loss weight.
        """
        super().__init__(weight=weight, name="HingeLoss")

    def forward(
        self,
        pred: torch.Tensor,
        target: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Compute hinge loss.

        Args:
            pred: Discriminator output.
            target: True for real, False for fake.

        Returns:
            Hinge loss.
        """
        if target:  # Real
            return F.relu(1 - pred).mean()
        else:  # Fake
            return F.relu(1 + pred).mean()

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Generator loss: maximize D(fake).

        Args:
            fake_pred: Discriminator output for fakes.

        Returns:
            Generator loss.
        """
        return -fake_pred.mean()


class R1Regularization(BaseLoss):
    """R1 gradient penalty regularization.

    Regularizes discriminator by penalizing gradients on real images.
    Used in StyleGAN2 for stable training.

    R1 = (grad_D(real))^2

    Example:
        >>> r1_reg = R1Regularization(weight=10.0)
        >>> # Compute during training
        >>> real.requires_grad_(True)
        >>> real_pred = discriminator(real)
        >>> r1_loss = r1_reg(real_pred, real)
    """

    def __init__(
        self,
        weight: float = 10.0,
        interval: int = 16,
    ) -> None:
        """Initialize R1 regularization.

        Args:
            weight: Regularization weight (gamma).
            interval: Apply every N iterations (lazy regularization).
        """
        super().__init__(weight=weight, name="R1Regularization")
        self.interval = interval

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute R1 regularization.

        Args:
            pred: Discriminator output for real images.
            target: Real images (must have requires_grad=True).

        Returns:
            R1 regularization loss.
        """
        # Compute gradients
        grad = torch.autograd.grad(
            outputs=pred.sum(),
            inputs=target,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute gradient penalty
        grad_penalty = grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()

        return grad_penalty


class PathLengthRegularization(BaseLoss):
    """Path length regularization for generator.

    Encourages smooth latent space by penalizing changes in
    image when latent changes.

    Used in StyleGAN2 for better interpolation.

    TODO (Students):
        Implement path length regularization:
        ```python
        def forward(self, fake_images, latents):
            noise = torch.randn_like(fake_images) / sqrt(H * W)
            grad = torch.autograd.grad(
                outputs=(fake_images * noise).sum(),
                inputs=latents,
                create_graph=True,
            )[0]
            path_lengths = grad.pow(2).mean(dim=-1).sqrt()
            # Compare to running mean
            pl_mean = self.pl_mean.lerp(path_lengths.mean(), 0.01)
            self.pl_mean.copy_(pl_mean.detach())
            return (path_lengths - pl_mean).pow(2).mean()
        ```
    """

    def __init__(
        self,
        weight: float = 2.0,
        interval: int = 4,
    ) -> None:
        """Initialize path length regularization.

        Args:
            weight: Regularization weight.
            interval: Apply every N iterations.
        """
        super().__init__(weight=weight, name="PathLengthRegularization")
        self.interval = interval
        self.register_buffer("pl_mean", torch.zeros(1))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute path length regularization.

        Args:
            pred: Generated images (B, 3, H, W).
            target: Latent codes used to generate images (B, num_ws, w_dim).

        Returns:
            Path length regularization loss.
        """
        # Get image dimensions
        batch_size = pred.shape[0]
        height, width = pred.shape[2], pred.shape[3]

        # Create noise for gradient computation
        noise = torch.randn_like(pred) / (height * width) ** 0.5

        # Compute gradients of the generator output w.r.t. latents
        grad = torch.autograd.grad(
            outputs=(pred * noise).sum(),
            inputs=target,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute path lengths
        path_lengths = grad.pow(2).sum(dim=-1).mean(dim=-1).sqrt()

        # Update running mean
        pl_mean = self.pl_mean.lerp(path_lengths.mean(), 0.01)
        self.pl_mean.copy_(pl_mean.detach())

        # Compute loss as deviation from mean
        path_penalty = (path_lengths - self.pl_mean).pow(2).mean()

        return path_penalty

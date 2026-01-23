"""Base Lightning module for face transformation.

Defines the abstract interface for training face transformation models
using PyTorch Lightning.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Note: Import will fail if pytorch-lightning not installed
# Students should install it: pip install pytorch-lightning
try:
    import pytorch_lightning as pl
except ImportError:
    # Create a dummy base class for type hints
    class pl:  # type: ignore
        class LightningModule:
            pass


class BaseTransformerModule(pl.LightningModule):
    """Abstract base class for face transformation Lightning modules.

    This class defines the interface for training face transformation
    models with PyTorch Lightning. Subclasses should implement the
    specific model architectures and loss computations.

    Attributes:
        encoder: Image-to-latent encoder.
        generator: Latent-to-image generator (e.g., StyleGAN2).
        losses: Dictionary of loss functions.
        loss_weights: Dictionary of loss weights.

    Example:
        >>> class MyTransformer(BaseTransformerModule):
        ...     def __init__(self, config):
        ...         super().__init__(config)
        ...         self.encoder = E4EEncoder()
        ...         self.generator = StyleGAN2Generator()
        ...         self.setup_losses()
        ...
        ...     def training_step(self, batch, batch_idx):
        ...         source = batch['source']
        ...         target_attrs = batch['target_attributes']
        ...         latent = self.encoder(source)
        ...         edited = self.edit_latent(latent, target_attrs)
        ...         generated = self.generator.synthesis(edited)
        ...         loss = self.compute_losses(source, generated)
        ...         return loss

    Implementation Guide:
        1. Initialize encoder, generator, and losses in __init__
        2. Implement training_step for training logic
        3. Implement validation_step for validation
        4. Implement configure_optimizers for optimizer setup
        5. Optionally implement edit_latent for attribute editing
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        w_dim: int = 512,
        num_ws: int = 18,
        **kwargs: Any,
    ) -> None:
        """Initialize base transformer module.

        Args:
            learning_rate: Learning rate for optimizer.
            w_dim: Dimension of W latent space.
            num_ws: Number of style vectors in W+.
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.w_dim = w_dim
        self.num_ws = num_ws

        # To be initialized by subclasses
        self.encoder: nn.Module | None = None
        self.generator: nn.Module | None = None
        self.losses: dict[str, nn.Module] = {}
        self.loss_weights: dict[str, float] = {}

    def setup_losses(self) -> None:
        """Setup loss functions.

        Called during initialization to setup loss functions.
        Subclasses should override this to add their losses.

        Example:
            >>> def setup_losses(self):
            ...     self.losses['reconstruction'] = L1Loss()
            ...     self.losses['perceptual'] = PerceptualLoss()
            ...     self.losses['identity'] = IdentityLoss()
            ...     self.loss_weights = {
            ...         'reconstruction': 1.0,
            ...         'perceptual': 0.8,
            ...         'identity': 0.1,
            ...     }
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Latent codes (B, num_ws, w_dim).
        """
        pass

    @abstractmethod
    def decode(self, w: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to images.

        Args:
            w: Latent codes (B, num_ws, w_dim).

        Returns:
            Generated images (B, 3, H, W).
        """
        pass

    def edit_latent(
        self,
        w: torch.Tensor,
        target_attributes: dict[str, Any],
    ) -> torch.Tensor:
        """Edit latent codes to achieve target attributes.

        Args:
            w: Original latent codes (B, num_ws, w_dim).
            target_attributes: Dictionary of target attribute values.

        Returns:
            Edited latent codes.
        """
        edited = w.clone()

        # Check if we have attribute directions
        if hasattr(self, "attribute_directions") and self.attribute_directions:
            for attr, direction in self.attribute_directions.items():
                if attr in target_attributes:
                    # Get target value and compute edit strength
                    target_value = target_attributes[attr]

                    # Handle different target formats
                    if isinstance(target_value, (int, float)):
                        strength = float(target_value)
                    elif isinstance(target_value, str):
                        # Map string labels to strengths
                        strength_map = {
                            "young": -2.0,
                            "old": 2.0,
                            "female": -2.0,
                            "male": 2.0,
                        }
                        strength = strength_map.get(target_value.lower(), 0.0)
                    elif isinstance(target_value, tuple):
                        # (target_label, strength) format
                        _, strength = target_value
                    else:
                        strength = 0.0

                    # Apply direction
                    if direction.dim() == 1:
                        # W space direction (512,)
                        direction = direction.unsqueeze(0).unsqueeze(0)
                    elif direction.dim() == 2:
                        # W+ space direction (num_ws, 512)
                        direction = direction.unsqueeze(0)

                    edited = edited + strength * direction.to(w.device)

        return edited

    def compute_losses(
        self,
        source: torch.Tensor,
        generated: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            source: Source images.
            generated: Generated images.
            **kwargs: Additional tensors (latents, targets, etc.).

        Returns:
            Dictionary of loss values.
        """
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            weight = self.loss_weights.get(name, 1.0)
            if name == "identity":
                # Identity loss compares source and generated
                loss_dict[name] = weight * loss_fn(generated, source)
            elif name == "reconstruction":
                # Reconstruction loss (if target is source)
                target = kwargs.get("target", source)
                loss_dict[name] = weight * loss_fn(generated, target)
            elif name == "perceptual":
                target = kwargs.get("target", source)
                loss_dict[name] = weight * loss_fn(generated, target)
            else:
                # Generic loss
                target = kwargs.get("target", source)
                loss_dict[name] = weight * loss_fn(generated, target)

        return loss_dict

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch dictionary with 'image', 'attributes', etc.
            batch_idx: Batch index.

        Returns:
            Total loss for backpropagation.
        """
        # Handle both paired and unpaired training
        if "source_image" in batch:
            # Paired training mode
            source = batch["source_image"]
            target = batch.get("target_image", source)
            target_attrs = batch.get("target_attributes", {})
        else:
            # Unpaired training mode
            source = batch["image"]
            target = source  # Reconstruction
            target_attrs = batch.get("target_attributes", {})

        # Encode source image to latent space
        latent = self.encode(source)

        # Edit latent if target attributes provided
        if target_attrs:
            try:
                latent = self.edit_latent(latent, target_attrs)
            except NotImplementedError:
                pass  # Skip editing if not implemented

        # Decode to generate output
        generated = self.decode(latent)

        # Compute losses
        loss_dict = self.compute_losses(source, generated, target=target, latent=latent)

        # Calculate total loss
        total_loss = sum(loss_dict.values())

        # Log losses
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Batch dictionary.
            batch_idx: Batch index.

        Returns:
            Validation loss.
        """
        # Handle both paired and unpaired validation
        if "source_image" in batch:
            source = batch["source_image"]
            target = batch.get("target_image", source)
            target_attrs = batch.get("target_attributes", {})
        else:
            source = batch["image"]
            target = source
            target_attrs = batch.get("target_attributes", {})

        # Encode source image
        latent = self.encode(source)

        # Edit latent if target attributes provided
        if target_attrs:
            try:
                latent = self.edit_latent(latent, target_attrs)
            except NotImplementedError:
                pass

        # Decode to generate output
        generated = self.decode(latent)

        # Compute losses
        loss_dict = self.compute_losses(source, generated, target=target, latent=latent)
        total_loss = sum(loss_dict.values())

        # Log losses
        for name, value in loss_dict.items():
            self.log(f"val/{name}", value, on_epoch=True, prog_bar=False)
        self.log("val/total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and schedulers.

        Returns:
            Optimizer configuration dictionary.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return {"optimizer": optimizer}

    def on_train_epoch_end(self) -> None:
        """Called at end of training epoch.

        Good place for:
        - Generating sample images
        - Computing epoch-level metrics
        - Saving visualizations
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
        target_attributes: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Forward pass for inference.

        Args:
            x: Input images.
            target_attributes: Target attributes for editing.

        Returns:
            Transformed images.
        """
        latent = self.encode(x)
        if target_attributes is not None:
            latent = self.edit_latent(latent, target_attributes)
        return self.decode(latent)


class GANTrainerMixin:
    """Mixin for GAN-based training with discriminator.

    Adds discriminator training logic for adversarial training.
    Use with BaseTransformerModule for GAN-based approaches.

    Example:
        >>> class MyGANModule(GANTrainerMixin, BaseTransformerModule):
        ...     def __init__(self, config):
        ...         super().__init__(config)
        ...         self.discriminator = PatchDiscriminator()
        ...         self.setup_gan_losses()
    """

    discriminator: nn.Module
    adv_loss: nn.Module
    r1_reg: nn.Module | None

    def setup_gan_losses(self) -> None:
        """Setup GAN-specific losses."""
        from src.models.losses import NonSaturatingLoss, R1Regularization

        self.adv_loss = NonSaturatingLoss()
        self.r1_reg = R1Regularization(weight=10.0, interval=16)

    def discriminator_step(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Discriminator training step.

        Args:
            real: Real images.
            fake: Fake/generated images.
            step: Current training step.

        Returns:
            Discriminator loss.
        """
        real_pred = self.discriminator(real)
        fake_pred = self.discriminator(fake.detach())

        d_loss = self.adv_loss(real_pred, True) + self.adv_loss(fake_pred, False)

        # R1 regularization
        if self.r1_reg and step % self.r1_reg.interval == 0:
            real.requires_grad_(True)
            real_pred = self.discriminator(real)
            r1_loss = self.r1_reg(real_pred, real)
            d_loss = d_loss + r1_loss

        return d_loss

    def generator_step(
        self,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Generator adversarial loss.

        Args:
            fake: Generated images.

        Returns:
            Generator adversarial loss.
        """
        fake_pred = self.discriminator(fake)
        return self.adv_loss.generator_loss(fake_pred)

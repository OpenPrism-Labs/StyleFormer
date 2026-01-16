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

        TODO (Students):
            Implement latent editing using one of:
            1. InterFaceGAN: Linear directions in W space
            2. GANSpace: PCA-based directions
            3. StyleSpace: Edit in S space
            4. Learned edit network

            Example with InterFaceGAN:
            ```python
            edited = w.clone()
            for attr, direction in self.attribute_directions.items():
                if attr in target_attributes:
                    strength = self.get_edit_strength(attr, target_attributes[attr])
                    edited = edited + strength * direction
            return edited
            ```
        """
        raise NotImplementedError(
            "Students: Implement latent editing for attribute manipulation. "
            "See docstring for approaches."
        )

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

        TODO (Students):
            Implement training step:
            ```python
            source = batch['image']
            target_attrs = batch.get('target_attributes', {})

            # Encode
            latent = self.encode(source)

            # Edit (if training with attribute editing)
            if target_attrs:
                latent = self.edit_latent(latent, target_attrs)

            # Decode
            generated = self.decode(latent)

            # Compute losses
            loss_dict = self.compute_losses(source, generated)
            total_loss = sum(loss_dict.values())

            # Log losses
            for name, value in loss_dict.items():
                self.log(f'train/{name}', value)
            self.log('train/total_loss', total_loss)

            return total_loss
            ```
        """
        raise NotImplementedError(
            "Students: Implement training_step. See docstring."
        )

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

        TODO (Students):
            Implement validation step similar to training_step.
        """
        raise NotImplementedError(
            "Students: Implement validation_step."
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and schedulers.

        Returns:
            Optimizer configuration dictionary.

        Example:
            >>> def configure_optimizers(self):
            ...     optimizer = torch.optim.Adam(
            ...         self.parameters(),
            ...         lr=self.learning_rate,
            ...     )
            ...     scheduler = torch.optim.lr_scheduler.StepLR(
            ...         optimizer, step_size=10000, gamma=0.5
            ...     )
            ...     return {
            ...         'optimizer': optimizer,
            ...         'lr_scheduler': {
            ...             'scheduler': scheduler,
            ...             'interval': 'step',
            ...         }
            ...     }
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
        """Setup GAN-specific losses.

        TODO (Students):
            ```python
            from src.models.losses import NonSaturatingLoss, R1Regularization

            self.adv_loss = NonSaturatingLoss()
            self.r1_reg = R1Regularization(weight=10.0, interval=16)
            ```
        """
        raise NotImplementedError("Students: Setup GAN losses.")

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

        TODO (Students):
            ```python
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
            ```
        """
        raise NotImplementedError("Students: Implement discriminator step.")

    def generator_step(
        self,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Generator adversarial loss.

        Args:
            fake: Generated images.

        Returns:
            Generator adversarial loss.

        TODO (Students):
            ```python
            fake_pred = self.discriminator(fake)
            return self.adv_loss.generator_loss(fake_pred)
            ```
        """
        raise NotImplementedError("Students: Implement generator step.")

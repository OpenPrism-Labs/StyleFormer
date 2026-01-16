"""Discriminator architectures for adversarial training.

Discriminators are used for:
1. Real/fake discrimination in GAN training
2. Feature extraction for perceptual losses

Available discriminators:
- BaseDiscriminator: Abstract base class

Students should implement discriminators if using GAN-based training.
"""

from src.models.discriminators.base import BaseDiscriminator

__all__ = ["BaseDiscriminator"]

"""Loss functions for face transformation.

This module provides various loss functions:
- Reconstruction losses (L1, L2, LPIPS)
- Identity preservation loss (ArcFace-based)
- Adversarial losses (GAN losses)
- Perceptual losses (VGG-based)
- Latent losses (W-space regularization)

Students should implement or integrate these losses for training.
"""

from src.models.losses.base import (
    BaseLoss,
    ReconstructionLoss,
    L1Loss,
    L2Loss,
)
from src.models.losses.perceptual import PerceptualLoss, LPIPSLoss
from src.models.losses.identity import IdentityLoss
from src.models.losses.adversarial import (
    AdversarialLoss,
    NonSaturatingLoss,
    HingeLoss,
    R1Regularization,
)

__all__ = [
    "BaseLoss",
    "ReconstructionLoss",
    "L1Loss",
    "L2Loss",
    "PerceptualLoss",
    "LPIPSLoss",
    "IdentityLoss",
    "AdversarialLoss",
    "NonSaturatingLoss",
    "HingeLoss",
    "R1Regularization",
]

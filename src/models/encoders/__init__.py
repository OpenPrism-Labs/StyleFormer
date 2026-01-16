"""Encoder architectures for face transformation.

Encoders map real face images to latent representations that can be
manipulated for attribute editing.

Available encoders:
- BaseEncoder: Abstract base class defining the encoder interface
- E4EEncoder: Encoder4Editing - optimized for editability
- PSPEncoder: Pixel2Style2Pixel - direct latent prediction

Students should implement the concrete encoder classes.
"""

from src.models.encoders.base import BaseEncoder

__all__ = ["BaseEncoder"]

"""Generator architectures for face transformation.

Generators synthesize face images from latent representations.
The primary generator is StyleGAN2.

Available generators:
- BaseGenerator: Abstract base class
- StyleGAN2Generator: Full StyleGAN2 generator with mapping and synthesis

Students should implement the concrete generator classes.
"""

from src.models.generators.base import BaseGenerator

__all__ = ["BaseGenerator"]

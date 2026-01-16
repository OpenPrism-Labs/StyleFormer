"""PyTorch Lightning modules for training.

Lightning modules encapsulate training logic:
- Model architecture
- Loss computation
- Optimizer configuration
- Training/validation steps

Available modules:
- BaseTransformerModule: Abstract base for face transformation
- StyleGANTransformerModule: StyleGAN-based latent editing (Approach B)

Students should implement their training modules here.
"""

from src.models.lightning_modules.base import BaseTransformerModule

__all__ = ["BaseTransformerModule"]

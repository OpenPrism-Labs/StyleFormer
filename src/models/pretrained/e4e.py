"""e4e (encoder4editing) pretrained model loading utilities.

The e4e encoder maps real face images to StyleGAN's W+ latent space,
enabling real image editing through latent manipulation.

e4e improves upon pSp by producing more editable latents that better
respond to semantic editing directions.

Example:
    >>> from src.models.pretrained import load_e4e_encoder
    >>> encoder = load_e4e_encoder("e4e-ffhq-1024")
    >>> latent = encoder(image)  # (B, 18, 512) in W+ space

References:
    - e4e paper: "Designing an Encoder for StyleGAN Image Manipulation"
    - GitHub: https://github.com/omertov/encoder4editing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pretrained.download import download_model, get_model_path


class E4EEncoderWrapper(nn.Module):
    """Wrapper for e4e encoder with consistent interface.

    The e4e encoder maps images to W+ latent space, producing
    per-layer style codes that can be edited for face manipulation.

    Attributes:
        encoder: The underlying encoder model.
        w_dim: Dimension of W latent space.
        num_ws: Number of W vectors (style layers).
        input_size: Expected input image size.

    Example:
        >>> encoder = E4EEncoderWrapper(model, w_dim=512, num_ws=18)
        >>> image = torch.randn(1, 3, 256, 256)  # [-1, 1] normalized
        >>> w_plus = encoder(image)  # (1, 18, 512)

    Note:
        Students should implement the actual encoder architecture.
    """

    def __init__(
        self,
        encoder: nn.Module,
        w_dim: int = 512,
        num_ws: int = 18,
        input_size: int = 256,
    ) -> None:
        """Initialize wrapper.

        Args:
            encoder: The underlying encoder module.
            w_dim: Dimension of W latent space.
            num_ws: Number of style layers in W+ space.
            input_size: Expected input image size.
        """
        super().__init__()
        self.encoder = encoder
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to W+ latent space.

        Args:
            x: Input images (B, 3, H, W) in range [-1, 1].

        Returns:
            W+ latent codes (B, num_ws, w_dim).
        """
        # Resize if needed
        if x.shape[-1] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        return self.encoder(x)

    def encode_to_w(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to W space (averaged across layers).

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            W latent codes (B, w_dim).
        """
        w_plus = self.forward(x)
        return w_plus.mean(dim=1)


def load_e4e_encoder(
    model_name_or_path: str | Path,
    device: str | torch.device = "cuda",
    **kwargs: Any,
) -> E4EEncoderWrapper:
    """Load a pretrained e4e encoder.

    Args:
        model_name_or_path: Either a model name from registry
            (e.g., "e4e-ffhq-1024") or path to checkpoint.
        device: Device to load model on.
        **kwargs: Additional arguments passed to wrapper.

    Returns:
        Wrapped e4e encoder.

    Example:
        >>> encoder = load_e4e_encoder("e4e-ffhq-1024")
        >>> image = load_and_preprocess_image("face.jpg")
        >>> w_plus = encoder(image)
        >>> # Now edit w_plus and decode with StyleGAN2

    Note:
        Students should implement the actual encoder architecture.
    """
    # Determine if it's a path or model name
    path = Path(model_name_or_path)
    if path.exists():
        checkpoint_path = path
    else:
        try:
            checkpoint_path = get_model_path(str(model_name_or_path))
            if not checkpoint_path.exists():
                download_model(str(model_name_or_path))
        except ValueError:
            raise ValueError(
                f"'{model_name_or_path}' is not a valid path or model name."
            )

    # Load checkpoint
    print(f"Loading e4e encoder from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "encoder" in checkpoint:
        state_dict = checkpoint["encoder"]
    elif "state_dict" in checkpoint:
        # Filter encoder keys
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("encoder.")
        }
    else:
        state_dict = checkpoint

    # Infer configuration
    w_dim = kwargs.pop("w_dim", 512)
    num_ws = kwargs.pop("num_ws", 18)
    input_size = kwargs.pop("input_size", 256)

    # Create encoder
    encoder = _create_e4e_encoder(
        state_dict,
        w_dim=w_dim,
        num_ws=num_ws,
    )
    encoder.load_state_dict(state_dict, strict=False)
    encoder = encoder.to(device)
    encoder.eval()

    wrapper = E4EEncoderWrapper(
        encoder,
        w_dim=w_dim,
        num_ws=num_ws,
        input_size=input_size,
        **kwargs,
    )

    return wrapper


def _create_e4e_encoder(
    state_dict: dict[str, torch.Tensor],
    w_dim: int,
    num_ws: int,
) -> nn.Module:
    """Create e4e encoder architecture matching state dict.

    Args:
        state_dict: State dict to match.
        w_dim: W latent dimension.
        num_ws: Number of style layers.

    Returns:
        Encoder module (uninitialized weights).

    TODO (Students):
        Implement the e4e encoder architecture.
        The encoder consists of:
        1. A feature pyramid network (FPN) backbone
        2. map2style layers that convert features to W+ codes
        3. Progressive training for multi-scale encoding

        Key components:
        - Backbone: Usually ResNet-based (e.g., IR-SE50)
        - map2style: Conv layers mapping features to style codes
        - Output: (B, num_ws, w_dim) tensor in W+ space

        Example structure:
        ```python
        class E4EEncoder(nn.Module):
            def __init__(self, w_dim=512, num_ws=18):
                super().__init__()
                self.backbone = IRSEBackbone()  # Feature extractor
                self.styles = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(512, 512, 3, 1, 1),
                        nn.LeakyReLU(),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(512, w_dim),
                    )
                    for _ in range(num_ws)
                ])

            def forward(self, x):
                features = self.backbone(x)
                styles = [style(feat) for style, feat in zip(self.styles, features)]
                return torch.stack(styles, dim=1)
        ```

        Reference: https://github.com/omertov/encoder4editing
    """
    raise NotImplementedError(
        "Students: Implement e4e encoder architecture. "
        "See docstring for guidance.\n"
        "Reference: https://github.com/omertov/encoder4editing"
    )


class PSPEncoderWrapper(E4EEncoderWrapper):
    """Wrapper for pSp (pixel2style2pixel) encoder.

    pSp is similar to e4e but uses a different training approach.
    It directly predicts W+ codes without the iterative refinement
    used in e4e.

    Note:
        The interface is identical to E4EEncoderWrapper.
        Main differences are in training, not architecture.
    """

    pass


def load_psp_encoder(
    model_name_or_path: str | Path,
    device: str | torch.device = "cuda",
    **kwargs: Any,
) -> PSPEncoderWrapper:
    """Load a pretrained pSp encoder.

    Args:
        model_name_or_path: Model name or path.
        device: Device to load on.
        **kwargs: Additional arguments.

    Returns:
        Wrapped pSp encoder.

    Note:
        See load_e4e_encoder for details.
        pSp and e4e have the same interface.
    """
    wrapper = load_e4e_encoder(model_name_or_path, device, **kwargs)
    return PSPEncoderWrapper(
        wrapper.encoder,
        w_dim=wrapper.w_dim,
        num_ws=wrapper.num_ws,
        input_size=wrapper.input_size,
    )

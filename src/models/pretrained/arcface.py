"""ArcFace pretrained model loading utilities.

ArcFace is a face recognition model that produces identity embeddings.
These embeddings are used for:
1. Identity preservation loss during training
2. Identity similarity evaluation

Example:
    >>> from src.models.pretrained import load_arcface
    >>> arcface = load_arcface("arcface-r100")
    >>> embedding = arcface(face_image)  # (B, 512)

References:
    - ArcFace paper: "ArcFace: Additive Angular Margin Loss for Deep
      Face Recognition", Deng et al., CVPR 2019
    - InsightFace: https://github.com/deepinsight/insightface
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pretrained.download import download_model, get_model_path


class ArcFaceWrapper(nn.Module):
    """Wrapper for ArcFace model with consistent interface.

    Provides identity embeddings for face images, useful for
    identity preservation during face transformation.

    Attributes:
        model: The underlying ArcFace model.
        embedding_dim: Dimension of identity embeddings (usually 512).
        input_size: Expected input image size (usually 112).

    Example:
        >>> arcface = ArcFaceWrapper(model, embedding_dim=512)
        >>> image = torch.randn(1, 3, 112, 112)  # [-1, 1] normalized
        >>> embedding = arcface(image)  # (1, 512)
        >>> embedding = F.normalize(embedding)  # Unit normalize
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_dim: int = 512,
        input_size: int = 112,
    ) -> None:
        """Initialize wrapper.

        Args:
            model: The underlying ArcFace model.
            embedding_dim: Dimension of embeddings.
            input_size: Expected input image size.
        """
        super().__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        self.input_size = input_size

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract identity embeddings from face images.

        Args:
            x: Input face images (B, 3, H, W) in range [-1, 1].
            normalize: Whether to L2 normalize embeddings.

        Returns:
            Identity embeddings (B, embedding_dim).
        """
        # Resize if needed
        if x.shape[-1] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        # Get embeddings
        embeddings = self.model(x)

        # Normalize
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def compute_similarity(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute identity similarity between two sets of faces.

        Args:
            x1: First set of face images (B, 3, H, W).
            x2: Second set of face images (B, 3, H, W).

        Returns:
            Cosine similarity scores (B,) in range [-1, 1].
        """
        emb1 = self.forward(x1, normalize=True)
        emb2 = self.forward(x2, normalize=True)
        return F.cosine_similarity(emb1, emb2, dim=1)

    def compute_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute identity distance between two sets of faces.

        Args:
            x1: First set of face images.
            x2: Second set of face images.

        Returns:
            Distance scores (B,) where 0 = same identity.
        """
        similarity = self.compute_similarity(x1, x2)
        return 1.0 - similarity


def load_arcface(
    model_name_or_path: str | Path,
    device: str | torch.device = "cuda",
    **kwargs: Any,
) -> ArcFaceWrapper:
    """Load a pretrained ArcFace model.

    Args:
        model_name_or_path: Either a model name from registry
            (e.g., "arcface-r100") or path to checkpoint.
        device: Device to load model on.
        **kwargs: Additional arguments passed to wrapper.

    Returns:
        Wrapped ArcFace model.

    Example:
        >>> arcface = load_arcface("arcface-r100")
        >>> # Compute identity similarity
        >>> sim = arcface.compute_similarity(face1, face2)
        >>> print(f"Identity similarity: {sim.item():.4f}")

    Note:
        Students should implement the actual model architecture.
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
            raise ValueError(f"'{model_name_or_path}' is not a valid path or model name.")

    # Load checkpoint
    print(f"Loading ArcFace from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Infer model type from name or state dict
    model_type = _infer_arcface_type(model_name_or_path, state_dict)

    # Create model
    model = _create_arcface_model(model_type, state_dict)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    wrapper = ArcFaceWrapper(
        model,
        embedding_dim=kwargs.pop("embedding_dim", 512),
        input_size=kwargs.pop("input_size", 112),
        **kwargs,
    )

    return wrapper


def _infer_arcface_type(
    name_or_path: str | Path,
    state_dict: dict[str, torch.Tensor],
) -> str:
    """Infer ArcFace model type.

    Args:
        name_or_path: Model name or path.
        state_dict: Model state dict.

    Returns:
        Model type string ("r50", "r100", etc.).
    """
    name = str(name_or_path).lower()

    if "r100" in name or "resnet100" in name:
        return "r100"
    elif "r50" in name or "resnet50" in name:
        return "r50"
    elif "r34" in name or "resnet34" in name:
        return "r34"
    elif "r18" in name or "resnet18" in name:
        return "r18"

    # Try to infer from state dict depth
    num_layers = len([k for k in state_dict.keys() if "layer" in k])
    if num_layers > 200:
        return "r100"
    elif num_layers > 100:
        return "r50"
    else:
        return "r50"  # Default


def _create_arcface_model(
    model_type: str,
    state_dict: dict[str, torch.Tensor],
) -> nn.Module:
    """Create ArcFace model architecture.

    Args:
        model_type: Model type ("r50", "r100", etc.).
        state_dict: State dict for weight loading.

    Returns:
        ArcFace model (uninitialized weights).
    """
    # Try to use InsightFace if available
    try:
        if model_type == "r100":
            from insightface.recognition.arcface_torch import iresnet100

            return iresnet100(num_features=512)
        elif model_type == "r50":
            from insightface.recognition.arcface_torch import iresnet50

            return iresnet50(num_features=512)
        elif model_type == "r34":
            from insightface.recognition.arcface_torch import iresnet34

            return iresnet34(num_features=512)
        elif model_type == "r18":
            from insightface.recognition.arcface_torch import iresnet18

            return iresnet18(num_features=512)
    except ImportError:
        pass

    # Minimal IR-SE ResNet implementation
    class BasicBlockIR(nn.Module):
        """Basic IR (Improved Residual) block for ArcFace."""

        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.prelu = nn.PReLU(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_ch)

            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.downsample = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x
            out = self.bn1(x)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.prelu(out)
            out = self.conv2(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            return out + identity

    class MinimalArcFace(nn.Module):
        """Minimal ArcFace (IR-SE ResNet) for identity embedding."""

        def __init__(self, layers: list[int], num_features: int = 512):
            super().__init__()

            # Input layer
            self.input_layer = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU(64),
            )

            # Build layers
            self.layer1 = self._make_layer(64, 64, layers[0], stride=2)
            self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(256, 512, layers[3], stride=2)

            # Output layer
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Dropout(0.4),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, num_features),
                nn.BatchNorm1d(num_features),
            )

        def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int):
            layers = [BasicBlockIR(in_ch, out_ch, stride)]
            for _ in range(1, blocks):
                layers.append(BasicBlockIR(out_ch, out_ch, 1))
            return nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_layer(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.output_layer(x)
            return x

    # Select architecture based on model type
    if model_type == "r100":
        layers = [3, 13, 30, 3]
    elif model_type == "r50":
        layers = [3, 4, 14, 3]
    elif model_type == "r34":
        layers = [3, 4, 6, 3]
    elif model_type == "r18":
        layers = [2, 2, 2, 2]
    else:
        layers = [3, 4, 14, 3]  # Default to R50

    return MinimalArcFace(layers=layers, num_features=512)


class IdentityLoss(nn.Module):
    """Identity preservation loss using ArcFace.

    Computes the cosine distance between source and generated face
    embeddings to encourage identity preservation.

    Example:
        >>> id_loss = IdentityLoss("arcface-r100")
        >>> loss = id_loss(source_images, generated_images)
        >>> loss.backward()
    """

    def __init__(
        self,
        model_name_or_path: str | Path = "arcface-r100",
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize identity loss.

        Args:
            model_name_or_path: ArcFace model to use.
            device: Device to run on.
        """
        super().__init__()
        self.arcface = load_arcface(model_name_or_path, device)

        # Freeze ArcFace
        for param in self.arcface.parameters():
            param.requires_grad = False

    def forward(
        self,
        source: torch.Tensor,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        """Compute identity loss.

        Args:
            source: Source face images (B, 3, H, W).
            generated: Generated face images (B, 3, H, W).

        Returns:
            Identity loss (mean cosine distance).
        """
        return self.arcface.compute_distance(source, generated).mean()

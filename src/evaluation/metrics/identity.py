"""Identity preservation metrics using face recognition.

Identity similarity measures how well the generated image preserves
the identity of the source face. This is crucial for face transformation
tasks where we want to change attributes (age, gender) while keeping
the same person recognizable.

Uses ArcFace or similar face recognition networks to extract identity embeddings.

References:
    - Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
      Face Recognition", CVPR 2019
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class IdentitySimilarity:
    """Calculate identity similarity between source and generated images.

    Uses a pretrained face recognition model (ArcFace) to extract identity
    embeddings and compute cosine similarity.

    Attributes:
        device: Device to run computations on.
        model: Face recognition model for identity extraction.

    Example:
        >>> id_sim = IdentitySimilarity(device="cuda")
        >>> # Single pair
        >>> similarity = id_sim.compute_similarity(source_img, generated_img)
        >>> print(f"Identity similarity: {similarity:.4f}")
        >>>
        >>> # Batch computation
        >>> mean_sim = id_sim.compute_batch_similarity(source_loader, generated_loader)
        >>> print(f"Mean identity similarity: {mean_sim:.4f}")

    Note:
        Students should implement ArcFace model loading in _load_model.
        Higher similarity (closer to 1.0) means better identity preservation.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        model_path: Path | str | None = None,
    ) -> None:
        """Initialize identity similarity calculator.

        Args:
            device: Device to run computations on.
            model_path: Path to pretrained ArcFace weights (optional).
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.model: nn.Module | None = None

    def _load_model(self) -> nn.Module:
        """Load pretrained face recognition model.

        Returns:
            Face recognition model (ArcFace or similar).

        TODO (Students):
            Implement ArcFace or similar face recognition model loading:

            Option 1: Use insightface library
            ```python
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name='buffalo_l')
            app.prepare(ctx_id=0)
            return app
            ```

            Option 2: Use a custom ArcFace implementation
            ```python
            from src.models.pretrained.arcface import ArcFaceModel
            model = ArcFaceModel()
            if self.model_path:
                model.load_state_dict(torch.load(self.model_path))
            model.eval()
            return model.to(self.device)
            ```

            The model should:
            - Accept 112x112 face images (or similar)
            - Output 512-dim identity embeddings
            - Be normalized to unit length
        """
        raise NotImplementedError(
            "Students: Implement face recognition model loading. "
            "See docstring for guidance."
        )

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for face recognition model.

        Args:
            images: Images tensor of shape (B, C, H, W) in range [-1, 1] or [0, 1].

        Returns:
            Preprocessed images ready for the model.

        Note:
            ArcFace typically expects:
            - Size: 112x112
            - Range: [0, 1] or [-1, 1] depending on implementation
            - RGB order
        """
        # Resize to model input size
        if images.shape[-1] != 112:
            images = F.interpolate(
                images,
                size=(112, 112),
                mode="bilinear",
                align_corners=False,
            )

        return images

    def extract_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Extract identity embeddings from images.

        Args:
            images: Batch of face images (B, C, H, W).

        Returns:
            Identity embeddings of shape (B, 512).
        """
        if self.model is None:
            self.model = self._load_model()

        images = self._preprocess(images.to(self.device))

        with torch.no_grad():
            embeddings = self.model(images)

        # Normalize to unit length
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def compute_similarity(
        self,
        source: torch.Tensor,
        generated: torch.Tensor,
    ) -> float:
        """Compute identity similarity between source and generated image.

        Args:
            source: Source face image (1, C, H, W) or (C, H, W).
            generated: Generated face image (1, C, H, W) or (C, H, W).

        Returns:
            Cosine similarity in range [-1, 1]. Higher is better.
        """
        if source.dim() == 3:
            source = source.unsqueeze(0)
        if generated.dim() == 3:
            generated = generated.unsqueeze(0)

        emb_source = self.extract_embedding(source)
        emb_generated = self.extract_embedding(generated)

        similarity = F.cosine_similarity(emb_source, emb_generated, dim=1)

        return float(similarity.item())

    def compute_batch_similarity(
        self,
        source_images: torch.Tensor,
        generated_images: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute mean identity similarity for a batch.

        Args:
            source_images: Batch of source images (B, C, H, W).
            generated_images: Batch of generated images (B, C, H, W).

        Returns:
            Tuple of (mean_similarity, std_similarity).
        """
        emb_source = self.extract_embedding(source_images)
        emb_generated = self.extract_embedding(generated_images)

        similarities = F.cosine_similarity(emb_source, emb_generated, dim=1)

        return float(similarities.mean().item()), float(similarities.std().item())

    def compute_from_dataloaders(
        self,
        source_loader: DataLoader,
        generated_loader: DataLoader,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """Compute identity metrics from paired dataloaders.

        Args:
            source_loader: DataLoader for source images.
            generated_loader: DataLoader for generated images.
            max_samples: Maximum number of samples to evaluate.

        Returns:
            Dictionary with mean, std, min, max similarity values.
        """
        all_similarities: list[float] = []
        n_samples = 0

        for source_batch, gen_batch in tqdm(
            zip(source_loader, generated_loader),
            desc="Computing identity similarity",
        ):
            # Handle different batch formats
            if isinstance(source_batch, dict):
                source_images = source_batch["image"]
            elif isinstance(source_batch, (list, tuple)):
                source_images = source_batch[0]
            else:
                source_images = source_batch

            if isinstance(gen_batch, dict):
                gen_images = gen_batch["image"]
            elif isinstance(gen_batch, (list, tuple)):
                gen_images = gen_batch[0]
            else:
                gen_images = gen_batch

            emb_source = self.extract_embedding(source_images)
            emb_gen = self.extract_embedding(gen_images)

            sims = F.cosine_similarity(emb_source, emb_gen, dim=1)
            all_similarities.extend(sims.cpu().tolist())

            n_samples += source_images.shape[0]
            if max_samples is not None and n_samples >= max_samples:
                break

        similarities = np.array(all_similarities[:max_samples] if max_samples else all_similarities)

        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "median": float(np.median(similarities)),
        }


class IdentityLoss(nn.Module):
    """Identity preservation loss for training.

    This loss encourages the model to preserve identity during transformation.
    It computes 1 - cosine_similarity between source and generated embeddings.

    Example:
        >>> id_loss = IdentityLoss(device="cuda")
        >>> loss = id_loss(source_images, generated_images)
        >>> loss.backward()
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        model_path: Path | str | None = None,
    ) -> None:
        """Initialize identity loss.

        Args:
            device: Device to run computations on.
            model_path: Path to pretrained face recognition weights.
        """
        super().__init__()
        self.similarity_calculator = IdentitySimilarity(
            device=device,
            model_path=model_path,
        )

    def forward(
        self,
        source: torch.Tensor,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        """Compute identity loss.

        Args:
            source: Source images (B, C, H, W).
            generated: Generated images (B, C, H, W).

        Returns:
            Identity loss (1 - cosine_similarity), averaged over batch.
        """
        emb_source = self.similarity_calculator.extract_embedding(source)
        emb_generated = self.similarity_calculator.extract_embedding(generated)

        similarity = F.cosine_similarity(emb_source, emb_generated, dim=1)
        loss = 1.0 - similarity

        return loss.mean()

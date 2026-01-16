"""Fréchet Inception Distance (FID) calculation.

FID measures the quality and diversity of generated images by comparing
the statistics of generated images to real images in InceptionV3 feature space.

Lower FID = better quality and diversity.

References:
    - Heusel et al., "GANs Trained by a Two Time-Scale Update Rule
      Converge to a Local Nash Equilibrium", NeurIPS 2017
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path


class FIDCalculator:
    """Calculate Fréchet Inception Distance between two image distributions.

    This class uses InceptionV3 features to compute FID score, which measures
    how similar the generated images are to real images in terms of both
    quality (realism) and diversity.

    Attributes:
        device: Device to run computations on.
        dims: Feature dimensionality (default: 2048 for InceptionV3 pool3).
        batch_size: Batch size for feature extraction.

    Example:
        >>> fid_calc = FIDCalculator(device="cuda")
        >>> # From dataloaders
        >>> fid_score = fid_calc.calculate_from_dataloaders(real_loader, fake_loader)
        >>> print(f"FID: {fid_score:.2f}")
        >>>
        >>> # From precomputed statistics
        >>> fid_score = fid_calc.calculate_from_statistics(mu1, sigma1, mu2, sigma2)

    Note:
        Students should implement the actual InceptionV3 model loading
        in the _load_inception method.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        dims: int = 2048,
        batch_size: int = 64,
    ) -> None:
        """Initialize FID calculator.

        Args:
            device: Device to run computations on.
            dims: Feature dimensionality from InceptionV3.
            batch_size: Batch size for feature extraction.
        """
        self.device = torch.device(device)
        self.dims = dims
        self.batch_size = batch_size
        self.inception: nn.Module | None = None

    def _load_inception(self) -> nn.Module:
        """Load InceptionV3 model for feature extraction.

        Returns:
            InceptionV3 model truncated at pool3 layer.

        TODO (Students):
            Implement InceptionV3 loading:
            1. Load pretrained InceptionV3 from torchvision
            2. Remove the final classification layer
            3. Return features from the pool3 layer (2048-dim)

            Example implementation:
            ```python
            from torchvision.models import inception_v3, Inception_V3_Weights

            inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            inception.fc = nn.Identity()  # Remove classifier
            inception.eval()
            return inception.to(self.device)
            ```
        """
        raise NotImplementedError(
            "Students: Implement InceptionV3 loading for FID calculation. "
            "See docstring for guidance."
        )

    def _extract_features(
        self,
        dataloader: DataLoader,
        max_samples: int | None = None,
    ) -> np.ndarray:
        """Extract InceptionV3 features from images.

        Args:
            dataloader: DataLoader yielding image batches.
            max_samples: Maximum number of samples to process.

        Returns:
            Feature array of shape (N, dims).
        """
        if self.inception is None:
            self.inception = self._load_inception()

        features_list: list[np.ndarray] = []
        n_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["image"]
                elif isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)

                # Resize to InceptionV3 input size (299x299)
                if images.shape[-1] != 299:
                    images = torch.nn.functional.interpolate(
                        images,
                        size=(299, 299),
                        mode="bilinear",
                        align_corners=False,
                    )

                # Extract features
                feats = self.inception(images)
                features_list.append(feats.cpu().numpy())

                n_samples += images.shape[0]
                if max_samples is not None and n_samples >= max_samples:
                    break

        features = np.concatenate(features_list, axis=0)
        if max_samples is not None:
            features = features[:max_samples]

        return features

    def _compute_statistics(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of features.

        Args:
            features: Feature array of shape (N, dims).

        Returns:
            Tuple of (mean, covariance) arrays.
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Calculate FID given statistics of two distributions.

        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

        Args:
            mu1: Mean of first distribution.
            sigma1: Covariance of first distribution.
            mu2: Mean of second distribution.
            sigma2: Covariance of second distribution.
            eps: Small constant for numerical stability.

        Returns:
            FID score (lower is better).
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Numerical stability
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical stability (imaginary component)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

        return float(fid)

    def calculate_from_dataloaders(
        self,
        real_loader: DataLoader,
        fake_loader: DataLoader,
        max_samples: int | None = 50000,
    ) -> float:
        """Calculate FID between two dataloaders.

        Args:
            real_loader: DataLoader for real images.
            fake_loader: DataLoader for generated images.
            max_samples: Maximum samples to use from each distribution.

        Returns:
            FID score.
        """
        # Extract features
        real_features = self._extract_features(real_loader, max_samples)
        fake_features = self._extract_features(fake_loader, max_samples)

        # Compute statistics
        mu1, sigma1 = self._compute_statistics(real_features)
        mu2, sigma2 = self._compute_statistics(fake_features)

        # Calculate FID
        return self.calculate_fid(mu1, sigma1, mu2, sigma2)

    def calculate_from_statistics(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        """Calculate FID from precomputed statistics.

        Useful when you've cached statistics for a reference dataset.

        Args:
            mu1: Mean of first distribution.
            sigma1: Covariance of first distribution.
            mu2: Mean of second distribution.
            sigma2: Covariance of second distribution.

        Returns:
            FID score.
        """
        return self.calculate_fid(mu1, sigma1, mu2, sigma2)

    def save_statistics(
        self,
        dataloader: DataLoader,
        path: Path,
        max_samples: int | None = 50000,
    ) -> None:
        """Compute and save statistics for a dataset.

        Args:
            dataloader: DataLoader for images.
            path: Path to save statistics (.npz file).
            max_samples: Maximum samples to use.
        """
        features = self._extract_features(dataloader, max_samples)
        mu, sigma = self._compute_statistics(features)
        np.savez(path, mu=mu, sigma=sigma)

    @staticmethod
    def load_statistics(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load precomputed statistics.

        Args:
            path: Path to .npz file with statistics.

        Returns:
            Tuple of (mu, sigma).
        """
        data = np.load(path)
        return data["mu"], data["sigma"]

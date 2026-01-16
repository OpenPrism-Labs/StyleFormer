"""Unified evaluator for face transformation models.

Provides a single interface to compute all evaluation metrics:
- FID (quality and diversity)
- Identity similarity (identity preservation)
- Attribute accuracy (transformation effectiveness)

Example:
    >>> evaluator = Evaluator(device="cuda")
    >>> results = evaluator.evaluate_full(
    ...     source_loader=source_loader,
    ...     generated_loader=generated_loader,
    ...     real_loader=real_loader,
    ...     target_attributes={"age": "old", "gender": "female"},
    ... )
    >>> print(evaluator.format_results(results))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics.fid import FIDCalculator
from src.evaluation.metrics.identity import IdentitySimilarity
from src.evaluation.metrics.attribute import AttributeAccuracy


@dataclass
class EvaluationConfig:
    """Configuration for evaluation.

    Attributes:
        compute_fid: Whether to compute FID score.
        compute_identity: Whether to compute identity similarity.
        compute_attribute: Whether to compute attribute accuracy.
        fid_max_samples: Max samples for FID calculation.
        identity_max_samples: Max samples for identity similarity.
        attribute_max_samples: Max samples for attribute accuracy.
        fid_dims: Feature dimensionality for FID.
        device: Device to run evaluation on.
    """

    compute_fid: bool = True
    compute_identity: bool = True
    compute_attribute: bool = True
    fid_max_samples: int = 50000
    identity_max_samples: int = 10000
    attribute_max_samples: int = 10000
    fid_dims: int = 2048
    device: str = "cuda"


@dataclass
class EvaluationResults:
    """Container for evaluation results.

    Attributes:
        fid: FID score (lower is better).
        identity_mean: Mean identity similarity (higher is better).
        identity_std: Std of identity similarity.
        attribute_accuracy: Dict of attribute accuracies.
        metadata: Additional metadata about the evaluation.
    """

    fid: float | None = None
    identity_mean: float | None = None
    identity_std: float | None = None
    identity_min: float | None = None
    identity_max: float | None = None
    attribute_accuracy: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "fid": self.fid,
            "identity": {
                "mean": self.identity_mean,
                "std": self.identity_std,
                "min": self.identity_min,
                "max": self.identity_max,
            },
            "attribute_accuracy": self.attribute_accuracy,
            "metadata": self.metadata,
        }


class Evaluator:
    """Unified evaluator for face transformation models.

    Combines FID, identity similarity, and attribute accuracy metrics
    into a single evaluation interface.

    Attributes:
        config: Evaluation configuration.
        fid_calculator: FID metric calculator.
        identity_calculator: Identity similarity calculator.
        attribute_calculator: Attribute accuracy calculator.

    Example:
        >>> evaluator = Evaluator(device="cuda")
        >>>
        >>> # Full evaluation
        >>> results = evaluator.evaluate_full(
        ...     source_loader=source_loader,
        ...     generated_loader=generated_loader,
        ...     real_loader=real_loader,
        ...     target_attributes={"age": "old"},
        ... )
        >>>
        >>> # Print formatted results
        >>> print(evaluator.format_results(results))

        >>> # Individual metrics
        >>> fid = evaluator.compute_fid(real_loader, generated_loader)
        >>> id_sim = evaluator.compute_identity(source_loader, generated_loader)
    """

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize evaluator.

        Args:
            config: Evaluation configuration. If None, uses defaults.
            device: Device to run evaluation on.
        """
        self.config = config or EvaluationConfig(device=str(device))
        self.device = torch.device(device)

        # Initialize metric calculators
        self.fid_calculator = FIDCalculator(
            device=self.device,
            dims=self.config.fid_dims,
        )
        self.identity_calculator = IdentitySimilarity(device=self.device)
        self.attribute_calculator = AttributeAccuracy(device=self.device)

    def load_attribute_classifiers(
        self,
        classifiers: dict[str, Path | str],
    ) -> None:
        """Load attribute classifiers for evaluation.

        Args:
            classifiers: Dict mapping attribute names to model paths.
        """
        for attr, path in classifiers.items():
            self.attribute_calculator.load_classifier(attr, path)

    def compute_fid(
        self,
        real_loader: DataLoader,
        generated_loader: DataLoader,
        max_samples: int | None = None,
    ) -> float:
        """Compute FID score.

        Args:
            real_loader: DataLoader for real images.
            generated_loader: DataLoader for generated images.
            max_samples: Maximum samples to use.

        Returns:
            FID score (lower is better).
        """
        max_samples = max_samples or self.config.fid_max_samples
        return self.fid_calculator.calculate_from_dataloaders(
            real_loader,
            generated_loader,
            max_samples=max_samples,
        )

    def compute_identity(
        self,
        source_loader: DataLoader,
        generated_loader: DataLoader,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """Compute identity similarity.

        Args:
            source_loader: DataLoader for source images.
            generated_loader: DataLoader for generated images.
            max_samples: Maximum samples to use.

        Returns:
            Dict with mean, std, min, max, median similarity.
        """
        max_samples = max_samples or self.config.identity_max_samples
        return self.identity_calculator.compute_from_dataloaders(
            source_loader,
            generated_loader,
            max_samples=max_samples,
        )

    def compute_attribute_accuracy(
        self,
        generated_loader: DataLoader,
        target_attributes: dict[str, str | int],
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """Compute attribute accuracy.

        Args:
            generated_loader: DataLoader for generated images.
            target_attributes: Target attributes and their values.
            max_samples: Maximum samples to use.

        Returns:
            Dict mapping attribute names to accuracy values.
        """
        max_samples = max_samples or self.config.attribute_max_samples
        return self.attribute_calculator.compute_accuracy(
            generated_loader,
            target_attributes,
            max_samples=max_samples,
        )

    def evaluate_full(
        self,
        source_loader: DataLoader,
        generated_loader: DataLoader,
        real_loader: DataLoader | None = None,
        target_attributes: dict[str, str | int] | None = None,
    ) -> EvaluationResults:
        """Run full evaluation pipeline.

        Args:
            source_loader: DataLoader for source images (for identity).
            generated_loader: DataLoader for generated images.
            real_loader: DataLoader for real target domain images (for FID).
                If None, FID is skipped.
            target_attributes: Target attributes for accuracy computation.
                If None or classifiers not loaded, attribute accuracy is skipped.

        Returns:
            EvaluationResults with all computed metrics.
        """
        results = EvaluationResults()
        results.metadata["device"] = str(self.device)

        # FID
        if self.config.compute_fid and real_loader is not None:
            try:
                results.fid = self.compute_fid(real_loader, generated_loader)
                results.metadata["fid_samples"] = self.config.fid_max_samples
            except NotImplementedError:
                results.metadata["fid_error"] = "InceptionV3 not implemented"

        # Identity similarity
        if self.config.compute_identity:
            try:
                id_results = self.compute_identity(source_loader, generated_loader)
                results.identity_mean = id_results["mean"]
                results.identity_std = id_results["std"]
                results.identity_min = id_results["min"]
                results.identity_max = id_results["max"]
                results.metadata["identity_samples"] = self.config.identity_max_samples
            except NotImplementedError:
                results.metadata["identity_error"] = "Face recognition model not implemented"

        # Attribute accuracy
        if (
            self.config.compute_attribute
            and target_attributes is not None
            and len(self.attribute_calculator.classifiers) > 0
        ):
            try:
                results.attribute_accuracy = self.compute_attribute_accuracy(
                    generated_loader,
                    target_attributes,
                )
                results.metadata["attribute_samples"] = self.config.attribute_max_samples
            except NotImplementedError:
                results.metadata["attribute_error"] = "Attribute classifiers not implemented"

        return results

    def format_results(self, results: EvaluationResults) -> str:
        """Format results as a human-readable string.

        Args:
            results: Evaluation results to format.

        Returns:
            Formatted string.
        """
        lines = ["=" * 50, "Evaluation Results", "=" * 50, ""]

        # FID
        if results.fid is not None:
            lines.append(f"FID Score: {results.fid:.2f}")
        elif "fid_error" in results.metadata:
            lines.append(f"FID Score: {results.metadata['fid_error']}")
        lines.append("")

        # Identity
        if results.identity_mean is not None:
            lines.append("Identity Similarity:")
            lines.append(f"  Mean: {results.identity_mean:.4f}")
            lines.append(f"  Std:  {results.identity_std:.4f}")
            lines.append(f"  Min:  {results.identity_min:.4f}")
            lines.append(f"  Max:  {results.identity_max:.4f}")
        elif "identity_error" in results.metadata:
            lines.append(f"Identity: {results.metadata['identity_error']}")
        lines.append("")

        # Attribute accuracy
        if results.attribute_accuracy:
            lines.append("Attribute Accuracy:")
            for attr, acc in results.attribute_accuracy.items():
                lines.append(f"  {attr}: {acc:.2%}")
        elif "attribute_error" in results.metadata:
            lines.append(f"Attributes: {results.metadata['attribute_error']}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)

    def save_results(
        self,
        results: EvaluationResults,
        path: Path | str,
    ) -> None:
        """Save results to JSON file.

        Args:
            results: Evaluation results to save.
            path: Path to save JSON file.
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

    @staticmethod
    def load_results(path: Path | str) -> EvaluationResults:
        """Load results from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Loaded evaluation results.
        """
        import json

        with open(path) as f:
            data = json.load(f)

        return EvaluationResults(
            fid=data.get("fid"),
            identity_mean=data.get("identity", {}).get("mean"),
            identity_std=data.get("identity", {}).get("std"),
            identity_min=data.get("identity", {}).get("min"),
            identity_max=data.get("identity", {}).get("max"),
            attribute_accuracy=data.get("attribute_accuracy", {}),
            metadata=data.get("metadata", {}),
        )

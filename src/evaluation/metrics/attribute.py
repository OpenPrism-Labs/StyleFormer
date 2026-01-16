"""Attribute classification accuracy metrics.

Measures how well the transformation achieves the target attributes.
For example, if we transform a young face to old, does an age classifier
correctly predict the generated face as old?

This is crucial for evaluating the effectiveness of attribute manipulation.
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


class AttributeAccuracy:
    """Calculate attribute classification accuracy on generated images.

    Uses pretrained attribute classifiers to verify that generated images
    have the desired target attributes.

    Attributes:
        device: Device to run computations on.
        classifiers: Dict mapping attribute names to classifier models.

    Example:
        >>> attr_acc = AttributeAccuracy(device="cuda")
        >>> attr_acc.load_classifier("age", age_classifier_path)
        >>> attr_acc.load_classifier("gender", gender_classifier_path)
        >>>
        >>> # Evaluate generated images
        >>> accuracy = attr_acc.compute_accuracy(
        ...     generated_loader,
        ...     target_attributes={"age": "old", "gender": "female"}
        ... )
        >>> print(f"Age accuracy: {accuracy['age']:.2%}")
        >>> print(f"Gender accuracy: {accuracy['gender']:.2%}")

    Note:
        Students should implement or load pretrained attribute classifiers.
        Each classifier should output probabilities for attribute classes.
    """

    # Default attribute configurations
    ATTRIBUTE_CLASSES: dict[str, list[str]] = {
        "age": ["young", "old"],
        "gender": ["male", "female"],
        "smile": ["no_smile", "smile"],
        "glasses": ["no_glasses", "glasses"],
        "beard": ["no_beard", "beard"],
    }

    def __init__(
        self,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize attribute accuracy calculator.

        Args:
            device: Device to run computations on.
        """
        self.device = torch.device(device)
        self.classifiers: dict[str, nn.Module] = {}

    def load_classifier(
        self,
        attribute: str,
        model_or_path: nn.Module | Path | str,
        num_classes: int | None = None,
    ) -> None:
        """Load an attribute classifier.

        Args:
            attribute: Name of the attribute (e.g., "age", "gender").
            model_or_path: Either a nn.Module or path to saved weights.
            num_classes: Number of classes (inferred from ATTRIBUTE_CLASSES if not provided).

        TODO (Students):
            Implement classifier loading. Options:
            1. Train your own classifiers on CelebA attributes
            2. Use pretrained models from timm/torchvision
            3. Fine-tune a ResNet on CelebA attribute labels

            Example implementation:
            ```python
            import timm

            if isinstance(model_or_path, nn.Module):
                classifier = model_or_path
            else:
                n_classes = num_classes or len(self.ATTRIBUTE_CLASSES.get(attribute, []))
                classifier = timm.create_model(
                    'resnet18',
                    pretrained=False,
                    num_classes=n_classes
                )
                classifier.load_state_dict(torch.load(model_or_path))

            classifier.eval()
            self.classifiers[attribute] = classifier.to(self.device)
            ```
        """
        raise NotImplementedError(
            "Students: Implement attribute classifier loading. "
            "See docstring for guidance."
        )

    def _create_default_classifier(self, attribute: str) -> nn.Module:
        """Create a default classifier architecture.

        Args:
            attribute: Attribute name to determine number of classes.

        Returns:
            Untrained classifier model.

        TODO (Students):
            Implement a simple classifier architecture:
            ```python
            import timm

            num_classes = len(self.ATTRIBUTE_CLASSES.get(attribute, ["class_0", "class_1"]))
            model = timm.create_model(
                'resnet18',
                pretrained=True,
                num_classes=num_classes
            )
            return model
            ```
        """
        raise NotImplementedError(
            "Students: Implement default classifier creation."
        )

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for attribute classifiers.

        Args:
            images: Images tensor (B, C, H, W).

        Returns:
            Preprocessed images.
        """
        # Most classifiers expect 224x224
        if images.shape[-1] != 224:
            images = F.interpolate(
                images,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        return images

    def predict(
        self,
        images: torch.Tensor,
        attribute: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict attribute class for images.

        Args:
            images: Batch of images (B, C, H, W).
            attribute: Attribute to predict.

        Returns:
            Tuple of (predicted_classes, probabilities).
        """
        if attribute not in self.classifiers:
            raise ValueError(
                f"Classifier for '{attribute}' not loaded. "
                f"Call load_classifier('{attribute}', path) first."
            )

        classifier = self.classifiers[attribute]
        images = self._preprocess(images.to(self.device))

        with torch.no_grad():
            logits = classifier(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        return preds, probs

    def compute_accuracy(
        self,
        dataloader: DataLoader,
        target_attributes: dict[str, str | int],
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """Compute attribute accuracy on generated images.

        Args:
            dataloader: DataLoader for generated images.
            target_attributes: Dict mapping attribute names to target class
                (either class name or index).
            max_samples: Maximum samples to evaluate.

        Returns:
            Dict mapping attribute names to accuracy values.

        Example:
            >>> accuracy = attr_acc.compute_accuracy(
            ...     gen_loader,
            ...     target_attributes={"age": "old", "gender": "female"}
            ... )
        """
        # Convert string targets to indices
        target_indices: dict[str, int] = {}
        for attr, target in target_attributes.items():
            if isinstance(target, str):
                classes = self.ATTRIBUTE_CLASSES.get(attr, [])
                if target not in classes:
                    raise ValueError(f"Unknown class '{target}' for attribute '{attr}'")
                target_indices[attr] = classes.index(target)
            else:
                target_indices[attr] = target

        # Count correct predictions
        correct: dict[str, int] = {attr: 0 for attr in target_attributes}
        total = 0

        for batch in tqdm(dataloader, desc="Computing attribute accuracy"):
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch["image"]
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            batch_size = images.shape[0]

            for attr in target_attributes:
                preds, _ = self.predict(images, attr)
                target_idx = target_indices[attr]
                correct[attr] += (preds == target_idx).sum().item()

            total += batch_size

            if max_samples is not None and total >= max_samples:
                break

        # Compute accuracies
        accuracies = {
            attr: correct[attr] / min(total, max_samples or total)
            for attr in target_attributes
        }

        return accuracies

    def compute_per_sample_accuracy(
        self,
        images: torch.Tensor,
        target_attributes: dict[str, str | int],
    ) -> dict[str, torch.Tensor]:
        """Compute per-sample accuracy for a batch.

        Args:
            images: Batch of images (B, C, H, W).
            target_attributes: Target attributes and their values.

        Returns:
            Dict mapping attribute names to boolean tensors (B,).
        """
        results: dict[str, torch.Tensor] = {}

        for attr, target in target_attributes.items():
            if isinstance(target, str):
                classes = self.ATTRIBUTE_CLASSES.get(attr, [])
                target_idx = classes.index(target)
            else:
                target_idx = target

            preds, _ = self.predict(images, attr)
            results[attr] = preds == target_idx

        return results


class AttributeClassificationLoss(nn.Module):
    """Attribute classification loss for training.

    Encourages generated images to have target attributes.
    Uses cross-entropy loss against target attribute classes.

    Example:
        >>> attr_loss = AttributeClassificationLoss(device="cuda")
        >>> attr_loss.load_classifiers({"age": age_model, "gender": gender_model})
        >>> loss = attr_loss(generated_images, {"age": 1, "gender": 0})  # old male
        >>> loss.backward()
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize attribute classification loss.

        Args:
            device: Device to run computations on.
        """
        super().__init__()
        self.accuracy_calculator = AttributeAccuracy(device=device)
        self.ce_loss = nn.CrossEntropyLoss()

    def load_classifiers(
        self,
        classifiers: dict[str, nn.Module | Path | str],
    ) -> None:
        """Load multiple attribute classifiers.

        Args:
            classifiers: Dict mapping attribute names to models or paths.
        """
        for attr, model_or_path in classifiers.items():
            self.accuracy_calculator.load_classifier(attr, model_or_path)

    def forward(
        self,
        generated: torch.Tensor,
        target_attributes: dict[str, int],
    ) -> torch.Tensor:
        """Compute attribute classification loss.

        Args:
            generated: Generated images (B, C, H, W).
            target_attributes: Dict mapping attribute names to target class indices.

        Returns:
            Total attribute classification loss.
        """
        total_loss = torch.tensor(0.0, device=generated.device)

        for attr, target_idx in target_attributes.items():
            if attr not in self.accuracy_calculator.classifiers:
                continue

            classifier = self.accuracy_calculator.classifiers[attr]
            images = self.accuracy_calculator._preprocess(generated)
            logits = classifier(images)

            # Create target tensor
            targets = torch.full(
                (generated.shape[0],),
                target_idx,
                dtype=torch.long,
                device=generated.device,
            )

            total_loss = total_loss + self.ce_loss(logits, targets)

        return total_loss

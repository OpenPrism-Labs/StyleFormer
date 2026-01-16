"""Image transformation pipelines for face datasets."""

from typing import Any

from torchvision import transforms
from torchvision.transforms import functional as TF


# ImageNet normalization (commonly used for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Face dataset normalization ([-1, 1] range, common for GANs)
FACE_MEAN = [0.5, 0.5, 0.5]
FACE_STD = [0.5, 0.5, 0.5]


def get_train_transforms(
    image_size: int = 256,
    horizontal_flip: bool = True,
    color_jitter: bool = True,
    random_crop: bool = False,
    normalize_mode: str = "face",
) -> transforms.Compose:
    """Get training transforms with augmentation.
    
    Args:
        image_size: Target image size.
        horizontal_flip: Apply random horizontal flip.
        color_jitter: Apply color jitter augmentation.
        random_crop: Apply random crop (vs center crop).
        normalize_mode: "face" for [-1,1] or "imagenet" for ImageNet stats.
        
    Returns:
        Composed transforms.
    """
    transform_list = []
    
    # Resize
    if random_crop:
        transform_list.append(transforms.Resize(int(image_size * 1.1)))
        transform_list.append(transforms.RandomCrop(image_size))
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Horizontal flip
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Color jitter
    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )
        )
    
    # To tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if normalize_mode == "face":
        transform_list.append(transforms.Normalize(FACE_MEAN, FACE_STD))
    elif normalize_mode == "imagenet":
        transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    
    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: int = 256,
    normalize_mode: str = "face",
) -> transforms.Compose:
    """Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size.
        normalize_mode: "face" for [-1,1] or "imagenet" for ImageNet stats.
        
    Returns:
        Composed transforms.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    if normalize_mode == "face":
        transform_list.append(transforms.Normalize(FACE_MEAN, FACE_STD))
    elif normalize_mode == "imagenet":
        transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    
    return transforms.Compose(transform_list)


def get_inference_transforms(
    image_size: int = 256,
    normalize_mode: str = "face",
) -> transforms.Compose:
    """Get inference transforms (alias for val transforms)."""
    return get_val_transforms(image_size, normalize_mode)


def denormalize(
    tensor: Any,
    normalize_mode: str = "face",
) -> Any:
    """Denormalize tensor back to [0, 1] range.
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W].
        normalize_mode: Normalization mode used.
        
    Returns:
        Denormalized tensor in [0, 1] range.
    """
    if normalize_mode == "face":
        mean = FACE_MEAN
        std = FACE_STD
    else:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    
    # Handle batch dimension
    if tensor.dim() == 4:
        mean = tensor.new_tensor(mean).view(1, 3, 1, 1)
        std = tensor.new_tensor(std).view(1, 3, 1, 1)
    else:
        mean = tensor.new_tensor(mean).view(3, 1, 1)
        std = tensor.new_tensor(std).view(3, 1, 1)
    
    return tensor * std + mean


class PairedTransform:
    """Apply same random transform to a pair of images.
    
    Useful for style transfer where source/target should have
    same augmentation (e.g., same crop, same flip).
    """
    
    def __init__(
        self,
        image_size: int = 256,
        horizontal_flip: bool = True,
        normalize_mode: str = "face",
    ) -> None:
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.normalize_mode = normalize_mode
        
        if normalize_mode == "face":
            self.mean = FACE_MEAN
            self.std = FACE_STD
        else:
            self.mean = IMAGENET_MEAN
            self.std = IMAGENET_STD
    
    def __call__(
        self,
        img1: Any,
        img2: Any,
    ) -> tuple[Any, Any]:
        """Apply same transform to both images.
        
        Args:
            img1: First PIL image.
            img2: Second PIL image.
            
        Returns:
            Tuple of transformed tensors.
        """
        # Resize
        img1 = TF.resize(img1, (self.image_size, self.image_size))
        img2 = TF.resize(img2, (self.image_size, self.image_size))
        
        # Random horizontal flip (same for both)
        if self.horizontal_flip and transforms.RandomHorizontalFlip.get_params(0.5):
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
        
        # To tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        
        # Normalize
        img1 = TF.normalize(img1, self.mean, self.std)
        img2 = TF.normalize(img2, self.mean, self.std)
        
        return img1, img2

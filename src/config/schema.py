"""Pydantic schemas for configuration validation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str = "baseline"
    seed: int = 42
    tags: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Training configuration."""
    max_epochs: int = 100
    val_check_interval: float = 1.0
    log_every_n_steps: int = 50
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = True


@dataclass
class AttributeDefinition:
    """Single attribute definition."""
    type: str = "continuous"  # continuous | discrete
    range: tuple[float, float] = (-3.0, 3.0)
    labels: tuple[str, str] = ("negative", "positive")
    celeba_attr: str = ""
    default_strength: float = 2.0


@dataclass
class TransformConfig:
    """Attribute transformation configuration."""
    mode: str = "simultaneous"  # simultaneous | sequential
    null_space_projection: bool = True
    orthogonalize: bool = True
    targets: dict[str, float] = field(default_factory=dict)


@dataclass
class AttributesConfig:
    """Multi-attribute configuration."""
    active: list[str] = field(default_factory=lambda: ["age", "gender"])
    definitions: dict[str, AttributeDefinition] = field(default_factory=dict)
    transform: TransformConfig = field(default_factory=TransformConfig)


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""
    horizontal_flip: bool = True
    color_jitter: bool = True
    random_crop: bool = False


@dataclass
class PairingConfig:
    """Dataset pairing configuration."""
    enabled: bool = False
    mode: str = "opposite"  # opposite | random | matched
    transfer_attr: str = "Male"
    preserve_attrs: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Dataset configuration."""
    name: str = "celeba_hq"
    root: str = "./data/celeba_hq"
    image_size: int = 256
    channels: int = 3
    splits: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    attributes: dict[str, Any] = field(default_factory=dict)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    augmentation: dict[str, AugmentationConfig] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    project: str = "styleformer"
    entity: str | None = None
    offline: bool = False
    log_images_every_n_steps: int = 500


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    accelerator: str = "auto"
    devices: int = 1
    strategy: str = "auto"


@dataclass
class PathsConfig:
    """Paths configuration."""
    root: str = "."
    data_dir: str = "./data"
    pretrained_dir: str = "./pretrained"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"


@dataclass 
class StyleFormerConfig:
    """Root configuration for StyleFormer experiments."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    attributes: AttributesConfig = field(default_factory=AttributesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

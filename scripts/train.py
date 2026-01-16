"""Training entry point using Hydra configuration."""

import logging
from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from src.data.datamodule import FaceDataModule
from src.utils.logging import setup_logging, setup_wandb


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration.
    """
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Print config
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    L.seed_everything(cfg.experiment.seed, workers=True)
    
    # Initialize W&B if configured
    if cfg.logging.get("project"):
        setup_wandb(
            project=cfg.logging.project,
            entity=cfg.logging.get("entity"),
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.experiment.name,
            tags=cfg.experiment.get("tags", []),
            offline=cfg.logging.get("offline", False),
        )
    
    # Create data module
    log.info(f"Loading dataset: {cfg.data.name}")
    datamodule = FaceDataModule.from_config(cfg)
    
    # TODO: Students implement model creation here
    # Example:
    # model = hydra.utils.instantiate(cfg.model)
    
    log.info("=" * 60)
    log.info("DATA MODULE READY")
    log.info(f"Dataset: {cfg.data.name}")
    log.info(f"Image size: {cfg.data.image_size}")
    log.info(f"Batch size: {cfg.dataloader.batch_size}")
    log.info(f"Selected attributes: {cfg.data.attributes.selected}")
    log.info("=" * 60)
    log.info("")
    log.info("TODO: Implement your model in src/models/")
    log.info("See src/models/README.md for instructions.")
    log.info("")
    
    # Verify datamodule works
    datamodule.setup("fit")
    log.info(f"Train samples: {len(datamodule.train_dataset)}")
    log.info(f"Val samples: {len(datamodule.val_dataset)}")
    
    # Get a sample batch
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    if "source_image" in batch:
        log.info(f"Paired mode: source_image shape = {batch['source_image'].shape}")
    else:
        log.info(f"Image shape: {batch['image'].shape}")
        log.info(f"Attributes shape: {batch['attributes'].shape}")


if __name__ == "__main__":
    main()

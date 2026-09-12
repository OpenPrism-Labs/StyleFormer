"""Hydra-configured data smoke check; no model is created or trained."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Load the configured data and inspect one batch (not a training run)."""
    import lightning as L

    from src.data.datamodule import FaceDataModule
    from src.utils.logging import setup_logging, setup_wandb

    setup_logging(level=logging.INFO)
    log.info("DATA SMOKE CHECK ONLY: no model is created or trained.")
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.experiment.seed, workers=True)

    datamodule = FaceDataModule.from_config(cfg)
    datamodule.setup("fit")
    log.info("Dataset: %s", cfg.data.name)
    log.info("Train samples: %d", len(datamodule.train_dataset))
    log.info("Val samples: %d", len(datamodule.val_dataset))

    train_loader = datamodule.train_dataloader()
    try:
        batch = next(iter(train_loader))
    except StopIteration as exc:
        raise RuntimeError(
            "Training loader has no batches. Check dataset splits and filters, "
            "reduce dataloader.batch_size, or set dataloader.drop_last=false."
        ) from exc

    if "source_image" in batch:
        log.info("Paired source image shape: %s", batch["source_image"].shape)
        log.info("Paired target image shape: %s", batch["target_image"].shape)
    else:
        log.info("Image shape: %s", batch["image"].shape)
        if "attributes" in batch:
            log.info("Attributes shape: %s", batch["attributes"].shape)

    if cfg.logging.get("project"):
        run = setup_wandb(
            project=cfg.logging.project,
            entity=cfg.logging.get("entity"),
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.experiment.name,
            tags=cfg.experiment.get("tags", []),
            offline=cfg.logging.get("offline", True),
        )
        if run is not None:
            run.finish()
    log.info("DATA SMOKE CHECK PASSED. Model implementation and training are not included.")


if __name__ == "__main__":
    main()

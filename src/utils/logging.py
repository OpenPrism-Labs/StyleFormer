"""Logging utilities."""

import logging
from pathlib import Path
from typing import Any

from rich.logging import RichHandler


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    name: str = "styleformer",
) -> logging.Logger:
    """Set up logging with rich console output.
    
    Args:
        level: Logging level.
        log_file: Optional file to write logs to.
        name: Logger name.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Rich console handler
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "styleformer") -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def setup_wandb(
    project: str,
    config: dict[str, Any],
    entity: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    offline: bool = False,
) -> Any:
    """Initialize Weights & Biases logging.
    
    Args:
        project: W&B project name.
        config: Configuration dictionary.
        entity: W&B entity (username or team).
        name: Run name.
        tags: Run tags.
        offline: Run in offline mode.
        
    Returns:
        W&B run object.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb not installed. Install with: pip install wandb"
        )
    
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        config=config,
        mode="offline" if offline else "online",
    )
    
    return run

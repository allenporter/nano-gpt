"""Utilities for saving and loading checkpoints of the model and training."""

import dataclasses
from dataclasses import dataclass
import pathlib
from typing import Any
import logging

import torch

from .config import GPTConfig, TrainConfig, DatasetConfig, EvalConfig

_LOGGER = logging.getLogger(__name__)


CHECKPOINT_DIR = pathlib.Path("checkpoints")


@dataclass(frozen=True, kw_only=True)
class Checkpoint:
    """Checkpoint of the model and training state."""

    model_state_dict: dict[str, Any]
    """State dict of the model."""

    config: GPTConfig
    """Config of the model."""

    step: int | None = None
    """Number of steps the model has been trained for."""

    val_loss_accum: float | None = None
    """Accumulated validation loss."""

    optimizer_state_dict: dict[str, Any] | None = None
    """State dict of the optimizer."""

    train_config: TrainConfig | None
    """Config of the training."""

    dataset_config: DatasetConfig | None
    """Config of the dataset."""

    eval_config: EvalConfig | None
    """Config of the evaluation."""


def save_checkpoint(
    checkpoint: Checkpoint,
    checkpoint_path: pathlib.Path,
) -> None:
    """Save the model to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dict = dataclasses.asdict(checkpoint)
    _LOGGER.debug(
        "Saving model checkpoint on step %s to %s", checkpoint.step, checkpoint_path
    )
    torch.save(checkpoint_dict, str(checkpoint_path))


def load_checkpoint(checkpoint_path: pathlib.Path) -> Checkpoint:
    """Load the model from disk."""
    checkpoint_dict = torch.load(str(checkpoint_path))
    return Checkpoint(
        model_state_dict=checkpoint_dict["model_state_dict"],
        config=GPTConfig(**checkpoint_dict["config"]),
        step=checkpoint_dict["step"],
        val_loss_accum=checkpoint_dict["val_loss_accum"],
        optimizer_state_dict=checkpoint_dict["optimizer_state_dict"],
        train_config=TrainConfig(**checkpoint_dict["train_config"]),
        dataset_config=DatasetConfig(**checkpoint_dict["dataset_config"]),
        eval_config=EvalConfig(**checkpoint_dict["eval_config"]),
    )

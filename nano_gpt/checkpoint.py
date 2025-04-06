"""Utilities for saving and loading checkpoints of the model and training."""

import dataclasses
from dataclasses import dataclass
import json
import logging
import pathlib
from typing import Any

import torch
import safetensors.torch as st

from .config import GPTConfig, TrainConfig, DatasetConfig, EvalConfig, SampleConfig
from .model import PRETRAINED_TRANSPOSED_WEIGHTS

__all__ = [
    "Checkpoint",
    "save_checkpoint",
    "load_checkpoint",
]

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

    train_config: TrainConfig
    """Config of the training."""

    dataset_config: DatasetConfig | None
    """Config of the dataset."""

    eval_config: EvalConfig | None
    """Config of the evaluation."""

    sample_config: SampleConfig | None
    """Config of the sampling."""

    name: str | None = None
    """Name of the checkpoint."""

    @property
    def model_state_dict_for_inference(self) -> dict[str, Any]:
        """Return the model state dict for inference."""
        new_state_dict = {}
        for k, v in self.model_state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[len("_orig_mod.") :]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict


def save_checkpoint(
    checkpoint: Checkpoint,
    checkpoint_path: pathlib.Path,
) -> None:
    """Save the model to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dict = dataclasses.asdict(checkpoint)
    _LOGGER.info("Saving model checkpoint to %s", checkpoint_path)
    torch.save(checkpoint_dict, str(checkpoint_path))
    _LOGGER.debug("Checkpoint saved")


def load_checkpoint(
    checkpoint_path: pathlib.Path, device: str | None = None
) -> Checkpoint:
    """Load the model from disk."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint_dict = torch.load(str(checkpoint_path), map_location=device)
    train_config_values = checkpoint_dict["train_config"]
    # The training starting step gets updated to the step at which the checkpoint was saved
    train_config_values["step"] = checkpoint_dict["step"]
    return Checkpoint(
        name=checkpoint_path.stem,
        model_state_dict=checkpoint_dict["model_state_dict"],
        config=GPTConfig(**checkpoint_dict["config"]),
        step=checkpoint_dict["step"],
        val_loss_accum=checkpoint_dict["val_loss_accum"],
        optimizer_state_dict=checkpoint_dict["optimizer_state_dict"],
        train_config=TrainConfig(**train_config_values),
        dataset_config=DatasetConfig(**checkpoint_dict["dataset_config"]),
        eval_config=EvalConfig(**checkpoint_dict["eval_config"]),
        sample_config=SampleConfig(**checkpoint_dict["sample_config"]),
    )


def export_checkpoint(
    checkpoint: Checkpoint,
    checkpoint_path: pathlib.Path,
    export_dir: pathlib.Path,
) -> None:
    """Export the checkpoint to safetensors format."""
    if not export_dir.exists():
        export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "model.safetensors"
    config_path = export_dir / "config.json"
    if export_path.exists():
        raise FileExistsError(f"Model export already exists: {export_path}")
    if config_path.exists():
        raise FileExistsError(f"Config file already exists: {config_path}")
    _LOGGER.info("Exporting checkpoint to %s", export_path)
    metadata = {
        "format": "pt",
        "model_name": checkpoint.name or "gpt2",
    }
    original_state = checkpoint.model_state_dict_for_inference
    # Put the weights in the format expected by GPT2LMHeadModel. See the
    # code in model.py which transposes the weights which is the inverse of
    # this operation.
    loaded = {}
    for k, v in original_state.items():
        if any(k.endswith(w) for w in PRETRAINED_TRANSPOSED_WEIGHTS):
            with torch.no_grad():
                loaded[k] = v.t().contiguous()
        else:
            loaded[k] = v.contiguous()

    model_config = dataclasses.asdict(checkpoint.config)
    config = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "n_ctx": model_config["block_size"],
        **model_config,
        "val_loss_accum": checkpoint.val_loss_accum,
        "train_config": dataclasses.asdict(checkpoint.train_config),
    }
    if checkpoint.dataset_config:
        config["dataset_config"] = dataclasses.asdict(checkpoint.dataset_config)
    if checkpoint.eval_config or checkpoint.sample_config:
        config["task_specific_params"] = {}
        if checkpoint.eval_config:
            config["task_specific_params"]["eval_config"] = dataclasses.asdict(
                checkpoint.eval_config
            )
        if checkpoint.sample_config:
            config["task_specific_params"]["sample_config"] = dataclasses.asdict(
                checkpoint.sample_config
            )

    st.save_file(loaded, str(export_path), metadata=metadata)
    config_path.write_text(json.dumps(config, indent=4))
    _LOGGER.debug("Checkpoint exported")

    _LOGGER.info("Verifying exported checkpoint")
    pt_size = checkpoint_path.stat().st_size
    size_diff = abs(pt_size - export_path.stat().st_size)
    diff_pct = (1.0 * size_diff) / pt_size
    _LOGGER.info("Exported file size: %d bytes", export_path.stat().st_size)
    _LOGGER.info("Original file size: %d bytes", pt_size)
    _LOGGER.info("Difference: %d bytes (%.2f%%)", size_diff, diff_pct * 100)

    reloaded = st.load_file(str(export_path))
    _LOGGER.info(
        "Verifying tensors (%d tensors)", len(checkpoint.model_state_dict_for_inference)
    )
    for k, pt_tensor in checkpoint.model_state_dict_for_inference.items():
        _LOGGER.debug("Verifying tensor %s", k)
        sf_tensor = reloaded[k]
        if any(k.endswith(w) for w in PRETRAINED_TRANSPOSED_WEIGHTS):
            if not torch.equal(pt_tensor.t(), sf_tensor):
                raise RuntimeError(f"The output tensors do not match for key {k}")
        else:
            if not torch.equal(pt_tensor, sf_tensor):
                raise RuntimeError(f"The output tensors do not match for key {k}")
    _LOGGER.info("All tensors match")

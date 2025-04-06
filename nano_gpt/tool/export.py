"""Command-line interface for exporting a model checkpoint to safet tensor format.

Usage:
```
usage: nano-gpt export [-h] [--pretrained {gpt2,gpt2-large,gpt2-medium,gpt2-xl}]
                       [--model {gpt2,gpt2-large,gpt2-medium,gpt2-xl,gpt2-xs,gpt2-xxs}] [--checkpoint CHECKPOINT] [--device DEVICE]
                       [--sequence-length SEQUENCE_LENGTH] [--seed SEED] [--compile | --no-compile] --export-path EXPORT_PATH

Export a model checkpoint to safetensors

options:
  -h, --help            show this help message and exit

model:
  --pretrained {gpt2,gpt2-large,gpt2-medium,gpt2-xl}
                        The name of the pretrained model to use.
  --model {gpt2,gpt2-large,gpt2-medium,gpt2-xl,gpt2-xs,gpt2-xxs}
                        Use the specified model name configuration default values.
  --checkpoint CHECKPOINT
                        Load a model from a checkpoint.
  --device DEVICE       The device to use.
  --sequence-length SEQUENCE_LENGTH
                        The sequence length used for input content in each micro batch.
  --seed SEED           The seed to use for sampling/training.
  --compile, --no-compile
                        Will compile the model if supported by the device.

export:
  --export-path EXPORT_PATH
                        Destination model path for exporting a checkpoint to safetensors.
```
"""

import argparse
import logging
import pathlib

from nano_gpt.checkpoint import export_checkpoint

from .model_config import (
    create_model_arguments,
    load_checkpoint_context,
)

_LOGGER = logging.getLogger(__name__)


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    create_model_arguments(args)
    group = args.add_argument_group("export")
    group.add_argument(
        "--export-dir",
        type=str,
        help="Destination directory path for exporting a checkpoint to safetensors.",
        required=True,
    )


def run(args: argparse.Namespace) -> int:
    """Run the export command."""

    with load_checkpoint_context(args) as checkpoint:
        if checkpoint is None:
            raise ValueError("Required flag --checkpoint not specified or not found")
        checkpoint_path = pathlib.Path(args.checkpoint)
        export_path = pathlib.Path(args.export_dir)
        export_checkpoint(
            checkpoint,
            checkpoint_path,
            export_path,
        )

    return 0

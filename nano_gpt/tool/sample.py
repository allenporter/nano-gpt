"""Command-line interface for sampling from a trained model.

Usage:
```
usage: nano-gpt sample [-h] [--pretrained {gpt2,gpt2-large,gpt2-medium,gpt2-xl}]
                       [--model {gpt2,gpt2-large,gpt2-medium,gpt2-xl,gpt2-xs,gpt2-xxs}] [--checkpoint CHECKPOINT] [--device DEVICE]
                       [--sequence-length SEQUENCE_LENGTH] [--seed SEED] [--compile | --no-compile]
                       [--sample-num-sequences SAMPLE_NUM_SEQUENCES] [--sample-max-length SAMPLE_MAX_LENGTH]
                       [--sample-seed SAMPLE_SEED]
                       [text ...]

Sample from a model

positional arguments:
  text                  The text to use as a prompt for sampling.

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

sample:
  --sample-num-sequences SAMPLE_NUM_SEQUENCES
                        The number of sequences to generate.
  --sample-max-length SAMPLE_MAX_LENGTH
                        The maximum length of the generated sequences.
  --sample-seed SAMPLE_SEED
                        The seed to use for sampling.
```
"""

import argparse
import logging
import dataclasses

import torch

from nano_gpt.model import sample

from .model_config import (
    create_model_arguments,
    model_from_args,
    create_sample_arguments,
    sample_config_from_args,
    load_checkpoint_context,
)


_LOGGER = logging.getLogger(__name__)


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    create_model_arguments(args, default_values={"seed": 42, "pretrained": "gpt2"})
    create_sample_arguments(args)
    args.add_argument(
        "text",
        type=str,
        nargs="*",
        default=["Hello, I'm a language model,"],
        help="The text to use as a prompt for sampling.",
    )


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    with load_checkpoint_context(args) as checkpoint:
        sample_config = sample_config_from_args(args, checkpoint)
        sample_config = dataclasses.replace(
            sample_config,
            text=" ".join(args.text),
        )
        _LOGGER.info(f"Sample config: {sample_config}")

        model, _, _ = model_from_args(args, checkpoint)

    model.eval()

    print(args.text)
    with torch.no_grad():
        samples = sample(
            model,
            model.enc,
            sample_config.text,
            num_return_sequences=sample_config.num_return_sequences,
            max_length=sample_config.max_length,
            device=args.device,
            seed=sample_config.seed,
        )
    for s in samples:
        print(">", s)

    return 0

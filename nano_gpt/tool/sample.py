"""Command-line interface for sampling from a trained model.

Usage:
```
usage: nano-gpt sample [-h] [--pretrained {gpt2-medium,gpt2-xl,gpt2,gpt2-large}]
                       [--model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs}] [--device DEVICE]
                       [--sequence-length SEQUENCE_LENGTH] [--seed SEED] [--compile | --no-compile]
                       [--sample-num-sequences SAMPLE_NUM_SEQUENCES] [--sample-max-length SAMPLE_MAX_LENGTH]
                       [text ...]

Sample from a model

positional arguments:
  text                  The text to use as a prompt for sampling.

options:
  -h, --help            show this help message and exit

model:
  --pretrained {gpt2-medium,gpt2-xl,gpt2,gpt2-large}
                        The name of the pretrained model to use.
  --model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs}
                        Use the specified model name configuration default values.
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
```
"""

import argparse
import logging
import dataclasses

from .model_config import (
    create_model_arguments,
    model_from_args,
    create_sample_arguments,
    sample_config_from_args,
)
from nano_gpt.model import sample


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
    sample_config = sample_config_from_args(args)
    sample_config = dataclasses.replace(
        sample_config,
        text=" ".join(args.text),
    )
    _LOGGER.info(f"Sample config: {sample_config}")

    model, _, _ = model_from_args(args)
    model.eval()

    print(args.text)
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

"""Command-line interface for sampling from a trained model.

Usage:
```
usage: nano-gpt sample [-h] [--pretrained {gpt2-xl,gpt2,gpt2-medium,gpt2-large}] [--model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs}]
                       [--device DEVICE] [--sequence-length SEQUENCE_LENGTH] [--seed SEED] [--num-sequences NUM_SEQUENCES] [--max-length MAX_LENGTH]
                       [text ...]

Sample from a model

positional arguments:
  text                  The text to use as a prompt for sampling.

options:
  -h, --help            show this help message and exit
  --num-sequences NUM_SEQUENCES
                        The number of sequences to generate.
  --max-length MAX_LENGTH
                        The maximum length of the generated sequences.

model:
  --pretrained {gpt2-xl,gpt2,gpt2-medium,gpt2-large}
                        The name of the pretrained model to use.
  --model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs}
                        Use the specified model name configuration default values.
  --device DEVICE       The device to use.
  --sequence-length SEQUENCE_LENGTH
                        The sequence length used for input content in each micro batch.
  --seed SEED           The seed to use for sampling/training.
```
"""

import argparse
import logging

import torch

from nano_gpt.model import GPT

_LOGGER = logging.getLogger(__name__)

from .model_config import create_model_arguments, model_from_args


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    create_model_arguments(
        args,
        default_values={
            "seed": 42,
            "pretrained": "gpt2"
        }
    )
    args.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="The number of sequences to generate.",
    )
    args.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="The maximum length of the generated sequences.",
    )
    args.add_argument(
        "text",
        type=str,
        nargs="*",
        default=["Hello, I'm a language model,"],
        help="The text to use as a prompt for sampling.",
    )
    


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    model, _, _ = model_from_args(args)
    model.eval()

    print(args.text)
    samples = model.sample(
        " ".join(args.text),
        args.num_sequences,
        args.max_length,
        device=args.device,
    )
    for sample in samples:
        print(">", sample)

    return 0

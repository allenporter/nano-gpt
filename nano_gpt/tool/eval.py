"""Command-line interface for evaling a trained model.

Usage:
```
usage: nano-gpt eval [-h] [--pretrained {gpt2-medium,gpt2-xl,gpt2-large,gpt2}] [--model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs}]
                     [--device DEVICE] [--sequence-length SEQUENCE_LENGTH] [--seed SEED]

Evaluate a model

options:
  -h, --help            show this help message and exit

model:
  --pretrained {gpt2-medium,gpt2-xl,gpt2-large,gpt2}
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

from nano_gpt.datasets import hellaswag
from nano_gpt import hellaswag_eval

from .model_config import create_model_arguments, model_from_args

_LOGGER = logging.getLogger(__name__)

DATASET = "hellaswag"
SPLIT = "validation"


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    create_model_arguments(args)
    args.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The number of samples to evaluate or all of omitted.",
    )


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    torch.set_float32_matmul_precision("high")

    model, tokenizer, _ = model_from_args(args)
    model.eval()

    hellaswag_val = hellaswag.load_dataset(SPLIT)
    result = hellaswag_eval.evaluate(
        model, tokenizer, hellaswag_val, args.device, num_samples=args.num_samples
    )
    print(result)

    return 0

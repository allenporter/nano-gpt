"""Command-line interface for sampling from a trained model.

Usage:
```
usage: nano-gpt sample [-h] [--pretrained PRETRAINED] [--device DEVICE] [--num-sequences NUM_SEQUENCES] [--max-length MAX_LENGTH]
                       [--seed SEED]
                       [text ...]

Sample from a model

positional arguments:
  text                  The text to use as a prompt for sampling.

options:
  -h, --help            show this help message and exit
  --pretrained PRETRAINED
                        The name of the pretrained model to use when sampling.
  --device DEVICE       The device to use for sampling.
  --num-sequences NUM_SEQUENCES
                        The number of sequences to generate.
  --max-length MAX_LENGTH
                        The maximum length of the generated sequences.
  --seed SEED           The seed to use for sampling.
```
"""

import argparse

import torch

from nano_gpt.model import GPT
from nano_gpt.tokenizer import get_tokenizer


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    args.add_argument(
        "--pretrained",
        type=str,
        help="The name of the pretrained model to use when sampling.",
    )
    args.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use for sampling.",
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
        "--seed",
        type=int,
        default=42,
        help="The seed to use for sampling.",
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
    model = GPT.from_pretrained(args.pretrained, get_tokenizer())
    model.eval()
    model = model.to(args.device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
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

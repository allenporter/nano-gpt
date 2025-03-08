"""Command-line interface for sampling from a trained model.

Usage:
```
usage: nano-gpt sample [-h] [--from-pretrained FROM_PRETRAINED]

Sample from a model

options:
  -h, --help            show this help message and exit
  --from-pretrained FROM_PRETRAINED
                        The name of the pretrained model to use when sampling.
```
"""

import argparse


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    args.add_argument(
        "--from-pretrained",
        type=str,
        help="The name of the pretrained model to use when sampling.",
    )


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    print("Running sample command")
    return 0

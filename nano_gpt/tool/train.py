"""Command-line interface for training the model.

Usage:
```
usage: nano-gpt train [-h] [--device DEVICE] [--batch_size BATCH_SIZE] [--sequence-length SEQUENCE_LENGTH]
                      [--seed SEED]

Sample from a model

options:
  -h, --help            show this help message and exit
  --device DEVICE       The device to use for sampling.
  --batch_size BATCH_SIZE
                        The number of batches of input from the dataset for training step.
  --sequence-length SEQUENCE_LENGTH
                        The sequence length used for input content.
  --seed SEED           The seed to use for training.
```
"""

import argparse
import logging
from typing import cast

import torch

from nano_gpt.model import GPT, GPTConfig
from nano_gpt.tokenizer import get_tokenizer
from nano_gpt.devices import get_device, get_dtype
from nano_gpt.data import get_data_loader
from nano_gpt import trainer

_LOGGER = logging.getLogger(__name__)


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    args.add_argument(
        "--device",
        type=str,
        default=get_device(),
        help="The device to use for sampling.",
    )
    args.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The size of the batch to use in each training step.",
    )
    args.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
        help="The sequence length used for input content.",
    )
    args.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to use for training.",
    )


# Increased from GPT-2 vocab size for training using a nice power of 2
VOCAB_SIZE = 50304


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    model_config = GPTConfig(
        vocab_size=VOCAB_SIZE,
    )
    tokenizer = get_tokenizer()
    model = GPT(model_config, tokenizer=tokenizer)
    _LOGGER.info("Using device %s", args.device)
    model = model.to(args.device)
    if args.device == "cuda":
        _LOGGER.debug("Compiling model")
        model = cast(GPT, torch.compile(model))
    else:
        _LOGGER.debug("Model will not be compiled")

    train_config = trainer.TrainConfig(
        B=args.batch_size,
        T=args.sequence_length,
    )
    data_loader = get_data_loader(
        enc=tokenizer,
        batch_size=train_config.B,
        token_len=train_config.T,
        device=args.device,
    )
    trainer.train(
        model, args.device, data_loader, train_config, dtype=get_dtype(args.device)
    )

    return 0

"""Command-line interface for training the model.

Usage:
```
usage: nano-gpt train [-h] [--device DEVICE] [--model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs,gpt2-xxxs}]
                      --dataset {tinyshakespeare,finewebedu} [--micro-batch-size MICRO_BATCH_SIZE] [--total-batch-size TOTAL_BATCH_SIZE]
                      [--sequence-length SEQUENCE_LENGTH] [--batch-size BATCH_SIZE] [--seed SEED]

Sample from a model

options:
  -h, --help            show this help message and exit
  --device DEVICE       The device to use for sampling.
  --model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs,gpt2-xxxs}
                        Use the specified model name configuration default values.
  --dataset {tinyshakespeare,finewebedu}
                        Use the specified dataset.
  --micro-batch-size MICRO_BATCH_SIZE
                        The number of batches of examples to use in each training micro step.
  --total-batch-size TOTAL_BATCH_SIZE
                        The number of the batch to use in each training step.
  --sequence-length SEQUENCE_LENGTH
                        The sequence length used for input content in each micro batch.
  --batch-size BATCH_SIZE
                        The number of tokens to use in each gradient accumulation batches.
  --seed SEED           The seed to use for training.
```
"""

import argparse
import logging
from typing import cast

import torch

from nano_gpt.config import config_from, MODELS
from nano_gpt.model import GPT
from nano_gpt.devices import get_device, get_dtype
from nano_gpt.datasets import tinyshakespeare, finewebedu
from nano_gpt.datasets.data_loader import preprocess_dataset
from nano_gpt.tokenizer import get_tokenizer
from nano_gpt.trainer import train


_LOGGER = logging.getLogger(__name__)


DATASETS = {
    "tinyshakespeare": tinyshakespeare.load_dataset,
    "finewebedu": finewebedu.load_dataset,
}


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    args.add_argument(
        "--device",
        type=str,
        default=get_device(),
        help="The device to use for sampling.",
    )
    args.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=MODELS,
        help="Use the specified model name configuration default values.",
    )
    args.add_argument(
        "--dataset",
        type=str,
        help="Use the specified dataset.",
        choices=DATASETS,
        required=True,
    )
    args.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="The number of batches of examples to use in each training micro step.",
    )
    args.add_argument(
        "--total-batch-size",
        type=int,
        default=None,
        help="The number of the batch to use in each training step.",
    )
    args.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="The sequence length used for input content in each micro batch.",
    )
    args.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="The number of tokens to use in each gradient accumulation batches.",
    )
    args.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to use for training.",
    )


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    config = config_from(
        args.model,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.sequence_length,
        total_batch_size=args.total_batch_size,
    )
    _LOGGER.debug("Config: %s", config)
    tokenizer = get_tokenizer()
    model = GPT(config.model_config, tokenizer=tokenizer)
    _LOGGER.info("Using device %s", args.device)
    model = model.to(args.device)
    if args.device == "cuda":
        _LOGGER.debug("Compiling model")
        model = cast(GPT, torch.compile(model))
    else:
        _LOGGER.debug("Model will not be compiled")

    dataset_fn = DATASETS[args.dataset]
    _LOGGER.info("Loading dataset %s", args.dataset)

    data_loader = preprocess_dataset(
        dataset_fn(split="train"),
        enc=tokenizer,
        config=config.train_config.dataset_config,
    )
    _LOGGER.info("Dataset loaded")
    train(
        model,
        args.device,
        data_loader,
        config.train_config,
        dtype=get_dtype(args.device),
    )

    return 0

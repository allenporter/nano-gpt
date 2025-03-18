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
import pathlib

import torch

from nano_gpt.devices import get_dtype
from nano_gpt.datasets import TRAIN_DATASETS
from nano_gpt.datasets.data_loader import read_preprocessed_corpus
from nano_gpt.trainer import train

from .model_config import (
    create_model_arguments,
    model_from_args,
    DATASET_DIR,
    create_eval_arguments,
    create_sample_arguments,
)


_LOGGER = logging.getLogger(__name__)


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    create_model_arguments(args)
    args.add_argument(
        "--dataset",
        type=str,
        help="Use the specified dataset.",
        choices=TRAIN_DATASETS.keys(),
        required=True,
    )
    args.add_argument(
        "--dataset-dir",
        type=str,
        help="Directory where the dataset is stored.",
        default=DATASET_DIR,
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
        help="The number of tokens to use in each gradient accumulation batch (of micro-batches).",
    )
    args.add_argument(
        "--streaming",
        type=str,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream the dataset without downloading the entire corpus.",
    )
    args.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="The maximum number of training steps.",
    )
    args.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="The number of steps between evaluations.",
    )
    create_eval_arguments(args)
    create_sample_arguments(args)


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    torch.set_float32_matmul_precision("high")

    model, tokenizer, config = model_from_args(args)
    if config is None:
        raise ValueError("No trainable model configuration found")

    _LOGGER.info("Loading dataset %s (streaming=%s)", args.dataset, args.streaming)

    dataset_dir = pathlib.Path(args.dataset_dir)
    train_ds = read_preprocessed_corpus(
        dataset_dir / f"{args.dataset}_train.npy", config.train_config.dataset_config
    )
    # TODO: Add back eval to the trainer
    # eval_ds = read_preprocessed_corpus(
    #     dataset_dir / f"{args.dataset}_validation.npy",
    #     config.train_config.dataset_config,
    # )
    _LOGGER.info("Dataset loaded")
    train(
        model,
        args.device,
        train_ds,
        config.train_config,
        dtype=get_dtype(args.device),
    )

    return 0

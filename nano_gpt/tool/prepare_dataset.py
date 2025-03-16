"""Command-line interface for preparing datasets for training runs.

Usage:
```
usage: nano-gpt prepare_dataset [-h] --dataset {tinyshakespeare,finewebedu} [--splits SPLITS]
                                [--tokens-per-shard TOKENS_PER_SHARD] [--dataset-dir DATASET_DIR]
                                [--num-procs NUM_PROCS]

Evaluate a model

options:
  -h, --help            show this help message and exit
  --dataset {tinyshakespeare,finewebedu}
                        Use the specified dataset.
  --splits SPLITS       Use the specified dataset.
  --tokens-per-shard TOKENS_PER_SHARD
                        Number of tokens per shard.
  --dataset-dir DATASET_DIR
                        Directory to store the dataset.
  --num-procs NUM_PROCS
                        Number of processes to use for preprocessing.
```
"""

import argparse
import logging
import os
import pathlib

from nano_gpt.datasets import tinyshakespeare, finewebedu
from nano_gpt.datasets.data_loader import preprocess_corpus
from nano_gpt.tokenizer import get_tokenizer


_LOGGER = logging.getLogger(__name__)

DATASET_DIR = "dataset_cache"
DATASETS = {
    "tinyshakespeare": tinyshakespeare.load_dataset,
    "finewebedu": finewebedu.load_dataset,
}
SPLITS = {"train", "validation"}


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    args.add_argument(
        "--dataset",
        type=str,
        help="Use the specified dataset.",
        choices=DATASETS,
        required=True,
    )
    args.add_argument(
        "--splits",
        type=str,
        help="Use the specified dataset.",
        default=",".join(SPLITS),
    )
    args.add_argument(
        "--tokens-per-shard",
        type=int,
        help="Number of tokens per shard.",
        default=10e6,  # 10 million tokens
    )
    args.add_argument(
        "--dataset-dir",
        type=str,
        help="Directory to store the dataset.",
        default=DATASET_DIR,
    )
    args.add_argument(
        "--num-procs",
        type=int,
        help="Number of processes to use for preprocessing.",
        default=(os.cpu_count() or 1) // 2,
    )


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""

    dataset_dir = pathlib.Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer()

    dataset_fn = DATASETS[args.dataset]
    _LOGGER.info("Loading dataset %s", args.dataset)

    splits = args.splits.split(",")
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split}, must be one of {SPLITS}")
        _LOGGER.info("Loading dataset %s for split %s", args.dataset, split)
        output_path = dataset_dir / f"{args.dataset}_{split}.npy"
        ds = dataset_fn(split=split, streaming=False)
        preprocess_corpus(
            ds,
            tokenizer,
            output_path,
            num_procs=args.num_procs or 1,
            tokens_per_shard=args.tokens_per_shard,
        )

    return 0

"""Command-line interface for preparing datasets for training runs.

Usage:
```
usage: nano-gpt prepare_dataset [-h] --dataset {tinyshakespeare,finewebedu}

Evaluate a model

options:
  -h, --help            show this help message and exit
  --dataset {tinyshakespeare,finewebedu}
                        Use the specified dataset.
```
"""

import argparse
import logging


from nano_gpt.datasets import tinyshakespeare, finewebedu


_LOGGER = logging.getLogger(__name__)


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


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    dataset_fn = DATASETS[args.dataset]
    _LOGGER.info("Loading dataset %s", args.dataset)

    splits = args.splits.split(",")
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split}, must be one of {SPLITS}")
        _LOGGER.info("Loading dataset %s for split %s", args.dataset, split)
        dataset_fn(split=split, streaming=False)

    return 0

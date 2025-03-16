"""Data loader library for the finewebedu 10B dataset.

This is a thin wrapper around the HuggingFace datasets library that
handles sharding the dataset.

See https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import logging

import datasets


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "load_dataset",
]

TOKEN_SIZE = 2**20
SHARD_TOKEN_SIZE = int(1e8)  # 100M tokens

# This dataset only has a train split so we create a validation split
# by taking the last 10% of the training data.
SPLITS = {
    "train": "train[:90%]",
    "validation": "train[90%:]",
}


def load_dataset(split: str, streaming: bool = True) -> datasets.Dataset:
    """Load the dataset."""
    return datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        streaming=streaming,
        split="train",
    )

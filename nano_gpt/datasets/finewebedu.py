"""Data loader library for the finewebedu 10B dataset.

This is a thin wrapper around the HuggingFace datasets library that
handles sharding the dataset.

See https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import logging
from collections.abc import Iterable

import datasets

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "load_dataset",
]

TOKEN_SiZE = 2**20


def load_dataset(split: str) -> Iterable[str]:
    """Load the dataset."""
    ds = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", streaming=True, split=split
    )
    return map(lambda x: x["text"], ds)

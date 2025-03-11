"""Data loader library for the tinyshakespeare dataset.

This is a thin wrapper around the HuggingFace datasets library.
"""

from collections.abc import Iterable

import datasets

from .data_loader import MapIterable

__all__ = [
    "load_dataset",
]


def load_dataset(split: str) -> Iterable[str]:
    """Load the dataset."""
    ds = datasets.load_dataset("tiny_shakespeare", trust_remote_code=True, split=split)
    return MapIterable(lambda x: x["text"], ds)

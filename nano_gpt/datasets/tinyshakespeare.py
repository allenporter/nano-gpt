"""Data loader library for the tinyshakespeare dataset.

This is a thin wrapper around the HuggingFace datasets library.
"""

import datasets


__all__ = [
    "load_dataset",
]


SHARD_TOKEN_SIZE = int(100000)


def load_dataset(split: str, streaming: bool = True) -> datasets.Dataset:
    """Load the dataset.

    Streaming flag is ignored because the tinyshakespeare dataset is small.
    """
    return datasets.load_dataset(
        "tiny_shakespeare", trust_remote_code=True, split=split
    )

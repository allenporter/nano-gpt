"""Data loader library for the tinyshakespeare dataset.

This is a thin wrapper around the HuggingFace datasets library.
"""

import datasets

from nano_gpt.config import TrainDataset


__all__ = [
    "DATASET",
    "load_dataset",
]


def load_dataset(split: str, streaming: bool = True) -> datasets.Dataset:
    """Load the dataset.

    Streaming flag is ignored because the tinyshakespeare dataset is small.
    """
    return datasets.load_dataset(
        "tiny_shakespeare", trust_remote_code=True, split=split
    )


DATASET = TrainDataset(
    name="tinyshakespeare",
    load_fn=load_dataset,
    total_tokens=301967,  # Approximately 300k tokens
    # Seta limit higher than the dataset size. The entire dataset is a
    # single record so it must fit in 1 shard.
    tokens_per_shard=400000,
)

"""Data loader library for the finewebedu 10B dataset.

This is a thin wrapper around the HuggingFace datasets library that
handles sharding the dataset.

See https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import logging

import datasets

from nano_gpt.config import TrainDataset


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "DATASET",
    "load_dataset",
]


# This dataset only has a train split so we create a validation split
# by taking the last 10% of the training data.
_SPLITS = {
    "train": "train[:97%]",
    "validation": "train[97%:]",
}


def load_dataset(split: str, streaming: bool = True) -> datasets.Dataset:
    """Load the dataset."""
    if split not in _SPLITS:
        raise ValueError(
            f"Invalid split: {split}. Must be one of {list(_SPLITS.keys())}."
        )
    return datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        streaming=streaming,
        split=_SPLITS[split],
    )


DATASET = TrainDataset(
    name="finewebedu",
    load_fn=load_dataset,
    total_tokens=int(10e9),  # 10B tokens,
    tokens_per_shard=int(100e6),  # 100M tokens
)

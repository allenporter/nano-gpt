"""Datasets for training and evaluating the model."""

from . import finewebedu, tinyshakespeare

__all__ = [
    "TRAIN_DATASETS",
    "finewebedu",
    "tinyshakespeare",
    "hellaswag",
]


TRAIN_DATASETS_LIST = [
    finewebedu.DATASET,
    tinyshakespeare.DATASET,
]
TRAIN_DATASETS = {dataset.name: dataset for dataset in TRAIN_DATASETS_LIST}

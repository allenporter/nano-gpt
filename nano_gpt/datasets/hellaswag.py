"""Data loader library for the hellaswag dataset."""

from dataclasses import dataclass
from collections.abc import Iterable
import logging
from typing import Any

import datasets
import torch

from .data_loader import MapIterable
from nano_gpt.tokenizer import Tokenizer

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "load_dataset",
]


NUM_ANSWERS = 4


@dataclass(frozen=True, kw_only=True)
class Sample:
    """A sample multiple choice question."""

    prefix: str
    """The prefix of the question."""

    endings: list[str]
    """List of possible endings."""

    label: int
    """Index of the correct ending."""

    @property
    def ending_texts(self) -> list[str]:
        """Return the completion candidates."""
        return [f" {ending}" for ending in self.endings]

    @property
    def completions(self) -> list[str]:
        """Return the completion candidates."""
        return [f"{self.prefix} {ending}" for ending in self.endings]

    @property
    def max_len(self) -> int:
        """Return the maximum length of the sample."""
        return max(len(row) for row in self.completions)

    def tokenize(self, tokenizer: Tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a sample and return the tokens and mask."""

        max_len = self.max_len
        prefix_toks = tokenizer.encode(self.prefix)
        prefix_masks = [0] * len(prefix_toks)

        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)
        for i, ending in enumerate(self.ending_texts):
            ending_toks = tokenizer.encode(ending)
            ending_mask = [1] * len(ending_toks)
            tok_row = prefix_toks + ending_toks
            mask_row = prefix_masks + ending_mask
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            mask[i, : len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask


def _make_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        prefix=record["ctx"], endings=record["endings"], label=int(record["label"])
    )


def load_dataset(split: str) -> Iterable[Sample]:
    """Load the dataset."""
    ds = datasets.load_dataset("Rowan/hellaswag", split=split)
    return MapIterable(_make_sample, ds)

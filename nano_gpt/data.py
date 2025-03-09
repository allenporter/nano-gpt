"""Data loader library.

This is a thin wrapper around the HuggingFace datasets library.
"""

from typing import Any, Self
import logging
from collections.abc import Iterator

import datasets
import torch

from .tokenizer import Tokenizer
from .config import DatasetConfig

_LOGGER = logging.getLogger(__name__)


def get_data_loader(
    enc: Tokenizer,
    config: DatasetConfig,
    device: Any,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Get the data loader."""
    return DataLoader(enc, config, device)


def _load_dataset() -> Any:
    """Load the dataset."""
    ds = datasets.load_dataset("tiny_shakespeare", trust_remote_code=True)
    return ds["train"]["text"][0]


class DataLoader:
    """Data loader to load batches from the dataset."""

    def __init__(self, enc: Tokenizer, config: DatasetConfig, device: Any) -> None:
        """Initialize Dataloader."""
        self.B = config.micro_batch_size
        self.T = config.sequence_length
        self.chunk_size = config.chunk_token_size

        self.data = _load_dataset()
        self.tokens = torch.tensor(enc.encode(self.data))
        self.pos = 0
        print(f"Loaded {len(self.tokens)} tokens")
        # Number of unique batches before we start the dataset over
        print(f"1 epoch = {len(self.tokens) // self.chunk_size} batches")

    def __iter__(self) -> "Self":
        self.pos = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch in the dataset."""
        # B = batch size
        # T = sequence of tokens (less than max sequence length)
        # The buf contains an extra token to use in the labels. The x
        # input doesn't include that last token. The labels starts with the first token.
        B, T = self.B, self.T
        buf = self.tokens[self.pos : self.pos + self.chunk_size + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += self.chunk_size
        if (self.pos + self.chunk_size + 1) > len(self.tokens):
            _LOGGER.info("Reached epoch")
            self.pos = 0
        return x, y

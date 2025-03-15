"""Data loader library or wrapping HuggingFace datasets."""

import itertools
from collections.abc import Iterable, Generator, Callable, Iterator
import logging
import pathlib
from typing import TypeVar

import torch
import datasets
import numpy as np

from nano_gpt.tokenizer import Tokenizer
from nano_gpt.config import DatasetConfig

__all__ = ["preprocess_dataset"]

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")
_V = TypeVar("_V")


class MapIterable(Iterable[_T]):
    """A mapped iterable that can restart.

    This is similar to itertools.map() but it can be restarted.
    """

    def __init__(self, map_func: Callable[[_V], _T], iterable: Iterable[_V]) -> None:
        """Initialize the mapped iterable."""
        self._map_func = map_func
        self._iterable = iterable

    def __iter__(self) -> Iterator[_T]:
        """Return the iterator."""
        new_it = iter(self._iterable)
        return map(self._map_func, new_it)


class ChainIterable(Iterable[_T]):
    """A chaining iterable that can restart.

    This is similar to itertools.chain.from_iterable but it can be restarted
    """

    def __init__(self, iterable: Iterable[list[_T]]) -> None:
        """Initialize the mapped iterable."""
        self._iterable = iterable

    def __iter__(self) -> Iterator[_T]:
        """Return the iterator."""
        new_it = iter(self._iterable)
        return itertools.chain.from_iterable(new_it)


def tokenize_dataset(
    enc: Tokenizer,
    ds: Iterable[str],
) -> Iterable[torch.Tensor]:
    """Tokenize the dataset and return as tensors."""
    tokenized_ds = MapIterable(enc.encode, ds)
    return MapIterable(torch.tensor, tokenized_ds)


def chunk_input(
    config: DatasetConfig,
    tokens: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Chunk the input into batches."""
    B, T = config.micro_batch_size, config.sequence_length
    pos = 0
    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    while (pos + config.chunk_token_size + 1) < len(tokens):
        buf = tokens[pos : pos + config.chunk_token_size + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        results.append((x, y))
        pos += config.chunk_token_size
    return results


def chunk_dataset(
    config: DatasetConfig,
    ds: Iterable[torch.Tensor],
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Chunk the dataset into batches of example inputs and labels based.

    This will ignore any data that is not a multiple of the chunk size. The
    chunk size is determined by the sequence length and the micro batch size of
    the dataset config.
    """

    def _chunk_input(
        tokens: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Chunk the input into batches."""
        return chunk_input(config, tokens)

    chunked = MapIterable(
        _chunk_input,
        ds,
    )
    return ChainIterable(chunked)


def cycle_dataset(ds: Iterable[_T]) -> Generator[_T]:
    """Cycle through the dataset.

    This is similar to itertools.cycle() but it can be restarted.
    """
    while True:
        samples = 0
        for example in ds:
            yield example
            samples += 1
        _LOGGER.info("Reached epoch")
        if samples == 0:
            raise ValueError("Empty dataset or dataset could not be restarted")


def preprocess_dataset(
    ds: Iterable[str],
    enc: Tokenizer,
    config: DatasetConfig,
) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """Preprocess the dataset."""
    tokenized_ds = tokenize_dataset(enc, ds)
    chunked_ds = chunk_dataset(config, tokenized_ds)
    return cycle_dataset(chunked_ds)


def preprocess_corpus(
    ds: datasets.Dataset,
    enc: Tokenizer,
    output_path: pathlib.Path,
    text_column: str = "text",
) -> None:
    """Preprocess a huggingface dataset and write to an output file."""
    text_ds = MapIterable(lambda x: x["text"], ds)
    tokenized_ds = tokenize_dataset(enc, text_ds)
    tokens = torch.concat(list(tokenized_ds))
    tokens_np = np.array(tokens)
    np.save(output_path, tokens_np)


def read_preprocessed_corpus(
    token_path: pathlib.Path,
    config: DatasetConfig,
) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """Read the preprocessed corpus."""
    tokens_np = np.load(token_path)
    tokens = torch.from_numpy(tokens_np)
    while True:
        for chunk in chunk_input(config, tokens):
            yield chunk

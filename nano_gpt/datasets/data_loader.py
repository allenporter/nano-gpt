"""Data loader library or wrapping HuggingFace datasets."""

import itertools
from collections.abc import Iterable, Generator, Callable, Iterator
import logging
import multiprocessing
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

PROCESS_CHUNK_SIZE = 16


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


class TokenizedFileWriter:
    """A file writer that writes tokenized files."""

    def __init__(self, path: pathlib.Path) -> None:
        """Initialize the file writer."""
        self._path = path
        self._tokens: list[torch.Tensor] = []
        self._num_tokens = 0

    @property
    def num_tokens(self) -> int:
        """Return the number of tokens."""
        return self._num_tokens

    def append(self, tokens: torch.Tensor) -> None:
        """Write the tokens to a file."""
        self._tokens.append(tokens)
        self._num_tokens += tokens.numel()

    def write(self) -> None:
        """Save the tokens to a file."""
        _LOGGER.debug("Writing %d tokens to %s", len(self._tokens), self._path)
        tokens = torch.concat(self._tokens)
        tokens_np = np.asarray(tokens, dtype=np.uint32)
        np.save(self._path, tokens_np)


class ShardedTokenizedFileWriter:
    """A file writer that writes tokenized files."""

    def __init__(self, path: pathlib.Path, tokens_per_shard: int) -> None:
        """Initialize the file writer."""
        self._path = path
        self._shard = -1
        self._tokens_per_shard = tokens_per_shard
        self._open_new_writer()

    def _open_new_writer(self) -> None:
        """Open a new writer."""
        self._shard += 1
        shard_filepath = pathlib.Path(f"{self._path}.{self._shard:03d}")
        _LOGGER.debug("Opening new writer for %s", shard_filepath)
        self._writer = TokenizedFileWriter(shard_filepath)

    def append(self, tokens: torch.Tensor) -> None:
        """Write the tokens to a file."""
        if self._writer.num_tokens + tokens.numel() > self._tokens_per_shard:
            if self._writer.num_tokens == 0:
                raise ValueError(
                    f"Too large for shard size: {tokens.numel()} > {self._tokens_per_shard}"
                )
            self._writer.write()
            self._open_new_writer()
        self._writer.append(tokens)

    def write(self) -> None:
        """Save the tokens to a file."""
        self._writer.write()


def preprocess_corpus(
    ds: datasets.Dataset,
    enc: Tokenizer,
    output_path: pathlib.Path,
    num_procs: int = 1,
    tokens_per_shard: int = 1000000,
    text_column: str = "text",
) -> None:
    """Preprocess a huggingface dataset and write to an output file."""
    text_ds = MapIterable(lambda x: x["text"], ds)

    with multiprocessing.Pool(num_procs) as pool:
        writer = ShardedTokenizedFileWriter(output_path, tokens_per_shard)
        for tokens in pool.imap(enc.encode, text_ds, chunksize=PROCESS_CHUNK_SIZE):
            writer.append(torch.tensor(tokens))
    writer.write()


class TokenizedFileReader:
    """A file reader that reads tokenized files."""

    def __init__(self, path: pathlib.Path) -> None:
        """Initialize the file reader."""
        self._path = path
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        self._tokens = np.load(path)
        _LOGGER.debug("Loaded %s with %d tokens", path, len(self._tokens))

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Return the iterator."""
        return iter([torch.from_numpy(self._tokens)])


class ShardedTokenizedFileReader:
    """A file reader for a set of sharded files identified by a glob filename."""

    def __init__(self, path: pathlib.Path) -> None:
        """Initialize the file reader."""
        self._path = path
        self._files = sorted(list(path.parent.glob(f"{path.name}.*")))
        _LOGGER.debug("Found %d files matching path: %s", len(self._files), path)
        if len(self._files) == 0:
            raise ValueError(f"No files found matching path: {path}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Return the iterator."""
        _LOGGER.debug("Starting iteration over sharded files")
        return iter(ShardedTokenizedFileReader(self._path)._iter())

    def _iter(self) -> Iterator[torch.Tensor]:
        """Iterate through the sharded files."""
        idx = 0
        while idx < len(self._files):
            yield from TokenizedFileReader(self._files[idx])
            idx += 1


def read_preprocessed_corpus(
    token_path: pathlib.Path,
    config: DatasetConfig,
) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """Read the preprocessed corpus."""
    reader = ShardedTokenizedFileReader(token_path)
    chunked_ds = chunk_dataset(config, reader)
    return cycle_dataset(chunked_ds)

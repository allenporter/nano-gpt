"""Data loader library or wrapping HuggingFace datasets."""

import itertools
from collections.abc import Iterable, Generator, Callable, Iterator
import logging
import multiprocessing
import pathlib
from typing import TypeVar

import datasets
import numpy as np
import torch
import tqdm

from nano_gpt.tokenizer import Tokenizer
from nano_gpt.config import DatasetConfig

__all__ = [
    "preprocess_corpus",
    "read_preprocessed_corpus",
]

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")
_V = TypeVar("_V")

PROCESS_CHUNK_SIZE = 32
SPLITS = {"train", "validation"}


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


class ChunkingIterable(Iterable[tuple[torch.Tensor, torch.Tensor]]):
    """An iterable that chunks the input into batches of example inputs and labels."""

    def __init__(
        self,
        config: DatasetConfig,
        tokens: torch.Tensor,
        worker_num: int = 0,
        worker_count: int = 1,
    ) -> None:
        """Initialize the chunking iterable."""
        if worker_num > worker_count:
            raise ValueError("worker_num must be less than worker_count")

        self._config = config
        self._tokens = tokens
        self._worker_num = worker_num
        self._worker_count = worker_count

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Return the iterator."""
        B, T = self._config.micro_batch_size, self._config.sequence_length
        pos = self._config.chunk_token_size * self._worker_num
        endpos = pos + self._config.chunk_token_size + 1
        while endpos <= len(self._tokens):
            buf = self._tokens[pos:endpos]
            x = buf[:-1].view(B, T)
            y = buf[1:].view(B, T)
            yield (x, y)
            pos += self._config.chunk_token_size * self._worker_count
            endpos += self._config.chunk_token_size * self._worker_count


def chunk_dataset(
    config: DatasetConfig,
    ds: Iterable[torch.Tensor],
    worker_num: int = 0,
    worker_count: int = 1,
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
        it = ChunkingIterable(config, tokens, worker_num, worker_count)
        return list(it)

    chunked = MapIterable(_chunk_input, ds)
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


class NumpyTokenizer:
    """Generate numpy arrays of tokenized data."""

    def __init__(self, enc: Tokenizer) -> None:
        """Initialize the tokenizer."""
        self._enc = enc

    def encode(self, text: str) -> np.ndarray:
        """Encode the text."""
        tokens = self._enc.encode(text)
        tokens_np = np.array(tokens)
        if not (0 <= tokens_np).all() and (tokens_np < 2**16).all():
            raise ValueError("token dictionary too large for uint16")
        return tokens_np.astype(np.uint16)


class TokenizedFileWriter:
    """A file writer that writes tokenized files."""

    def __init__(self, path: pathlib.Path, max_tokens: int) -> None:
        """Initialize the file writer."""
        self._path = path
        self._tokens: np.ndarray = np.empty((max_tokens,), dtype=np.uint16)
        self._num_tokens = 0

    @property
    def num_tokens(self) -> int:
        """Return the number of tokens."""
        return self._num_tokens

    def append(self, tokens: np.ndarray) -> None:
        """Write the tokens to a file."""
        self._tokens[self._num_tokens : self._num_tokens + len(tokens)] = tokens
        self._num_tokens += len(tokens)

    def write(self) -> None:
        """Save the tokens to a file."""
        _LOGGER.debug("Writing %d tokens to %s", len(self._tokens), self._path)
        tokens_np = self._tokens[: self.num_tokens]
        np.save(self._path, tokens_np)
        self._tokens = np.zeros(1, dtype=np.uint16)


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
        self._writer: TokenizedFileWriter | None = TokenizedFileWriter(
            shard_filepath, self._tokens_per_shard
        )
        self._pbar = tqdm.tqdm(
            desc=f"Shard {self._shard}",
            total=self._tokens_per_shard,
        )

    def append(self, tokens: np.ndarray) -> None:
        """Write the tokens to a file."""
        if self._writer is None:
            raise ValueError("Writer is not initialized")
        if self._writer.num_tokens + len(tokens) > self._tokens_per_shard:
            if self._writer.num_tokens == 0:
                raise ValueError(
                    f"Too large for shard size: {len(tokens)} > {self._tokens_per_shard}"
                )
            self._writer.write()
            self._open_new_writer()
        self._writer.append(tokens)
        self._pbar.update(len(tokens))

    def write(self) -> None:
        """Save the tokens to a file."""
        if self._writer is None:
            raise ValueError("Writer is not initialized")
        self._writer.write()
        self._writer = None


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
    writer = ShardedTokenizedFileWriter(output_path, tokens_per_shard)
    tokenizer = NumpyTokenizer(enc)
    with multiprocessing.Pool(num_procs) as pool:
        for tokens in pool.imap(
            tokenizer.encode, text_ds, chunksize=PROCESS_CHUNK_SIZE
        ):
            writer.append(tokens)
    writer.write()


def read_tokenized_file(path: pathlib.Path) -> torch.Tensor:
    """Read a tokenized file."""
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    tokens = np.load(path)
    _LOGGER.debug("Loaded %s with %d tokens", path, len(tokens))
    tokens_np = tokens.astype(np.int32)
    return torch.tensor(tokens_np, dtype=torch.long)


class ShardedTokenizedFileReader:
    """A file reader for a set of sharded files identified by a glob filename."""

    def __init__(self, path: pathlib.Path) -> None:
        """Initialize the file reader."""
        self._path = path
        self._files = sorted(list(path.parent.glob(f"{path.name}.*")))
        if len(self._files) == 0:
            raise ValueError(f"No files found matching path: {path}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Return the iterator."""
        return iter(self._iter())

    def _iter(self) -> Generator[torch.Tensor]:
        """Iterate through the sharded files."""
        _LOGGER.debug(
            "Reading %d files matching path: %s", len(self._files), self._path
        )
        idx = 0
        while idx < len(self._files):
            yield read_tokenized_file(self._files[idx])
            idx += 1


def read_preprocessed_corpus(
    token_path: pathlib.Path,
    config: DatasetConfig,
    worker_num: int = 0,
    worker_count: int = 1,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Read the preprocessed corpus.

    If worker_num and worker_count are provided, it will read only the
    specified worker's portion of the data.
    """

    class Iter(Iterable[tuple[torch.Tensor, torch.Tensor]]):

        def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            """Return the iterator."""
            _LOGGER.debug("Reading preprocessed corpus from %s", token_path)
            while True:
                samples = 0
                for tokens in ShardedTokenizedFileReader(token_path):
                    for chunk in ChunkingIterable(
                        config, tokens, worker_num, worker_count
                    ):
                        yield chunk
                        samples += 1
                _LOGGER.info("Reached epoch")
                if samples == 0:
                    raise ValueError("Empty dataset or dataset could not be restarted")

    return Iter()

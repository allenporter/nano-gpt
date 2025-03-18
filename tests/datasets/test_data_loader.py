"""Tests for the data loader."""

from collections.abc import Generator
import itertools
import pathlib
import tempfile

import pytest
import datasets
import numpy as np

from nano_gpt.datasets.data_loader import (
    chunk_dataset,
    tokenize_dataset,
    cycle_dataset,
    read_preprocessed_corpus,
    preprocess_corpus,
    MapIterable,
    ShardedTokenizedFileWriter,
)
from nano_gpt.config import DatasetConfig
from nano_gpt.tokenizer import Tokenizer

import torch


def test_tokenize_dataset(fake_tokenizer: Tokenizer) -> None:
    """Test tokenize_dataset."""
    ds = ["this is test data"]
    tokenized = tokenize_dataset(fake_tokenizer, ds)
    values = list(tokenized)
    assert len(values) == 1
    tokens = values[0][:5].tolist()
    assert fake_tokenizer.decode(tokens) == "this "

    # Verify that the returned iterator can be restarted
    values2 = list(tokenized)
    assert len(values2) == 1
    assert values[0].tolist() == values2[0].tolist()


def test_chunk_dataset(fake_tokenizer: Tokenizer) -> None:
    """Test chunk_dataset."""
    config = DatasetConfig(micro_batch_size=2, sequence_length=2)
    tokens = fake_tokenizer.encode("this is test data!")
    t = torch.tensor(tokens)
    chunks_iter = chunk_dataset(config, [t])
    pairs = [
        ([fake_tokenizer.decode(x) for x in xs], [fake_tokenizer.decode(y) for y in ys])
        for xs, ys in chunks_iter
    ]
    assert pairs == [
        (["th", "is"], ["hi", "s "]),
        ([" i", "s "], ["is", " t"]),
        (["te", "st"], ["es", "t "]),  # codespell:ignore
        ([" d", "at"], ["da", "ta"]),
    ]


def test_chunk_dataset_worker_split(fake_tokenizer: Tokenizer) -> None:
    """Test chunk_dataset split by workers."""
    config = DatasetConfig(micro_batch_size=4, sequence_length=10)
    assert config.chunk_token_size == 40
    t = torch.tensor(
        [1] * 40
        + [2] * 40
        + [3] * 40
        + [4] * 40
        + [5] * 40
        + [6] * 40
        + [7] * 40
        + [8] * 40
        + [9] * 40
        + [10] * 40,
    )
    # 3 workers chunk up the dataset
    pairs1 = list(chunk_dataset(config, [t], worker_num=0, worker_count=3))
    pairs2 = list(chunk_dataset(config, [t], worker_num=1, worker_count=3))
    pairs3 = list(chunk_dataset(config, [t], worker_num=2, worker_count=3))
    # Drop the y values
    x1 = [pair[0] for pair in pairs1]
    x2 = [pair[0] for pair in pairs2]
    x3 = [pair[0] for pair in pairs3]
    # Flatten the output and verify the results
    assert len(x1) == 3
    assert x1[0].shape == (4, 10)
    assert (x1[0].numpy() == np.array([[1] * 10] * 4)).all()
    assert (x1[1].numpy() == np.array([[4] * 10] * 4)).all()
    assert (x1[2].numpy() == np.array([[7] * 10] * 4)).all()
    assert len(x2) == 3
    assert x2[0].shape == (4, 10)
    assert (x2[0].numpy() == np.array([[2] * 10] * 4)).all()
    assert (x2[1].numpy() == np.array([[5] * 10] * 4)).all()
    assert (x2[2].numpy() == np.array([[8] * 10] * 4)).all()
    assert len(x3) == 3
    assert x3[0].shape == (4, 10)
    assert (x3[0].numpy() == np.array([[3] * 10] * 4)).all()
    assert (x3[1].numpy() == np.array([[6] * 10] * 4)).all()
    assert (x3[2].numpy() == np.array([[9] * 10] * 4)).all()


def test_cycle_dataset(fake_tokenizer: Tokenizer) -> None:
    """Test cycle_dataset."""
    data = [
        torch.tensor(1),
        torch.tensor(2),
        torch.tensor(3),
    ]
    ds = cycle_dataset(
        MapIterable(lambda x: x, data),
    )
    result = list(itertools.islice(ds, 8))
    assert result == [
        torch.tensor(1),
        torch.tensor(2),
        torch.tensor(3),
        torch.tensor(1),
        torch.tensor(2),
        torch.tensor(3),
        torch.tensor(1),
        torch.tensor(2),
    ]


@pytest.fixture
def tmpdir() -> Generator[pathlib.Path]:
    """Fixture for temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


def test_preprocess_corpus(fake_tokenizer: Tokenizer, tmpdir: pathlib.Path) -> None:
    """Test preprocess_corpus."""

    tmp_path = pathlib.Path(tmpdir) / "finewebedu.npy"
    preprocess_corpus(
        datasets.Dataset.from_dict({"text": ["this is test data", "record 2"]}),
        fake_tokenizer,
        tmp_path,
        num_procs=2,
    )
    assert len(list(tmpdir.glob("*.npy.*"))) == 1
    ds = read_preprocessed_corpus(
        tmp_path,
        DatasetConfig(micro_batch_size=2, sequence_length=2),
    )
    limited_iter = itertools.islice(ds, 14)

    pairs = [
        ([fake_tokenizer.decode(x) for x in xs], [fake_tokenizer.decode(y) for y in ys])
        for xs, ys in limited_iter
    ]
    assert pairs == [
        (["th", "is"], ["hi", "s "]),
        ([" i", "s "], ["is", " t"]),
        (["te", "st"], ["es", "t "]),  # codespell:ignore
        ([" d", "at"], ["da", "ta"]),
        (["ar", "ec"], ["re", "co"]),
        (["or", "d "], ["rd", " 2"]),
        (["th", "is"], ["hi", "s "]),
        ([" i", "s "], ["is", " t"]),
        (["te", "st"], ["es", "t "]),  # codespell:ignore
        ([" d", "at"], ["da", "ta"]),
        (["ar", "ec"], ["re", "co"]),
        (["or", "d "], ["rd", " 2"]),
        (["th", "is"], ["hi", "s "]),
        ([" i", "s "], ["is", " t"]),
    ]


@pytest.mark.parametrize(
    ("total_items", "tokens_per_item", "tokens_per_shard", "expected_num_files"),
    [
        (10, 3, 8, 5),  # 30 tokens with 6 tokens per file is 5 files
        (10, 1, 8, 2),  # 10 tokens with 8 tokens per file is 2 files
        (16, 1, 8, 2),  # 16 tokens with 8 tokens per file is 2 files
        (17, 1, 8, 3),  # 17 tokens with 8 tokens per file is 3 files
        (20, 2, 8, 5),  # 40 tokens with 8 tokens per file is 5 files
    ],
)
def test_sharded_file_writer(
    tmpdir: pathlib.Path,
    total_items: int,
    tokens_per_item: int,
    tokens_per_shard: int,
    expected_num_files: int,
) -> None:
    """Test ShardedFileWriter."""

    writer = ShardedTokenizedFileWriter(
        tmpdir / "test", tokens_per_shard=tokens_per_shard
    )
    for i in range(total_items):
        toks = np.array([i] * tokens_per_item)
        writer.append(toks)
    writer.write()

    files = list(tmpdir.glob("test.*"))
    assert len(files) == expected_num_files

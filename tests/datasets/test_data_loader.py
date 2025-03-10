"""Tests for the data loader."""

import itertools

from nano_gpt.datasets.data_loader import (
    chunk_dataset,
    tokenize_dataset,
    cycle_dataset,
    preprocess_dataset,
    MapIterable,
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
    tokens = fake_tokenizer.encode("this is test data")
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
    ]


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


def test_preoprocess_dataset(fake_tokenizer: Tokenizer) -> None:
    """Test preprocess_dataset."""

    ds = preprocess_dataset(
        ["this is test data"],
        fake_tokenizer,
        DatasetConfig(micro_batch_size=2, sequence_length=2),
    )
    limited_iter = itertools.islice(ds, 10)

    pairs = [
        ([fake_tokenizer.decode(x) for x in xs], [fake_tokenizer.decode(y) for y in ys])
        for xs, ys in limited_iter
    ]
    assert pairs == [
        (["th", "is"], ["hi", "s "]),
        ([" i", "s "], ["is", " t"]),
        (["te", "st"], ["es", "t "]),  # codespell:ignore
        (["th", "is"], ["hi", "s "]),
        ([" i", "s "], ["is", " t"]),
        (["te", "st"], ["es", "t "]),  # codespell:ignore
        (["th", "is"], ["hi", "s "]),
        ([" i", "s "], ["is", " t"]),
        (["te", "st"], ["es", "t "]),  # codespell:ignore
        (["th", "is"], ["hi", "s "]),
    ]

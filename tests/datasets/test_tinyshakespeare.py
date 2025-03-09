"""Tests for the data loader."""

from nano_gpt.datasets.tinyshakespeare import get_data_loader
from nano_gpt.config import DatasetConfig
from nano_gpt.tokenizer import Tokenizer


def test_get_data_loader(fake_tokenizer: Tokenizer) -> None:
    """Test that we can get the data loader."""

    config = DatasetConfig(micro_batch_size=2, sequence_length=4)
    data_loader = get_data_loader(fake_tokenizer, config)

    assert data_loader is not None
    assert hasattr(data_loader, "__iter__")
    assert hasattr(data_loader, "__next__")

    item = next(data_loader)
    assert isinstance(item, tuple)
    assert len(item) == 2
    x, y = item
    assert x.shape == (config.micro_batch_size, config.sequence_length)
    assert y.shape == (config.micro_batch_size, config.sequence_length)

    assert [fake_tokenizer.decode(c) for c in x] == [
        "this",  # codespell:ignore
        " is ",
    ]
    assert [fake_tokenizer.decode(c) for c in y] == [
        "his ",
        "is t",
    ]


def test_reached_epoch(fake_tokenizer: Tokenizer) -> None:
    """Test that we can get the data loader."""

    config = DatasetConfig(micro_batch_size=1, sequence_length=4)
    data_loader = get_data_loader(fake_tokenizer, config)

    ds = iter(data_loader)
    batches = [next(ds) for _ in range(6)]
    # flatten the list of batches (micro_batch_size=1)
    pairs = [(x[0], y[0]) for x, y in batches]
    assert [
        (fake_tokenizer.decode(x.tolist()), fake_tokenizer.decode(y.tolist()))
        for x, y in pairs
    ] == [
        ("this", "his "),
        (" is ", "is t"),
        ("test", "est "),
        (" dat", "data"),
        ("this", "his "),
        (" is ", "is t"),
    ]

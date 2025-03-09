"""Tests for the data loader."""

from nano_gpt.data import get_data_loader
from nano_gpt.config import config_from
from nano_gpt.tokenizer import Tokenizer


def test_get_data_loader(fake_tokenizer: Tokenizer) -> None:
    """Test that we can get the data loader."""

    train_config = config_from("gpt2", batch_size=2, sequence_length=4).train_config
    data_loader = get_data_loader(fake_tokenizer, train_config, device="cpu")

    assert data_loader is not None
    assert hasattr(data_loader, "__iter__")
    assert hasattr(data_loader, "__next__")

    item = next(data_loader)
    assert isinstance(item, tuple)
    assert len(item) == 2
    x, y = item
    assert x.shape == (train_config.batch_size, train_config.sequence_length)
    assert y.shape == (train_config.batch_size, train_config.sequence_length)

    assert [fake_tokenizer.decode(c) for c in x] == [
        "this",  # codespell:ignore
        " is ",
    ]
    assert [fake_tokenizer.decode(c) for c in y] == [
        "his ",
        "is t",
    ]

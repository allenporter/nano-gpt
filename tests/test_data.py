"""Tests for the data loader."""

from nano_gpt.data import get_data_loader
from nano_gpt.tokenizer import Tokenizer


def test_get_data_loader(fake_tokenizer: Tokenizer) -> None:
    """Test that we can get the data loader."""

    batch_size = 2
    token_len = 3
    device = "cpu"

    data_loader = get_data_loader(fake_tokenizer, batch_size, token_len, device)

    assert data_loader is not None
    assert hasattr(data_loader, "__iter__")
    assert hasattr(data_loader, "__next__")

    item = next(data_loader)
    assert isinstance(item, tuple)
    assert len(item) == 2
    x, y = item
    assert [fake_tokenizer.decode(c) for c in x] == [
        "thi",  # codespell:ignore
        "s i",
    ]
    assert [fake_tokenizer.decode(c) for c in y] == [
        "his",
        " is",
    ]

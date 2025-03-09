"""Tests for the GPT-2 model architecture."""

from nano_gpt.model import GPT
from nano_gpt.config import GPTConfig
from nano_gpt.data import get_data_loader
from nano_gpt.tokenizer import Tokenizer


def test_block_size(fake_tokenizer: Tokenizer) -> None:
    """Test that the block size is correct."""

    vocab_size = 256
    config = GPTConfig(
        block_size=8,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )

    model = GPT(config, tokenizer=fake_tokenizer)

    batch_size = 2
    token_len = 3
    data_loader = get_data_loader(
        fake_tokenizer, batch_size=batch_size, token_len=token_len, device="cput"
    )
    ds = iter(data_loader)
    x, y = next(ds)
    assert x.shape == (batch_size, token_len)
    assert y.shape == (batch_size, token_len)
    logits, loss = model(x, y)
    assert logits.shape == (batch_size, token_len, vocab_size)
    assert isinstance(loss.item(), float)

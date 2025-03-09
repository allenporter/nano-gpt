"""Tests for the GPT-2 model architecture."""

from nano_gpt.model import GPT
from nano_gpt.config import GPTConfig, config_from
from nano_gpt.datasets.tinyshakespeare import get_data_loader
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

    train_config = config_from(
        "gpt2", micro_batch_size=2, sequence_length=4
    ).train_config
    data_loader = get_data_loader(
        fake_tokenizer,
        train_config.dataset_config,
        device="cpu",
    )
    ds = iter(data_loader)
    x, y = next(ds)
    assert x.shape == (train_config.micro_batch_size, train_config.sequence_length)
    assert y.shape == (train_config.micro_batch_size, train_config.sequence_length)
    logits, loss = model(x, y)
    assert logits.shape == (
        train_config.micro_batch_size,
        train_config.sequence_length,
        vocab_size,
    )
    assert isinstance(loss.item(), float)

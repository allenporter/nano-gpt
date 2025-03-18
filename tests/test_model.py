"""Tests for the GPT-2 model architecture."""

from nano_gpt.model import GPT
from nano_gpt.config import GPTConfig, DatasetConfig
from nano_gpt.datasets.data_loader import preprocess_dataset
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

    dataset_config = DatasetConfig(micro_batch_size=2, sequence_length=4)
    data_loader = preprocess_dataset(
        ["this is test data"],
        fake_tokenizer,
        dataset_config,
    )
    ds = iter(data_loader)
    x, y = next(ds)
    assert x.shape == (dataset_config.micro_batch_size, dataset_config.sequence_length)
    assert y.shape == (dataset_config.micro_batch_size, dataset_config.sequence_length)
    logits, loss = model(x, y)
    assert logits.shape == (
        dataset_config.micro_batch_size,
        dataset_config.sequence_length,
        vocab_size,
    )
    assert isinstance(loss.item(), float)

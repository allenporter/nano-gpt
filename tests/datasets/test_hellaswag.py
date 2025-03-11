"""Tests for the hellaswag dataset."""

from nano_gpt.datasets.hellaswag import Sample
from nano_gpt.tokenizer import Tokenizer


def test_tokenize_sample(fake_tokenizer: Tokenizer) -> None:
    """Test the Sample class."""
    sample = Sample(
        prefix="red fish,",
        endings=[
            "green fish",
            "blue fish",
            "yellow fish",
            "purple fish",
        ],
        label=1,
    )
    tokens, mask = sample.tokenize(fake_tokenizer)
    assert tokens.shape == (4, 21)
    assert fake_tokenizer.decode(tokens[0].tolist()) == "red fish, green fish\x00"
    assert mask.shape == (4, 21)
    assert mask[0].tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ]

"""Fixtures for nano_gpt tests."""

from unittest.mock import patch
from collections.abc import Sequence
from typing import Generator

import pytest

from nano_gpt.tokenizer import Tokenizer


class FakeTokenizer(Tokenizer):
    """Fake tokenizer for testing."""

    def encode(self, text: str) -> list[int]:
        """Encode the text into tokens."""
        return [ord(c) for c in text]

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the tokens into text."""
        return "".join([chr(t) for t in tokens])


@pytest.fixture
def fake_tokenizer() -> Tokenizer:
    """Fixture to prepare a fake tokenizer for testing."""
    return FakeTokenizer()


@pytest.fixture(autouse=True)
def fake_dataset() -> Generator:
    """Fixture to prepare a fake dataset for testing."""
    with patch(
        "nano_gpt.datasets.tinyshakespeare.datasets.load_dataset"
    ) as mock_load_dataset:
        mock_load_dataset.return_value = {"train": {"text": ["this is test data"]}}
        yield

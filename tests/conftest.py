"""Fixtures for nano_gpt tests."""

from collections.abc import Sequence
import os
import tempfile

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
def huggingface_cache_fixture() -> None:
    """Fixture to prepare a cache directory for huggingface."""

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["HF_HOME"] = tmpdir

"""Tests for tokenizer module."""

from unittest.mock import Mock, patch

from nano_gpt.tokenizer import DocumentTokenizer


def test_document_tokenzer() -> None:
    """Test the fake tokenizer."""

    fake_tokenizer = Mock()

    with patch("nano_gpt.tokenizer.tiktoken.get_encoding", return_value=fake_tokenizer):
        fake_tokenizer.encode_ordinary.return_value = [ord(c) for c in "hello"]
        fake_tokenizer._special_tokens = {"<|endoftext|>": "EOT"}

        tokenizer = DocumentTokenizer("fake-encoding")

        text = "hello"
        tokens = tokenizer.encode(text)
        assert tokens == ["EOT"] + [ord(c) for c in "hello"]

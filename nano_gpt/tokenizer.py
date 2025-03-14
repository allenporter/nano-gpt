"""Tokenizer for GPT-2.

This is a thin wrapper around the tiktoken library, allowing for easy
unit testing.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import tiktoken

__all__ = [
    "Tokenizer",
    "get_tokenizer",
]


class Tokenizer(ABC):
    """Abstract base class for tokenizers.

    This is a thin wrapper around tokenizer libraries and supports encode
    and decode.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode the text into tokens."""

    @abstractmethod
    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the tokens into text."""


class TiktokenTokenizer(Tokenizer):
    """Tokenizer for GPT-2 using tiktoken."""

    def __init__(self, encoding_name: str) -> None:
        """Initialize the tokenizer."""
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        """Encode the text into tokens."""
        return self.encoding.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the tokens into text."""
        return self.encoding.decode(tokens)


class DocumentTokenizer(Tokenizer):
    """Tokenizer for GPT-2 using tiktoken that encodes documents."""

    def __init__(self, encoding_name: str) -> None:
        """Initialize the tokenizer."""
        self.encoding = tiktoken.get_encoding(encoding_name)
        self._eot = self.encoding._special_tokens["<|endoftext|>"]

    def encode(self, text: str) -> list[int]:
        """Encode the text into tokens."""
        tokens = [self._eot]
        # Ignore any special tokesn in the document
        tokens.extend(self.encoding.encode_ordinary(text))
        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the tokens into text."""
        raise NotImplementedError("DocumentTokenizer does not support decoding")


def get_tokenizer() -> Tokenizer:
    """Get the GPT-2 tokenizer."""
    return TiktokenTokenizer("gpt2")


def get_document_tokenizer() -> Tokenizer:
    """Get the GPT-2 tokenizer for encoding documents for pre-training (only)."""
    return DocumentTokenizer("gpt2")

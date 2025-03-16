"""Tests for the tinyshakespeare dataset."""

from nano_gpt.datasets.tinyshakespeare import load_dataset


def test_load_dataset() -> None:
    """Test loading the dataset."""

    ds = load_dataset("train")
    example = next(iter(map(lambda x: x["text"], ds)))
    assert example[0:15] == "First Citizen:\n"

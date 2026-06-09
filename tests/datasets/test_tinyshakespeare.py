"""Tests for the tinyshakespeare dataset."""

from unittest.mock import patch
from nano_gpt.datasets.tinyshakespeare import load_dataset


def test_load_dataset() -> None:
    """Test loading the dataset."""

    mock_dataset = [
        {"text": "First Citizen:\nThis is a mock tiny shakespeare dataset."}
    ]
    with patch("datasets.load_dataset", return_value=mock_dataset) as mock_load:
        ds = load_dataset("train")
        mock_load.assert_called_once_with(
            "tiny_shakespeare", trust_remote_code=True, split="train"
        )
        example = next(iter(map(lambda x: x["text"], ds)))
        assert example[0:15] == "First Citizen:\n"

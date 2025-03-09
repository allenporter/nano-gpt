"""Tests for the training module."""

from unittest.mock import patch

from nano_gpt.trainer import TrainStats
from nano_gpt.config import config_from


def test_train_stats() -> None:
    """Test the TrainStats class."""

    config = config_from("gpt2", micro_batch_size=4, sequence_length=256).train_config
    assert config.dataset_config.chunk_token_size == 1024  # Used in tok/sec below

    stats = TrainStats(config)
    assert stats.step == 0

    t0 = 1741493238.878981
    with patch("nano_gpt.trainer.time.time", return_value=t0):
        stats.start_step()
        assert stats.step == 0

    t1 = t0 + 0.100
    with patch("nano_gpt.trainer.time.time", return_value=t1):
        stats.end_step(loss=1.0, norm=1.0)

    assert stats.step == 1
    assert (
        str(stats)
        == "step: 0 | loss: 1.0000 | norm: 1.0000 | dt: 100.00ms | tok/sec: 5242885.00"
    )

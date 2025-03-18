"""Tests for the checkpointing functionality."""

import pathlib
import tempfile
import pytest

import torch

from nano_gpt.checkpoint import Checkpoint, save_checkpoint, load_checkpoint
from nano_gpt.config import GPTConfig, TrainConfig, DatasetConfig, EvalConfig
from nano_gpt.model import GPT
from nano_gpt.tokenizer import Tokenizer


@pytest.fixture
def minimal_model(fake_tokenizer: Tokenizer) -> GPT:
    """Create a minimal model for testing."""
    config = GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=1,
        n_head=1,
        n_embd=8,
    )
    return GPT(config, tokenizer=fake_tokenizer)


@pytest.fixture
def checkpoint_data(minimal_model: GPT) -> Checkpoint:
    """Create test checkpoint data."""
    return Checkpoint(
        model_state_dict=minimal_model.state_dict(),
        config=minimal_model.config,
        step=100,
        val_loss_accum=0.5,
        optimizer_state_dict={"state": {}, "param_groups": []},
        train_config=TrainConfig(
            total_batch_size=512,
            micro_batch_size=16,
            sequence_length=32,
        ),
        dataset_config=DatasetConfig(),
        eval_config=EvalConfig(),
    )


def test_save_and_load_checkpoint(checkpoint_data: Checkpoint) -> None:
    """Test saving and loading a checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = pathlib.Path(tmpdir) / "test_checkpoint.pt"

        # Save checkpoint
        save_checkpoint(checkpoint_data, checkpoint_path)
        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_checkpoint = load_checkpoint(checkpoint_path)

        # Verify all fields match
        assert loaded_checkpoint.step == checkpoint_data.step
        assert loaded_checkpoint.val_loss_accum
        assert checkpoint_data.val_loss_accum
        assert loaded_checkpoint.val_loss_accum == checkpoint_data.val_loss_accum
        assert loaded_checkpoint.config == checkpoint_data.config
        assert loaded_checkpoint.train_config == checkpoint_data.train_config
        assert loaded_checkpoint.dataset_config == checkpoint_data.dataset_config
        assert loaded_checkpoint.eval_config == checkpoint_data.eval_config

        # Verify model state dict keys match
        assert set(loaded_checkpoint.model_state_dict.keys()) == set(
            checkpoint_data.model_state_dict.keys()
        )

        # Verify optimizer state dict matches
        assert (
            loaded_checkpoint.optimizer_state_dict
            == checkpoint_data.optimizer_state_dict
        )


def test_save_and_load_model_state(
    minimal_model: GPT, fake_tokenizer: Tokenizer
) -> None:
    """Test that model state is preserved when saving and loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = pathlib.Path(tmpdir) / "test_checkpoint.pt"

        # Create a checkpoint with the model state
        checkpoint = Checkpoint(
            model_state_dict=minimal_model.state_dict(),
            config=minimal_model.config,
            step=100,
            train_config=TrainConfig(
                total_batch_size=512,
                micro_batch_size=16,
                sequence_length=32,
            ),
            dataset_config=DatasetConfig(),
            eval_config=EvalConfig(),
        )

        # Save checkpoint
        save_checkpoint(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded_checkpoint = load_checkpoint(checkpoint_path)

        # Create a new model with the same config
        new_model = GPT(loaded_checkpoint.config, tokenizer=fake_tokenizer)

        # Load the state dict into the new model
        new_model.load_state_dict(loaded_checkpoint.model_state_dict)

        # Verify the models have the same parameters
        for p1, p2 in zip(minimal_model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

"""Configuration module."""

import dataclasses
from dataclasses import dataclass
import enum
import logging
import pathlib
from typing import Protocol

import datasets

__all__ = [
    "GPTConfig",
    "DatasetConfig",
    "TrainConfig",
    "TrainedModelConfig",
    "Models",
    "config_from",
    "model_config_from_pretrained",
    "TrainDataset",
]

_LOGGER = logging.getLogger(__name__)

VOCAB_SIZE = 50257  # Fixed size for GPT model checkpoints
NICE_VOCAB_SIZE = 50304  # Vocab size with nice power of 2, for training
BLOCK_SIZE = 1024  # Fixed size for GPT model checkpoints
DEFAULT_MICRO_BATCH_SIZE = 16


@dataclass(frozen=True, kw_only=True)
class GPTConfig:
    """This class defines the configuration for the GPT model.

    This configuration is used for inference.
    """

    block_size: int = BLOCK_SIZE
    """The maximum context length."""

    vocab_size: int = VOCAB_SIZE
    """The size of the vocabulary."""

    n_layer: int = 12
    """The number of transformer blocks."""

    n_head: int = 12
    """The number of attention heads."""

    n_embd: int = 768
    """The size of the embedding vector."""


@dataclass(frozen=True, kw_only=True)
class DatasetConfig:
    """This class defines the configuration for chunking the dataset."""

    dataset_dir: str = "dataset_cache"
    """The directory where the dataset is stored."""

    dataset_name: str = "tinyshakespeare"
    """The name of the dataset."""

    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    """Batch size (micro batch) (B) used for each forward/backward pass."""

    sequence_length: int = BLOCK_SIZE
    """Sequence length (T) used for input content. Same as block_size."""

    @property
    def chunk_token_size(self) -> int:
        """Number of tokens in each micro batch."""
        return self.micro_batch_size * self.sequence_length

    def dataset_path(self, split: str) -> pathlib.Path:
        """Return the path to the dataset."""
        dataset_dir = pathlib.Path(self.dataset_dir)
        return dataset_dir / f"{self.dataset_name}_{split}.npy"


@dataclass(frozen=True, kw_only=True)
class SampleConfig:
    """This class defines the configuration for sampling the dataset."""

    num_return_sequences: int = 5
    """The number of sequences to generate."""

    max_length: int = 30
    """The maximum length of the generated sequences."""

    text: str = "Hello, I'm a language model,"
    """The text to use as a prompt for sampling."""

    seed: int = 42
    """The seed to use for sampling."""


@dataclass(frozen=True, kw_only=True)
class EvalConfig:
    """This class defines the configuration for the validation loss and HellaSwag eval."""

    validation_steps: int = 20
    """Number of validation loss iterations to perform each eval round."""

    hellaswag_samples: int | None = None
    """The number of HellaSwag evaluation results to sample or None for the entire set."""


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Implementats the GPT-3 learning rate."""

    total_batch_size: int
    """Total batch size in number of tokens for each gradient update.

    If this is larger than B * T, then the batch size is divided into
    micro-batches of size B * T as part of gradient accumulation.
    """

    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    """Batch size (micro batch) (B) used for each forward/backward pass."""

    sequence_length: int = BLOCK_SIZE
    """Sequence length (T) used for input content. Same as block_size."""

    max_lr: float = 6e-4
    """Maximum learning rate."""

    min_lr_ratio: float = 0.1
    """Minimum learning rate ratio in terms of the max learning rate."""

    # 2**19 tokens per step
    # 10e9 - 10 billion tokens / 2**19 = 19073
    # warmup over 375 million tokens from GPT2 papager.
    # 375e6 / 2**19 = 715 steps
    # The warmup is very mild and could be made more aggressive

    warmup_steps: int = 715
    """Number of warmup steps before getting to the max learning rate."""

    max_steps: int = 19073
    """Total number of training steps to perform."""

    eval_steps: int = 250
    """Number of steps between each evaluation of validation loss."""

    def __post_init__(self) -> None:
        """Post init."""
        if self.total_batch_size % self.chunk_token_size != 0:
            raise ValueError(
                "Total batch size must be divisible by B * T"
                f" but got {self.total_batch_size} % {self.chunk_token_size}"
            )

    @property
    def chunk_token_size(self) -> int:
        """Number of tokens in each micro batch."""
        return self.micro_batch_size * self.sequence_length

    @property
    def min_lr(self) -> float:
        """Minimum learning rate."""
        return self.max_lr * self.min_lr_ratio

    @property
    def grad_accum_steps(self) -> int:
        """Number of gradient accumulation steps."""
        return self.total_batch_size // self.chunk_token_size

    def log_info(self) -> None:
        """String representation."""
        _LOGGER.info("Token batch size: %s", self.micro_batch_size)
        _LOGGER.info("Sequence length: %s", self.sequence_length)
        _LOGGER.info("Total token batch size: %s", self.total_batch_size)
        _LOGGER.info("Gradient accumulation steps: %s", self.grad_accum_steps)


@dataclass(frozen=True)
class TrainedModelConfig:
    """This class defines the configuration for the GPT model."""

    model_name: str
    """The name of the model."""

    model_config: GPTConfig
    """The configuration for the model."""

    train_config: TrainConfig
    """The configuration for the training."""


class Models(enum.Enum):
    """This class defines the configuration for the GPT model."""

    GPT2_SMALL = TrainedModelConfig(
        "gpt2",  # 124M params
        GPTConfig(n_layer=12, n_head=12, n_embd=768),
        TrainConfig(
            total_batch_size=2**19,  # ~0.5M, in number of tokens
            max_lr=6e-4,
        ),
    )
    GPT2_MEDIUM = TrainedModelConfig(
        "gpt2-medium",  # 350M params
        GPTConfig(n_layer=24, n_head=16, n_embd=1024),
        TrainConfig(
            total_batch_size=2**19,  # ~0.5M, in number of tokens
            max_lr=3e-4,
        ),
    )
    GPT2_LARGE = TrainedModelConfig(
        "gpt2-large",  # 774M params
        GPTConfig(n_layer=36, n_head=20, n_embd=1280),
        TrainConfig(
            total_batch_size=2**19,  # ~0.5M, in number of tokens
            max_lr=2.5e-4,
        ),
    )
    GPT2_XL = TrainedModelConfig(
        "gpt2-xl",  # 1558M params
        GPTConfig(n_layer=48, n_head=25, n_embd=1600),
        TrainConfig(
            total_batch_size=2**20,  #  ~1M, in number of tokens
            max_lr=2e-4,
        ),
    )

    # These are model sizes that were made up for this project
    GPT2_XS = TrainedModelConfig(
        "gpt2-xs",  # 58M params
        GPTConfig(n_layer=10, n_head=10, n_embd=512),
        TrainConfig(
            total_batch_size=2**18,  # ~0.25M, in number of tokens
            max_lr=3e-4,
        ),
    )
    GPT2_XXS = TrainedModelConfig(
        "gpt2-xxs",  # ~3M params
        GPTConfig(n_layer=4, n_head=4, n_embd=64),
        TrainConfig(
            total_batch_size=2**16,  # ~0.065M, in number of tokens
            max_lr=3e-4,
        ),
    )


PRETRAINED = {
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
}
MODELS = {model.value.model_name: model.value for model in Models}


def config_from(
    model_type: str,
    micro_batch_size: int | None = None,
    sequence_length: int | None = None,
    total_batch_size: int | None = None,
    max_steps: int | None = None,
    eval_steps: int | None = None,
) -> TrainedModelConfig:
    """Return the configuration for the model."""
    if (config := MODELS.get(model_type)) is None:
        raise ValueError(f"Unknown model type: {model_type}")
    model_config_updates = {}
    train_config_updates = {}
    if micro_batch_size is not None:
        train_config_updates["micro_batch_size"] = micro_batch_size
    if sequence_length is not None:
        train_config_updates["sequence_length"] = sequence_length
        model_config_updates["block_size"] = sequence_length
    if total_batch_size is not None:
        train_config_updates["total_batch_size"] = total_batch_size
    if max_steps is not None:
        train_config_updates["max_steps"] = max_steps
    if eval_steps is not None:
        train_config_updates["eval_steps"] = eval_steps
    return TrainedModelConfig(
        model_name=config.model_name,
        model_config=dataclasses.replace(
            config.model_config,
            **model_config_updates,
        ),
        train_config=dataclasses.replace(
            config.train_config,
            **train_config_updates,
        ),
    )


def model_config_from_pretrained(model_type: str) -> GPTConfig:
    """Return the configuration for the pretrained model."""
    if model_type not in PRETRAINED:
        raise ValueError(f"Unknown model type: {model_type}")
    config = config_from(model_type)
    return config.model_config


class LoadDataset(Protocol):
    """A protocol for loading a dataset."""

    def __call__(self, split: str, streaming: bool = False) -> datasets.Dataset:
        """Load a dataset."""


@dataclass
class TrainDataset:
    """A dataset."""

    name: str
    """The name of the dataset."""

    load_fn: LoadDataset
    """The function to load the dataset."""

    total_tokens: int
    """The total number of tokens in the dataset."""

    tokens_per_shard: int
    """The number of tokens per shard."""

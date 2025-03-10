"""Configuration module."""

import dataclasses
from dataclasses import dataclass
import enum
import logging

__all__ = [
    "GPTConfig",
    "DatasetConfig",
    "TrainConfig",
    "TrainedModelConfig",
    "Models",
    "config_from",
    "model_config_from_pretrained",
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

    micro_batch_size: int
    """Batch size (micro batch) (B) used for each forward/backward pass."""

    sequence_length: int
    """Sequence length (T) used for input content. Same as block_size."""

    @property
    def chunk_token_size(self) -> int:
        """Number of tokens in each micro batch."""
        return self.micro_batch_size * self.sequence_length


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

    warmup_steps: int = 10
    """Number of warmup steps before getting to the max learning rate."""

    max_steps: int = 50
    """Total number of training steps to perform."""

    def __post_init__(self) -> None:
        """Post init."""
        if self.total_batch_size % self.dataset_config.chunk_token_size != 0:
            raise ValueError(
                "Total batch size must be divisible by B * T"
                f" but got {self.total_batch_size} % {self.dataset_config.chunk_token_size}"
            )

    @property
    def dataset_config(self) -> DatasetConfig:
        """Dataset configuration."""
        return DatasetConfig(
            micro_batch_size=self.micro_batch_size,
            sequence_length=self.sequence_length,
        )

    @property
    def min_lr(self) -> float:
        """Minimum learning rate."""
        return self.max_lr * self.min_lr_ratio

    @property
    def grad_accum_steps(self) -> int:
        """Number of gradient accumulation steps."""
        return self.total_batch_size // self.dataset_config.chunk_token_size

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
        "gpt2-xxs",  # ~19M params
        GPTConfig(n_layer=8, n_head=8, n_embd=256),
        TrainConfig(
            total_batch_size=2**18,  # ~0.25M, in number of tokens
            max_lr=3e-4,
        ),
    )

    GPT2_XXXS = TrainedModelConfig(
        "gpt2-xxxs",  # ~7M params
        GPTConfig(n_layer=4, n_head=4, n_embd=128),
        TrainConfig(
            total_batch_size=2**17,  # ~0.13M, in number of tokens
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

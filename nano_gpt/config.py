"""Configuration module."""

from dataclasses import dataclass
import enum


VOCAB_SIZE = 50257  # Fixed size for GPT model checkpoints
BLOCK_SIZE = 1024  # Fixed size for GPT model checkpoints


@dataclass
class GPTConfig:
    """This class defines the configuration for the GPT model."""

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


BATCH_SIZE = 524288  # GPT-2 uses 2**19, ~0.5M, in number of tokens per batch
NICE_VOCAB_SIZE = 50304  # Vocab size with nice power of 2


@dataclass
class TrainConfig:
    """Implementats the GPT-3 learning rate."""

    B: int = 16
    """Batch size (micro batch) used for each forward/backward pass."""

    T: int = BLOCK_SIZE
    """Sequence length used for input content."""

    total_batch_size: int = BATCH_SIZE
    """Total batch size in number of tokens for each gradient update."""

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
        self.min_lr = self.max_lr * self.min_lr_ratio
        if self.total_batch_size % (self.B * self.T) != 0:
            raise ValueError(
                "Total batch size must be divisible by B * T"
                f" but got {self.total_batch_size} % {self.B * self.T}"
            )
        self.grad_accum_steps = self.total_batch_size // (self.B * self.T)


# PRETRAINED_MODELS = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
# PRETRAINED_MODEL_CONFIG = {
#     "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
#     "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
#     "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
#     "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
# }


class TrainedModelConfig(enum.Enum):
    """This class defines the configuration for the GPT model."""

    GPT2_SMALL = (
        "gpt2",  # 124M params
        GPTConfig(n_layer=12, n_head=12, n_embd=768),
        TrainConfig(
            total_batch_size=2**19,  # ~0.5M, in number of tokens
            max_lr=6e-4,
        ),
    )
    GPT2_MEDIUM = (
        "gpt2-medium",  # 350M params
        GPTConfig(n_layer=24, n_head=16, n_embd=1024),
        TrainConfig(
            total_batch_size=2**19,  # ~0.5M, in number of tokens
            max_lr=3e-4,
        ),
    )
    GPT2_LARGE = (
        "gpt2-large",  # 774M params
        GPTConfig(n_layer=36, n_head=20, n_embd=1280),
        TrainConfig(
            total_batch_size=2**19,  # ~0.5M, in number of tokens
            max_lr=2.5e-4,
        ),
    )
    GPT2_XL = (
        "gpt2-xl",  # 1558M params
        GPTConfig(n_layer=48, n_head=25, n_embd=1600),
        TrainConfig(
            total_batch_size=2**20,  #  ~1M, in number of tokens
            max_lr=2e-4,
        ),
    )

    # These are model sizes that were made up for this project
    GPT2_XS = (
        "gpt2-xs",  # 58M params
        GPTConfig(n_layer=10, n_head=10, n_embd=512),
        TrainConfig(
            total_batch_size=2**18,  # ~0.25M, in number of tokens
            max_lr=3e-4,
        ),
    )

    GPT2_XXS = (
        "gpt2-xxs",  # 19M params
        GPTConfig(n_layer=8, n_head=8, n_embd=256),
        TrainConfig(
            total_batch_size=2**18,  # ~0.25M, in number of tokens
            max_lr=3e-4,
        ),
    )

    def __init__(self, model_name: str, config: GPTConfig, train_config: TrainConfig):
        """Initialize the configuration."""
        self.model_name = model_name
        self.model_config = config
        self.train_config = train_config


PRETRAINED = {
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
}
MODELS = {model.model_name: model for model in TrainedModelConfig}



def config_from(model_type: str) -> TrainedModelConfig:
    """Return the configuration for the model."""
    if (config := MODELS.get(model_type)) is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return config


def config_from_pretrained(model_type: str) -> GPTConfig:
    """Return the configuration for the pretrained model."""
    if model_type not in PRETRAINED:
        raise ValueError(f"Unknown model type: {model_type}")
    config = config_from(model_type)
    return config.model_config

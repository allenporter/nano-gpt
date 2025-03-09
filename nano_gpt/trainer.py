"""Trainer for nano-gpt."""

import time
from dataclasses import dataclass
import math
import logging
from typing import Any
from collections.abc import Iterator

import torch

from .model import GPT

_LOGGER = logging.getLogger(__name__)

# GPT-2 for smaller model uses 2**19, ~0.5M, in number of tokens
BATCH_SIZE = 524288


@dataclass
class TrainConfig:
    """Implementats the GPT-3 learning rate."""

    B: int = 16
    """Batch size (micro batch) used for each forward/backward pass."""

    T: int = 1024
    """Sequence length."""

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

    def get_lr(self, step: int) -> float:
        """Learning rate."""
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


@dataclass
class TrainStats:
    """Training statistics."""

    def __init__(self, config: TrainConfig) -> None:
        """Initialize the training statistics."""
        self.step = 0
        self.t0: float = 0.0
        self.config = config
        self.stats: dict[str, Any] = {}

    def start_step(self) -> None:
        """Start the step."""
        self.t0 = time.time()

    def end_step(self, loss: torch.Tensor, norm: float) -> None:
        """Step the statistics."""
        t1 = time.time()
        dt = (t1 - self.t0) * 1000
        tok_per_sec = (self.config.B * self.config.T) / (t1 - self.t0)
        self.stats.update(
            {
                "step": self.step,
                "loss": f"{loss.item():0.4f}",
                "norm": f"{norm:0.4f}",
                "dt": f"{dt:0.2f}ms",
                "tok/sec": f"{tok_per_sec:0.2f}",
            }
        )
        self.step += 1

    def __str__(self) -> str:
        """String representation."""
        return " | ".join(f"{key}: {value}" for key, value in self.stats.items())


def train(
    model: GPT,
    device: Any,
    data_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    config: TrainConfig,
    dtype: Any,
) -> None:
    """Train the model."""
    _LOGGER.info("Token batch size: %s", config.B)
    _LOGGER.info("Sequence length: %s", config.T)
    _LOGGER.info("Total token batch size: %s", config.total_batch_size)
    _LOGGER.info("Gradient accumulation steps: %s", config.grad_accum_steps)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=config.get_lr(0), device=device
    )

    ds = iter(data_loader)

    stats = TrainStats(config)

    for step in range(config.max_steps):
        stats.start_step()
        optimizer.zero_grad()

        loss: torch.Tensor = torch.tensor(0.0)
        for micro_step in range(config.grad_accum_steps):
            _LOGGER.debug("micro_step: %s", micro_step)
            x, y = next(ds)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=dtype):
                logits, loss = model(x, y)
            loss = loss / config.grad_accum_steps
            loss.backward()  # type: ignore[no-untyped-call]

        # Prevent the model from getting large shocks of gradient magnitude
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the learning rate based on the step
        lr = config.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        if "cuda" in device:
            torch.cuda.synchronize()

        stats.end_step(loss, norm)
        print(stats)

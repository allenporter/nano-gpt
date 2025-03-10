"""Trainer for nano-gpt."""

import time
from dataclasses import dataclass
import math
import logging
from typing import Any
from collections.abc import Iterator

import torch

from .model import GPT
from .config import TrainConfig

__all__ = [
    "train",
]


_LOGGER = logging.getLogger(__name__)


def get_lr(config: TrainConfig, step: int) -> float:
    """Learning rate."""
    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    if step > config.max_steps:
        return config.min_lr
    decay_ratio = (step - config.warmup_steps) / (
        config.max_steps - config.warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


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

    def end_step(self, loss: float, norm: float) -> None:
        """Step the statistics."""
        t1 = time.time()
        dt = (t1 - self.t0) * 1000
        tok_per_sec = self.config.total_batch_size / (t1 - self.t0)
        self.stats.update(
            {
                "step": self.step,
                "loss": f"{loss:0.4f}",
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
    config.log_info()
    is_cuda = "cuda" in device
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=get_lr(config, 0),
        use_fused=is_cuda,
    )

    ds = iter(data_loader)

    stats = TrainStats(config)

    for step in range(config.max_steps):
        stats.start_step()
        optimizer.zero_grad()

        loss_accum = 0.0
        for micro_step in range(config.grad_accum_steps):
            _LOGGER.debug("micro_step: %s", micro_step)
            x, y = next(ds)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=dtype):
                logits, loss = model(x, y)
            loss = loss / config.grad_accum_steps
            loss_accum += loss.item()
            loss.backward()

        # Prevent the model from getting large shocks of gradient magnitude
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the learning rate based on the step
        lr = get_lr(config, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        if is_cuda:
            torch.cuda.synchronize()

        stats.end_step(loss_accum, norm)
        print(stats)

"""Trainer for nano-gpt."""

import time
from dataclasses import dataclass
import math
import logging
from typing import Any

import torch

from .model import GPT
from .data import DataLoader

_LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Implementats the GPT-3 learning rate."""

    B: int = 16
    """Batch size."""

    T: int = 1024
    """Sequence length."""

    total_batch_size: int = 524288
    """Total batch size."""

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
        self.grad_accum_steps = self.total_batch_size // (self.B**self.T)

    def get_lr(self, step: int) -> float:
        """Learning rate."""
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


def train(
    model: GPT, device: Any, data_loader: DataLoader, config: TrainConfig
) -> None:
    """Train the model."""
    if data_loader.B != config.B:
        raise ValueError(
            f"Batch size of data loader {data_loader.B} does not match config {config.B}"
        )
    if data_loader.T != config.T:
        raise ValueError(
            f"Sequence length of data loader {data_loader.T} does not match config {config.T}"
        )

    _LOGGER.debug("total_batch_size: %s", config.total_batch_size)
    _LOGGER.debug("grad_accum_steps: %s", config.grad_accum_steps)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=config.get_lr(0), device=device
    )

    ds = iter(data_loader)

    for step in range(config.max_steps):
        t0 = time.time()
        optimizer.zero_grad()

        for micro_step in config.grad_accum_steps:
            x, y = next(ds)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / config.grad_accum_steps
            loss.backward()

        # Prevent the model from getting large shocks of gradient magnitude
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the learning rate based on the step
        lr = config.get_lr(step)
        for param_group in optimizer.param_group:  # type: ignore[attr-defined]
            param_group["lr"] = lr

        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = data_loader.chunk_size / (t1 - t0)

        print(
            "|".join(
                [
                    f"step {step}",
                    f"loss {loss.item():0.6f}",
                    f"norm: {norm:0.4f}",
                    f"dt: {dt:0.2f}ms",
                    f"tok/sec: {tokens_per_sec:0.2f}",
                ]
            )
        )

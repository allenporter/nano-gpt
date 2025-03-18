"""Trainer for nano-gpt."""

from collections.abc import Iterator, Iterable
from dataclasses import dataclass
import logging
import math
import os
import time
from typing import Any

import torch
from torch.distributed import init_process_group

from . import hellaswag_eval
from .model import GPT
from .config import TrainConfig, EvalConfig
from .datasets import hellaswag

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
class ValStats:
    """Validation statistics."""

    def __init__(self) -> None:
        """Initialize the validation statistics."""
        self.val_loss = 0.0
        self.val_loss_accum = 0.0


def compute_loss(
    model: GPT,
    device: str,
    dtype: Any,
    log_label: str,
    ds: Iterator[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    backward: bool,
) -> float:
    """Compute the validation loss.

    It is expected that the model is called in eval mode.
    This will consume items from the dataset, so it needs
    to be in the correct state before calling.
    """
    loss_accum = 0.0
    for step in range(steps):
        _LOGGER.debug("loss micro step %s: %s", log_label, step)
        x, y = next(ds)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=dtype):
            logits, loss = model(x, y)
        loss = loss / steps
        loss_accum += loss.detach()
        if backward:
            loss.backward()
    return loss_accum


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
        lr = get_lr(self.config, self.step)
        self.stats.update(
            {
                "step": self.step,
                "loss": f"{loss:0.4f}",
                "norm": f"{norm:0.4f}",
                "dt": f"{dt:0.2f}ms",
                "tok/sec": f"{tok_per_sec:0.2f}",
                "lr": f"{lr:0.6f}",
            }
        )
        self.step += 1

    def __str__(self) -> str:
        """String representation."""
        return " | ".join(f"{key}: {value}" for key, value in self.stats.items())


class WorkerState:
    """State for multi-processing using Distributed Data Parallel."""

    def __init__(self, device: str) -> None:
        """Initialize the state."""
        # set up DDP (distributed data parallel).
        # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
        self.ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if self.ddp:
            if device != "cuda":
                raise ValueError("DDP requested but requested device is not cuda")
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            self.master_process = (
                self.ddp_rank == 0
            )  # this process will do logging, checkpointing etc.
        else:
            # vanilla, non-DDP run
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.master_process = True
            # attempt to autodetect device
            self.device = device


def train(
    model: GPT,
    device: Any,
    config: TrainConfig,
    dtype: Any,
    train_data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    eval_config: EvalConfig | None = None,
    val_data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    hellaswag_loader: Iterable[hellaswag.Sample] | None = None,
) -> None:
    """Train the model."""
    config.log_info()
    is_cuda = "cuda" in device
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=get_lr(config, 0),
        use_fused=is_cuda,
    )

    train_ds = iter(train_data_loader)
    stats = TrainStats(config)

    for step in range(config.max_steps):
        last_step = step == config.max_steps - 1
        stats.start_step()

        if eval_config is not None and (step % config.eval_steps == 0 or last_step):
            if val_data_loader is not None:
                val_ds = iter(val_data_loader)
                model.eval()
                with torch.no_grad():
                    val_loss_accum = compute_loss(
                        model,
                        device,
                        dtype,
                        "val",
                        val_ds,
                        eval_config.validation_steps,
                        backward=False,
                    )
                print(
                    f"validation: loss: {val_loss_accum:0.4f} | steps: {eval_config.validation_steps}"
                )
            if hellaswag_loader is not None:
                model.eval()
                result = hellaswag_eval.evaluate(
                    model,
                    model.enc,
                    hellaswag_loader,
                    device,
                    eval_config.hellaswag_samples,
                )
                print(f"hellaswag: {result}")

        model.train()
        optimizer.zero_grad()
        loss_accum = compute_loss(
            model,
            device,
            dtype,
            "train",
            train_ds,
            config.grad_accum_steps,
            backward=True,
        )

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

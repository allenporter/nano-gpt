"""Trainer for nano-gpt."""

from collections.abc import Iterator, Iterable
import dataclasses
from dataclasses import dataclass
import logging
import math
import os
import pathlib
import time
from typing import Any

import torch
from torch import nn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from . import hellaswag_eval
from .model import sample, GPT
from .config import TrainConfig, EvalConfig, SampleConfig, DatasetConfig
from .datasets import hellaswag
from .checkpoint import save_checkpoint, Checkpoint
from .devices import get_dtype

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


@dataclass(frozen=True, kw_only=True)
class ValStats:
    """Validation statistics."""

    loss_accum: float = 0.0

    def __str__(self) -> str:
        """String representation."""
        return f"val: {self.loss_accum:0.4f}" ""


class WorkerState:
    """State for multi-processing using Distributed Data Parallel."""

    def __init__(self, device: str) -> None:
        """Initialize the state."""
        # set up DDP (distributed data parallel).
        # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
        self.ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if self.ddp and device != "cuda":
            self.ddp = False
            _LOGGER.warning(
                "DDP requested but requested device is not cuda, disabling DDP"
            )
        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.device = device

    @property
    def is_cuda(self) -> bool:
        """Check if the device is CUDA."""
        return self.device == "cuda"

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type."""
        return get_dtype(self.device)

    @property
    def is_primary(self) -> bool:
        """The primary process will do logging, checkpointing, etc."""
        return self.ddp_rank == 0


def compute_loss(
    model: nn.Module,
    worker_state: WorkerState,
    log_label: str,
    ds: Iterator[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    backward: bool,
) -> torch.Tensor:
    """Compute the validation loss.

    It is expected that the model is called in eval mode.
    This will consume items from the dataset, so it needs
    to be in the correct state before calling.
    """
    if not steps:
        raise ValueError("steps must be greater than 0")
    loss_accum = torch.zeros(1, device=worker_state.device)
    for step in range(steps):
        _LOGGER.debug("loss micro step %s: %s", log_label, step)
        x, y = next(ds)
        x, y = x.to(worker_state.device), y.to(worker_state.device)
        if worker_state.ddp:
            model.require_backward_grad_sync = step == (steps - 1)  # type: ignore[assignment]
        with torch.autocast(device_type=worker_state.device, dtype=worker_state.dtype):
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


def train(
    raw_model: GPT,
    worker_state: WorkerState,
    config: TrainConfig,
    train_data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    eval_config: EvalConfig | None = None,
    dataset_config: DatasetConfig | None = None,
    val_data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    hellaswag_loader: Iterable[hellaswag.Sample] | None = None,
    sample_config: SampleConfig | None = None,
) -> None:
    """Train the model."""
    config.log_info()

    model: nn.Module = raw_model
    tokenizer = raw_model.enc
    if worker_state.ddp:
        model = DDP(model, device_ids=[worker_state.ddp_local_rank])

    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=get_lr(config, 0),
        use_fused=worker_state.is_cuda,
    )
    _LOGGER.debug("Optimizer: %s", optimizer.state_dict())

    train_ds = iter(train_data_loader)
    stats = TrainStats(config)

    for step in range(config.max_steps):
        last_step = step == config.max_steps - 1
        stats.start_step()

        val_stats: ValStats | None = None
        eval_step = step % config.eval_steps == 0
        checkpoint_step = step % config.checkpoint_steps == 0
        if (
            (eval_step or last_step)
            and val_data_loader is not None
            and eval_config is not None
            and eval_config.validation_steps
        ):
            model.eval()
            val_ds = iter(val_data_loader)
            with torch.no_grad():
                val_loss_accum = compute_loss(
                    model,
                    worker_state,
                    "val",
                    val_ds,
                    eval_config.validation_steps,
                    backward=False,
                )
            if worker_state.ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            val_stats = ValStats(loss_accum=val_loss_accum.item())
            if worker_state.is_primary:
                print(f"{step} {val_stats}")

        if (
            (checkpoint_step or last_step)
            and worker_state.is_primary
            and config.checkpoint_dir is not None
        ):
            checkpoint_path = (
                pathlib.Path(config.checkpoint_dir) / f"checkpoint_{step}.pt"
            )
            checkpoint = Checkpoint(
                model_state_dict=model.state_dict(),
                config=raw_model.config,
                step=step,
                val_loss_accum=(
                    val_stats.loss_accum if val_stats is not None else None
                ),
                optimizer_state_dict=optimizer.state_dict(),
                train_config=config,
                dataset_config=dataset_config,
                eval_config=eval_config,
            )
            save_checkpoint(checkpoint, checkpoint_path)
        if (
            (eval_step or last_step)
            and hellaswag_loader is not None
            and eval_config is not None
            and eval_config.hellaswag_samples
        ):
            model.eval()
            with torch.no_grad():
                hellaswag_result = hellaswag_eval.evaluate(
                    model,
                    tokenizer,
                    hellaswag_loader,
                    worker_state.device,
                    eval_config.hellaswag_samples,
                )
            if worker_state.ddp:
                num_total = torch.tensor(
                    hellaswag_result.total,
                    dtype=torch.long,
                    device=worker_state.device,
                )
                num_correct_norm = torch.tensor(
                    hellaswag_result.correct,
                    dtype=torch.long,
                    device=worker_state.device,
                )
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                hellaswag_result = dataclasses.replace(
                    hellaswag_result,
                    total=int(num_total.item()),
                    correct=int(num_correct_norm.item()),
                )
            print(f"hellaswag: {hellaswag_result}")
        if (
            step > 0
            and eval_step
            and sample_config is not None
            and sample_config.num_return_sequences
        ):
            samples = sample(
                model,
                tokenizer,
                sample_config.text,
                num_return_sequences=sample_config.num_return_sequences,
                max_length=sample_config.max_length,
                device=worker_state.device,
                seed=sample_config.seed + worker_state.ddp_rank,
            )
            for i, s in enumerate(samples):
                print(f"rank {worker_state.ddp_rank} sample {i}: {s}")

        model.train()
        optimizer.zero_grad()
        loss_accum = compute_loss(
            model,
            worker_state,
            "train",
            train_ds,
            config.grad_accum_steps,
            backward=True,
        )
        if worker_state.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Prevent the model from getting large shocks of gradient magnitude
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the learning rate based on the step
        lr = get_lr(config, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        if worker_state.is_cuda:
            torch.cuda.synchronize()

        stats.end_step(loss_accum.item(), norm)
        print(stats)

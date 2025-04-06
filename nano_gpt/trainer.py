"""Trainer for nano-gpt.

This module provides a trainer for the nano-gpt model. It provides a main training
loop that can be used to train the model on a dataset. It also provides a function
for computing the loss on a dataset, and a class for managing the state of the
training process.

This supports DDP for multi-GPU training. The training process is also resumable
using checkpoints.
"""

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
from .log import LogRecord, create_log

__all__ = [
    "train",
    "create_optimizer",
]


_LOGGER = logging.getLogger(__name__)


def get_lr(config: TrainConfig, step: int) -> float:
    """Learning rate based on the current step."""
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
    """Validation statistics for logging."""

    step: int = 0
    loss_accum: float = 0.0

    def log_record(self) -> LogRecord:
        """Log record."""
        return LogRecord(
            log_type="val",
            data={
                "step": self.step,
                "loss": f"{self.loss_accum:0.4f}",
            },
        )


class WorkerState:
    """State for multi-processing using Distributed Data Parallel."""

    def __init__(self, device: str) -> None:
        """Initialize the state."""
        # set up DDP (distributed data parallel).
        # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
        self.ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if self.ddp and "cuda" not in device:
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
            torch.cuda.set_device(self.device)
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.device = device

    @property
    def is_cuda(self) -> bool:
        """Check if the device is CUDA."""
        return "cuda" in self.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type."""
        return get_dtype(self.device)

    @property
    def is_primary(self) -> bool:
        """The primary process will do logging, checkpointing, etc."""
        return self.ddp_rank == 0

    def __str__(self) -> str:
        """String representation."""
        return f"WorkerState(ddp={self.ddp}, ddp_rank={self.ddp_rank}, ddp_local_rank={self.ddp_local_rank}, ddp_world_size={self.ddp_world_size}, device={self.device})"


def compute_loss(
    model: nn.Module,
    worker_state: WorkerState,
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
    if not steps:
        raise ValueError("steps must be greater than 0")
    loss_accum = 0.0  # torch.zeros(1, device=worker_state.device)
    for step in range(steps):
        x, y = next(ds)
        x, y = x.to(worker_state.device), y.to(worker_state.device)
        if worker_state.ddp:
            model.require_backward_grad_sync = step == (steps - 1)  # type: ignore[assignment]
        with torch.autocast(device_type=worker_state.device, dtype=worker_state.dtype):
            logits, loss = model(x, y)
        loss = loss / steps
        loss_accum += loss.detach().item()
        if backward:
            loss.backward()
    return loss_accum


@dataclass
class TrainStats:
    """Training statistics for logging."""

    def __init__(self, config: TrainConfig) -> None:
        """Initialize the training statistics."""
        self.step = config.step
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

    def log_record(self) -> LogRecord:
        """Log record."""
        return LogRecord(
            log_type="train",
            data=self.stats,
        )


def create_optimizer(
    raw_model: GPT,
    config: TrainConfig,
    checkpoint: Checkpoint | None,
    is_cuda: bool,
) -> torch.optim.Optimizer:
    """Create the optimizer with the option to resume from a checkpoint."""
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=get_lr(config, 0),
        use_fused=is_cuda,
    )
    if checkpoint is not None:
        _LOGGER.info("Loading optimizer state from checkpoint")
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    _LOGGER.debug("Optimizer: %s", optimizer.state_dict())
    return optimizer


def train(
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    worker_state: WorkerState,
    config: TrainConfig,
    train_data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    eval_config: EvalConfig | None = None,
    dataset_config: DatasetConfig | None = None,
    val_data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    hellaswag_loader: Iterable[hellaswag.Sample] | None = None,
    sample_config: SampleConfig | None = None,
) -> None:
    """Train the model.

    This is the main training loop. It will train the model for the number of steps
    specified in the config. It will also evaluate the model on the validation set
    and save checkpoints.
    """
    config.log_info(worker_state.ddp_world_size)
    if worker_state.is_primary:
        log = create_log(
            pathlib.Path(config.log_file) if config.log_file else None, log_stdout=True
        )
    else:
        log = create_log(None, False)

    model: nn.Module = raw_model
    tokenizer = raw_model.enc
    model.to(worker_state.device)
    if worker_state.ddp:
        model = DDP(model, device_ids=[worker_state.ddp_local_rank])

    train_ds = iter(train_data_loader)
    stats = TrainStats(config)
    for step in range(config.step, config.max_steps):
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
                vall_loss_tensor = torch.tensor(val_loss_accum, device=worker_state.device)
                dist.all_reduce(vall_loss_tensor, op=dist.ReduceOp.AVG)
            val_stats = ValStats(step=step, loss_accum=val_loss_accum)
            if worker_state.is_primary:
                log.log(val_stats.log_record())

        if (
            step != 0
            and (step != config.step)  # don't save the initial checkpoint
            and (checkpoint_step or last_step)
            and worker_state.is_primary
            and config.checkpoint_dir is not None
        ):
            checkpoint_path = (
                pathlib.Path(config.checkpoint_dir) / f"checkpoint_{step:06d}.pt"
            )
            checkpoint: Checkpoint = Checkpoint(
                model_state_dict=raw_model.state_dict(),
                config=raw_model.config,
                step=step,
                val_loss_accum=(
                    val_stats.loss_accum if val_stats is not None else None
                ),
                optimizer_state_dict=optimizer.state_dict(),
                train_config=config,
                dataset_config=dataset_config,
                eval_config=eval_config,
                sample_config=sample_config,
            )
            save_checkpoint(checkpoint, checkpoint_path)
        if (
            step != 0
            and (eval_step or last_step)
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
            if worker_state.is_primary:
                log.log(hellaswag_result.log_record())
        if (
            step > 0
            and eval_step
            and sample_config is not None
            and sample_config.num_return_sequences
        ):
            model.eval()
            with torch.no_grad():
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
            config.grad_accum_steps(worker_state.ddp_world_size),
            backward=True,
        )
        if worker_state.ddp:
            loss_tensor = torch.tensor(loss_accum, device=worker_state.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        # Prevent the model from getting large shocks of gradient magnitude
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the learning rate based on the step
        lr = get_lr(config, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if worker_state.is_cuda:
            torch.cuda.synchronize()

        stats.end_step(loss_accum, norm.item())
        if worker_state.is_primary:
            log.log(stats.log_record())

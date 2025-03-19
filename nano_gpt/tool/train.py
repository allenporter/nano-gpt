"""Command-line interface for training the model.

Usage:
```
usage: nano-gpt train [-h] [--pretrained {gpt2-large,gpt2-medium,gpt2,gpt2-xl}]
                      [--model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs}] [--device DEVICE]
                      [--sequence-length SEQUENCE_LENGTH] [--seed SEED] [--compile | --no-compile] [--total-batch-size TOTAL_BATCH_SIZE]
                      [--streaming | --no-streaming] [--max-steps MAX_STEPS] [--eval-steps EVAL_STEPS]
                      [--validation-steps VALIDATION_STEPS] [--hellaswag-samples HELLASWAG_SAMPLES]
                      [--sample-num-sequences SAMPLE_NUM_SEQUENCES] [--sample-max-length SAMPLE_MAX_LENGTH] [--sample-seed SAMPLE_SEED]
                      [--dataset {finewebedu,tinyshakespeare}] [--dataset-dir DATASET_DIR] [--micro-batch-size MICRO_BATCH_SIZE]

Train a model

options:
  -h, --help            show this help message and exit
  --total-batch-size TOTAL_BATCH_SIZE
                        The number of tokens to use in each gradient accumulation batch (of micro-batches).
  --streaming, --no-streaming
                        Stream the dataset without downloading the entire corpus.
  --max-steps MAX_STEPS
                        The maximum number of training steps.
  --eval-steps EVAL_STEPS
                        The number of steps between evaluations.
  --micro-batch-size MICRO_BATCH_SIZE
                        The number of batches of examples to pull from the dataset in each micro step.

model:
  --pretrained {gpt2-large,gpt2-medium,gpt2,gpt2-xl}
                        The name of the pretrained model to use.
  --model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs}
                        Use the specified model name configuration default values.
  --device DEVICE       The device to use.
  --sequence-length SEQUENCE_LENGTH
                        The sequence length used for input content in each micro batch.
  --seed SEED           The seed to use for sampling/training.
  --compile, --no-compile
                        Will compile the model if supported by the device.

eval:
  --validation-steps VALIDATION_STEPS
                        Number of validation loss iterations to perform each eval round.
  --hellaswag-samples HELLASWAG_SAMPLES
                        The number of HellaSwag evaluation results to sample or None for the entire set.

sample:
  --sample-num-sequences SAMPLE_NUM_SEQUENCES
                        The number of sequences to generate.
  --sample-max-length SAMPLE_MAX_LENGTH
                        The maximum length of the generated sequences.
  --sample-seed SAMPLE_SEED
                        The seed to use for sampling.

dataset:
  --dataset {finewebedu,tinyshakespeare}
                        Use the specified dataset.
  --dataset-dir DATASET_DIR
                        Directory where the dataset is stored.
```
"""

import argparse
import logging
from collections.abc import Iterable
import datetime

import torch

from nano_gpt.datasets.data_loader import read_preprocessed_corpus
from nano_gpt.datasets import hellaswag
from nano_gpt.trainer import train, WorkerState
from nano_gpt.checkpoint import CHECKPOINT_DIR

from .model_config import (
    create_model_arguments,
    create_eval_arguments,
    create_sample_arguments,
    create_dataset_arguments,
    dataset_config_from_args,
    eval_config_from_args,
    model_from_args,
    sample_config_from_args,
    load_checkpoint_context,
)


_LOGGER = logging.getLogger(__name__)


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    create_model_arguments(args)
    args.add_argument(
        "--total-batch-size",
        type=int,
        help="The number of tokens to use in each gradient accumulation batch (of micro-batches).",
    )
    args.add_argument(
        "--streaming",
        type=str,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream the dataset without downloading the entire corpus.",
    )
    args.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="The maximum number of training steps.",
    )
    args.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="The number of steps between evaluations.",
    )
    args.add_argument(
        "--checkpoint-steps",
        type=int,
        default=None,
        help="The number of steps between checkpoints.",
    )
    args.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(CHECKPOINT_DIR),
        help="The path to the checkpoint directory",
    )
    args.add_argument(
        "--log-file",
        type=str,
        default="train_{now}.log".format(
            now=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ),
        help="The path to the log file.",
    )

    create_eval_arguments(args)
    create_sample_arguments(args)
    create_dataset_arguments(args)


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    torch.set_float32_matmul_precision("high")
    if args.dataset is None:
        raise ValueError("Required flag --dataset not set")

    with load_checkpoint_context(args) as checkpoint:
        eval_config = eval_config_from_args(args, checkpoint)
        sample_config = sample_config_from_args(args, checkpoint)
        _LOGGER.info(f"Sample config: {sample_config}")

        model, tokenizer, config = model_from_args(args, checkpoint)
        if config is None:
            raise ValueError("No trainable model configuration found")

        dataset_config = dataset_config_from_args(args, checkpoint)
        _LOGGER.info(f"Dataset config: {dataset_config}")

    worker_state = WorkerState(args.device)
    _LOGGER.info("Worker state: %s", worker_state)

    _LOGGER.info("Loading dataset %s (streaming=%s)", args.dataset, args.streaming)
    train_data_loader = read_preprocessed_corpus(
        dataset_config.dataset_path("train"),
        dataset_config,
        worker_num=worker_state.ddp_rank,
        worker_count=worker_state.ddp_world_size,
    )
    val_data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None
    hellaswag_val: Iterable[hellaswag.Sample] | None = None
    if eval_config.validation_steps:
        val_data_loader = read_preprocessed_corpus(
            dataset_config.dataset_path("validation"),
            dataset_config,
            worker_num=worker_state.ddp_rank,
            worker_count=worker_state.ddp_world_size,
        )
    if eval_config.hellaswag_samples is not None:
        hellaswag_val = hellaswag.load_dataset("validation")
    _LOGGER.info("Dataset loaded")
    train(
        model,
        worker_state,
        config.train_config,
        train_data_loader=train_data_loader,
        eval_config=eval_config,
        dataset_config=dataset_config,
        val_data_loader=val_data_loader,
        hellaswag_loader=hellaswag_val,
        sample_config=sample_config,
    )
    return 0

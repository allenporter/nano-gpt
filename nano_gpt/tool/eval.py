"""Command-line interface for evaling a trained model.

Usage:
```
usage: nano-gpt eval [-h] [--pretrained {gpt2-large,gpt2-xl,gpt2,gpt2-medium}]
                     [--model {gpt2,gpt2-medium,gpt2-large,gpt2-xl,gpt2-xs,gpt2-xxs}] [--device DEVICE]
                     [--sequence-length SEQUENCE_LENGTH] [--seed SEED] [--compile | --no-compile]
                     [--validation-steps VALIDATION_STEPS] [--hellaswag-samples HELLASWAG_SAMPLES]
                     [--dataset {finewebedu,tinyshakespeare}] [--dataset-dir DATASET_DIR] [--micro-batch-size MICRO_BATCH_SIZE]

Evaluate a model

options:
  -h, --help            show this help message and exit
  --micro-batch-size MICRO_BATCH_SIZE
                        The number of batches of examples to pull from the dataset in each micro step.

model:
  --pretrained {gpt2-large,gpt2-xl,gpt2,gpt2-medium}
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

dataset:
  --dataset {finewebedu,tinyshakespeare}
                        Use the specified dataset.
  --dataset-dir DATASET_DIR
                        Directory where the dataset is stored.
```
"""

import argparse
import logging

import torch

from nano_gpt.datasets import hellaswag
from nano_gpt.datasets.data_loader import read_preprocessed_corpus
from nano_gpt import hellaswag_eval, trainer
from nano_gpt.devices import get_dtype

from .model_config import (
    create_model_arguments,
    model_from_args,
    eval_config_from_args,
    create_eval_arguments,
    create_dataset_arguments,
    dataset_config_from_args,
)

_LOGGER = logging.getLogger(__name__)

DATASET = "hellaswag"
SPLIT = "validation"


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    create_model_arguments(args)
    create_eval_arguments(args)
    create_dataset_arguments(args)


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    torch.set_float32_matmul_precision("high")

    eval_config = eval_config_from_args(args)
    _LOGGER.info(f"Eval config: {eval_config}")
    model, tokenizer, _ = model_from_args(args)
    model.eval()

    if args.dataset and eval_config.validation_steps:
        dataset_config = dataset_config_from_args(args)
        _LOGGER.info(f"Dataset config: {dataset_config}")
        val_data_loader = read_preprocessed_corpus(
            dataset_config.dataset_path(SPLIT),
            dataset_config,
        )
        val_ds = iter(val_data_loader)
        with torch.no_grad():
            loss_accum = trainer.compute_loss(
                model,
                device=args.device,
                dtype=get_dtype(args.device),
                log_label="val",
                ds=val_ds,
                steps=eval_config.validation_steps,
                backward=False,
            )
        print(
            f"validation loss: {loss_accum:0.4f} | steps: {eval_config.validation_steps}"
        )

    hellaswag_val = hellaswag.load_dataset(SPLIT)
    result = hellaswag_eval.evaluate(
        model,
        tokenizer,
        hellaswag_val,
        args.device,
        eval_config.hellaswag_samples,
    )
    print(f"hellaswag: {result}")

    return 0

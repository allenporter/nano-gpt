"""Shared library for command line flags for loading models."""

from argparse import ArgumentParser, BooleanOptionalAction
import logging
from typing import Any, cast

import torch

from nano_gpt.config import (
    MODELS,
    PRETRAINED,
    config_from,
    TrainedModelConfig,
    HellaSwagEvalConfig,
    SampleConfig,
)
from nano_gpt.model import GPT
from nano_gpt.devices import get_device
from nano_gpt.tokenizer import get_tokenizer, Tokenizer

_LOGGER = logging.getLogger(__name__)

DATASET_DIR = "dataset_cache"


def create_model_arguments(
    args: ArgumentParser, default_values: dict[str, Any] | None = None
) -> None:
    """Create arguments for model loading."""
    if default_values is None:
        default_values = {}
    group = args.add_argument_group("model")
    group.add_argument(
        "--pretrained",
        type=str,
        choices=PRETRAINED,
        help="The name of the pretrained model to use.",
    )
    group.add_argument(
        "--model",
        type=str,
        default=default_values.get("model", "gpt2"),
        choices=MODELS,
        help="Use the specified model name configuration default values.",
    )
    group.add_argument(
        "--device",
        type=str,
        default=get_device(),
        help="The device to use.",
    )
    group.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="The sequence length used for input content in each micro batch.",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=default_values.get("seed", 0),
        help="The seed to use for sampling/training.",
    )
    group.add_argument(
        "--compile",
        type=str,
        action=BooleanOptionalAction,
        default=True,
        help="Will compile the model if supported by the device.",
    )


def _check_model_arguments(args: Any) -> None:
    """Check that the model arguments are valid."""
    if args.pretrained is None and args.model is None:
        raise ValueError("Either --pretrained or --model must be specified")


def model_config_from_args(
    args: Any,
) -> TrainedModelConfig:
    """Create a model from the flags."""
    return config_from(
        args.model,
        **{
            key: value
            for key in {"micro_batch_size", "sequence_length", "total_batch_size"}
            if (value := getattr(args, key, None)) is not None
        },
    )


def model_from_args(args: Any) -> tuple[GPT, Tokenizer, TrainedModelConfig | None]:
    """Create a model from the flags."""
    _check_model_arguments(args)
    tokenizer = get_tokenizer()
    trained_model_config: TrainedModelConfig | None = None
    if args.pretrained is not None:
        _LOGGER.info("loading weights from pretrained gpt: %s" % args.pretrained)
        model = GPT.from_pretrained(args.pretrained, tokenizer=tokenizer)
    else:
        trained_model_config = config_from(
            args.model,
            **{
                key: value
                for key in {
                    "micro_batch_size",
                    "sequence_length",
                    "total_batch_size",
                    "max_steps",
                    "eval_steps",
                    "num_eval_samples",
                }
                if (value := getattr(args, key, None)) is not None
            },
        )
        model_config = trained_model_config.model_config
        _LOGGER.debug("initializing model from config: %s", model_config)
        model = GPT(model_config, tokenizer=tokenizer)
    model.to(args.device)
    if args.device == "cuda":
        if args.compile:
            _LOGGER.info("Compiling model")
            model = cast(GPT, torch.compile(model))
        else:
            _LOGGER.debug("Not compiling model")
    else:
        _LOGGER.debug("Model will not be compiled")

    if args.seed:
        _LOGGER.info("Setting seed to %s", args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    _LOGGER.debug("args: %s", args)

    return model, tokenizer, trained_model_config


def create_eval_arguments(args: ArgumentParser) -> None:
    """Create arguments for model evaluation."""
    group = args.add_argument_group("eval")
    group.add_argument(
        "--eval-num-samples",
        type=int,
        default=None,
        help="The number of samples to evaluate or all of omitted.",
    )


def eval_config_from_args(args: Any) -> HellaSwagEvalConfig:
    """Create an HellaSwagEvalConfig from the flags."""
    values = {}
    if args.eval_num_samples is not None:
        values["num_samples"] = args.eval_num_samples
    return HellaSwagEvalConfig(**values)


def create_sample_arguments(args: ArgumentParser) -> None:
    """Create arguments for model sampling."""
    group = args.add_argument_group("sample")
    group.add_argument(
        "--sample-num-sequences",
        type=int,
        default=5,
        help="The number of sequences to generate.",
    )
    group.add_argument(
        "--sample-max-length",
        type=int,
        default=30,
        help="The maximum length of the generated sequences.",
    )
    group.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="The seed to use for sampling.",
    )


def sample_config_from_args(args: Any) -> SampleConfig:
    """Create an SampleConfig from the flags."""
    values = {}
    if args.sample_num_sequences is not None:
        values["num_return_sequences"] = args.sample_num_sequences
    if args.sample_max_length is not None:
        values["max_length"] = args.sample_max_length
    if args.sample_seed is not None:
        values["seed"] = args.sample_seed
    return SampleConfig(**values)

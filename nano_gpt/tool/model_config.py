"""Shared library for command line flags for loading models."""

from argparse import ArgumentParser
import logging
from typing import Any, cast

import torch

from nano_gpt.config import (
    MODELS,
    PRETRAINED,
    config_from,
    TrainedModelConfig,
)
from nano_gpt.model import GPT
from nano_gpt.devices import get_device
from nano_gpt.tokenizer import get_tokenizer, Tokenizer

_LOGGER = logging.getLogger(__name__)


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


def _check_model_arguments(args: Any) -> None:
    """Check that the model arguments are valid."""
    if args.pretrained is None and args.model is None:
        raise ValueError("Either --pretrained or --model must be specified")
    if args.pretrained is not None and args.model is not None:
        raise ValueError("Only one of --pretrained or --model can be specified")


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
                for key in {"micro_batch_size", "sequence_length", "total_batch_size"}
                if (value := getattr(args, key, None)) is not None
            },
        )
        model_config = trained_model_config.model_config
        _LOGGER.debug("initializing model from config: %s", model_config)
        model = GPT(model_config, tokenizer=tokenizer)
    model.to(args.device)
    if args.device == "cuda":
        _LOGGER.debug("Compiling model")
        model = cast(GPT, torch.compile(model))
    else:
        _LOGGER.debug("Model will not be compiled")

    if args.seed:
        _LOGGER.info("Setting seed to %s", args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    _LOGGER.debug("args: %s", args)

    return model, tokenizer, trained_model_config

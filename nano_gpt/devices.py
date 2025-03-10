"""Library for working with hardware devices."""

import torch
import logging


_LOGGER = logging.getLogger(__name__)


def get_device() -> str:
    """Pick the best device available."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def get_dtype(device: str) -> torch.dtype:
    """Get the type for the device."""
    if device == "mps":
        return torch.float16  # bfloat16 not supported on MPS
    if device == "cuda":
        return torch.bfloat16
    return torch.bfloat16

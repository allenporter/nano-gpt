"""Module for running the huggingface eval against the model.

This module provides a function for evaluating a model on the hellaswag dataset. The
dataset contains multiple choice questions with a single correct answer. The model
must predict the correct answer from a list of 4 choices.
"""

from collections.abc import Iterable
from dataclasses import dataclass
import logging
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from .datasets import hellaswag
from .devices import get_dtype
from .tokenizer import Tokenizer
from .log import LogRecord

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "evaluate",
    "HellaSwagResult",
]

DATASET = "hellaswag"
SPLIT = "validation"


def get_likely_row(
    tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor
) -> int:
    """Get the most likely row from the logits."""
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return cast(int, pred_norm)


@dataclass
class HellaSwagResult:
    """HellaSwag result."""

    accuracy: float = 0.0
    total: int = 0
    correct: int = 0

    def add_result(self, correct: bool) -> None:
        """Add a result."""
        self.total += 1
        self.correct += int(correct)
        self.accuracy = self.correct / self.total

    @property
    def stats(self) -> dict[str, float | str]:
        """Get the stats."""
        return {
            "accuracy": f"{self.accuracy:0.4f}",
            "total": self.total,
            "correct": self.correct,
        }

    def log_record(self) -> LogRecord:
        """Log record."""
        return LogRecord(
            log_type="hellaswag",
            data=self.stats,
        )


def evaluate(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: Iterable[hellaswag.Sample],
    device: str,
    num_samples: int | None = None,
) -> HellaSwagResult:
    """Evaluate the model on the hellaswag dataset."""
    result = HellaSwagResult()
    for i, example in enumerate(dataset):
        if num_samples is not None and i >= num_samples:
            break
        tokens, mask = example.tokenize(tokenizer)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=get_dtype(device)):
                logits, loss = model(tokens)
            pred_norm = get_likely_row(tokens, mask, logits)
        result.add_result(pred_norm == example.label)
        _LOGGER.debug("hellaswag: %s", result.log_record())
    return result

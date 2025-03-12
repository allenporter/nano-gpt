"""Command-line interface for evaling a trained model.

Usage:
```
usage: nano-gpt eval [-h] [--pretrained {gpt2-xl,gpt2-medium,gpt2-large,gpt2}] [--device DEVICE]

Evaluate a model

options:
  -h, --help            show this help message and exit
  --pretrained {gpt2-xl,gpt2-medium,gpt2-large,gpt2}
                        The name of the pretrained model to use when sampling.
  --device DEVICE       The device to use for sampling.
```
"""

import argparse
import logging
from typing import cast

import torch
from torch.nn import functional as F

from nano_gpt.model import GPT
from nano_gpt.tokenizer import get_tokenizer
from nano_gpt.devices import get_device, get_dtype
from nano_gpt.config import PRETRAINED
from nano_gpt.datasets import hellaswag

_LOGGER = logging.getLogger(__name__)

DATASET = "hellaswag"
SPLIT = "validation"


def create_arguments(args: argparse.ArgumentParser) -> None:
    """Get parsed passed in arguments."""
    args.add_argument(
        "--pretrained",
        type=str,
        choices=PRETRAINED,
        help="The name of the pretrained model to use when sampling.",
    )
    args.add_argument(
        "--device",
        type=str,
        default=get_device(),
        help="The device to use for sampling.",
    )


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


def run(args: argparse.Namespace) -> int:
    """Run the sample command."""
    torch.set_float32_matmul_precision("high")
    model = GPT.from_pretrained(args.pretrained, get_tokenizer())
    model.eval()
    model = model.to(args.device)
    tokenizer = get_tokenizer()

    num_total = 0
    num_correct = 0
    for i, example in enumerate(hellaswag.load_dataset(SPLIT)):
        tokens, mask = example.tokenize(tokenizer)
        tokens = tokens.to(args.device)
        mask = mask.to(args.device)
        with torch.no_grad():
            with torch.autocast(device_type=args.device, dtype=get_dtype(args.device)):
                logits, loss = model(tokens)
            pred_norm = get_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct += int(pred_norm == example.label)
        accuracy = num_correct / num_total
        print(f"Accuracy: {num_correct}/{num_total} = {accuracy:.4f}")

    return 0

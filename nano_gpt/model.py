"""This module defines the GPT model architecture.

This is a thin wrapper around the HuggingFace transformers library and uses
the approach from the GPT-2/GPT-3 papers.
"""

from typing import cast
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from .tokenizer import Tokenizer
from .config import GPTConfig, model_config_from_pretrained

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "GPT",
    "sample",
]


class CausalSelfAttention(nn.Module):
    """Attention module."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize MLP."""
        super().__init__()
        # Batch of key/query/value projects for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1  # type: ignore[assignment]
        # Regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference."""
        B, T, C = x.size()
        # Compute the query, key, value for all heads in the batch.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        # Each are (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute attention with a fused kernel of fast attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reassemble and concat everything
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize MLP."""
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """A transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize Block."""
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """This class defines the GPT model."""

    def __init__(self, config: GPTConfig, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        # Final classifier
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.enc = tokenizer

        # Share weights for input and output embeddings. This is about 30% of
        # the model weights.
        self.transformer.wte.weight = self.lm_head.weight  # type: ignore[union-attr]
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Perform additional weight initialization to match gpt-2."""
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.05
                torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=std)

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        use_fused: bool,
    ) -> torch.optim.AdamW:
        """Return the optimizer."""
        # start with all params that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim optim_groups
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        _LOGGER.info(
            "num decay_params %s (tensors) / %s (parameters)",
            len(decay_params),
            num_decay_params,
        )
        _LOGGER.info(
            "num nodecay_params %s (tensors) / %s (parameters)",
            len(nodecay_params),
            num_nodecay_params,
        )
        _LOGGER.info("Using fused adamw : %s", use_fused)
        return torch.optim.AdamW(
            params=optim_groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Perform the forward pass.

        Returns the output values and loss if targets are provided.
        """
        B, T = x.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
            )
        # Forward token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)  # Shape (T)

        # (T, n_emb)
        pos_emb = cast(nn.Embedding, self.transformer.wpe)(pos)
        # (B, T, n_emb)
        tok_emb = cast(nn.Embedding, self.transformer.wte)(x)
        x = tok_emb + pos_emb
        # Forward transformer blocks
        for block in cast(nn.ModuleList, self.transformer.h):
            x = block(x)
        # Forward the final layernorm
        x = cast(nn.LayerNorm, self.transformer.ln_f)(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                # Flatten to (BxT, vocab_size)
                logits.view(-1, logits.size(-1)),
                # Flatten to (BxT)
                targets.view(-1),
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str, tokenizer: Tokenizer) -> "GPT":
        """Load the GPT from the pretrained model."""
        _LOGGER.info("loading weights from pretrained gpt: %s" % model_type)
        model_config = model_config_from_pretrained(model_type)
        model = GPT(model_config, tokenizer=tokenizer)
        sd = model.state_dict()
        sd_keys = [
            k for k in sd.keys() if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        PRETRAINED_TRANSPOSED_WEIGHTS = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in PRETRAINED_TRANSPOSED_WEIGHTS):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


def sample(
    model: nn.Module,
    tokenizer: Tokenizer,
    text: str,
    num_return_sequences: int,
    max_length: int,
    device: str,
    seed: int = 42,
) -> list[str]:
    """Sample from the model from text input."""
    tokenized_text = tokenizer.encode(text)
    tokens = torch.tensor(tokenized_text, dtype=torch.long)  # (8, )
    # Replicate input tokens
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    # x is (B, T)
    x = tokens.to(device)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)

    # With each loop iteration we'll append a token to the sequence. This is
    # adding one more column to x each time.
    while x.size(1) < max_length:
        logits, _ = model(x)  # (B, T, vocab_size)
        # Take the logits at the last position (next character) and drop the others.
        # This is correct but inefficient implementation of sampling.
        # Question: What is T?
        logits = logits[:, -1, :]  # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # Do top-k sampling of 50 which is the huggingface default. Get the top 50
        # probabilities and set all other tokens to probability of zero. This helps
        # keep the model on track so it doesn't go off the rails as easily.
        # Both are (5, 50)
        topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
        # Select a token from the top 5
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
        # Gather corresponding indicidies
        xcol = torch.gather(topk_indicies, -1, ix)
        # Append the new character to the sequence (one for each in the batch)
        x = torch.cat((x, xcol), dim=-1)

    samples = []
    for i in range(num_return_sequences):
        seq_tokens = x[i, :max_length].tolist()
        decoded = tokenizer.decode(seq_tokens)
        samples.append(decoded)

    return samples

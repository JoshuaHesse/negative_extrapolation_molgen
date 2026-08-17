from __future__ import annotations

import copy
import math

import torch
from torch import nn


class SmilesRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        return self.output(out)


class SmilesTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[1] > self.max_len:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_len={self.max_len}")
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        hidden = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        hidden = hidden + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(
            hidden,
            mask=mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        return self.output(self.norm(hidden))


def build_model(vocab_size: int, config: dict) -> nn.Module:
    config = dict(config)
    architecture = config.pop("architecture", "rnn")
    if architecture == "rnn":
        return SmilesRNN(vocab_size, **config)
    if architecture == "transformer":
        return SmilesTransformer(vocab_size, **config)
    raise ValueError(f"Unknown model architecture '{architecture}'")


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def matches_any_pattern(name: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    return any(pattern in name for pattern in patterns)


def parameter_update_norm(
    base: nn.Module,
    tuned: nn.Module,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> float:
    """Return the global L2 norm of a scoped parameter update."""
    base_state = base.state_dict()
    tuned_state = tuned.state_dict()
    squared_norm = 0.0
    for key, value in base_state.items():
        included = include_patterns is None or matches_any_pattern(key, include_patterns)
        excluded = matches_any_pattern(key, exclude_patterns)
        if not torch.is_floating_point(value) or not included or excluded:
            continue
        delta = (tuned_state[key] - value).detach().float()
        squared_norm += float(torch.sum(delta * delta).cpu())
    return squared_norm**0.5


def negative_extrapolate(
    base: nn.Module,
    bad: nn.Module,
    scale: float,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    norm_match_to: nn.Module | None = None,
) -> nn.Module:
    """Extrapolate away from ``bad``, optionally matching another update norm.

    Norm matching uses one global L2 norm over the floating-point tensors in
    the requested parameter scope. This preserves the learned direction and
    only controls its total displacement from the base model.
    """
    merged = copy.deepcopy(base)
    base_state = base.state_dict()
    bad_state = bad.state_dict()
    merged_state = merged.state_dict()
    direction_scale = 1.0
    if norm_match_to is not None:
        direction_norm = parameter_update_norm(
            base,
            bad,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        reference_norm = parameter_update_norm(
            base,
            norm_match_to,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        if direction_norm == 0.0:
            raise ValueError("Cannot norm-match a zero parameter-update direction.")
        direction_scale = reference_norm / direction_norm

    for key in merged_state:
        included = include_patterns is None or matches_any_pattern(key, include_patterns)
        excluded = matches_any_pattern(key, exclude_patterns)
        if torch.is_floating_point(merged_state[key]) and included and not excluded:
            direction = bad_state[key] - base_state[key]
            merged_state[key] = base_state[key] - scale * direction_scale * direction
        else:
            merged_state[key] = base_state[key]
    merged.load_state_dict(merged_state)
    return merged

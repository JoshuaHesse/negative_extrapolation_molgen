from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from neon_molgen.model import build_model
from neon_molgen.tokenizer import SmilesTokenizer


def save_checkpoint(path: str | Path, model: nn.Module, tokenizer: SmilesTokenizer, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, SmilesTokenizer, dict]:
    checkpoint = torch.load(Path(path), map_location=device)
    tokenizer = SmilesTokenizer.from_dict(checkpoint["tokenizer"])
    config = checkpoint["config"]
    model = build_model(len(tokenizer.itos), config["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, config


def save_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

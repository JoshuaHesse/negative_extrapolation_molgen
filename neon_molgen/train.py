from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from tqdm import tqdm

from neon_molgen.data import make_loader
from neon_molgen.tokenizer import SmilesTokenizer


def train_model(
    model: nn.Module,
    smiles: list[str],
    tokenizer: SmilesTokenizer,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    desc: str,
    seed: int | None = None,
) -> nn.Module:
    model, _ = train_model_with_history(
        model,
        smiles,
        tokenizer,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        desc=desc,
        seed=seed,
    )
    return model


def train_model_with_history(
    model: nn.Module,
    smiles: list[str],
    tokenizer: SmilesTokenizer,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    desc: str,
    seed: int | None = None,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    loader = make_loader(smiles, tokenizer, batch_size=batch_size, shuffle=True, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        progress = tqdm(loader, desc=desc, leave=False)
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, padding_mask=x.eq(tokenizer.pad_id))
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            n_batches += 1
            progress.set_postfix(loss=loss_value)
        row = {
            "epoch": float(epoch + 1),
            "loss_mean": running_loss / max(n_batches, 1),
            "n_batches": float(n_batches),
        }
        history.append(row)
        if on_epoch_end is not None:
            on_epoch_end(row)
    return model, history

from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm

from neon_molgen.tokenizer import SmilesTokenizer


@torch.no_grad()
def sample_smiles(
    model: nn.Module,
    tokenizer: SmilesTokenizer,
    *,
    n_samples: int,
    max_len: int,
    temperature: float,
    batch_size: int,
    device: str,
) -> list[str]:
    model = model.to(device)
    model.eval()
    samples = []

    for start in tqdm(range(0, n_samples, batch_size), desc="sample", leave=False):
        current_batch = min(batch_size, n_samples - start)
        tokens = torch.full((current_batch, 1), tokenizer.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(current_batch, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = model(tokens)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(tokenizer.eos_id)
            if bool(finished.all()):
                break

        for row in tokens.cpu().tolist():
            samples.append(tokenizer.decode(row))
    return samples

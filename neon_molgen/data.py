from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from neon_molgen.tokenizer import SmilesTokenizer


def read_smiles(path: str | Path, *, limit: int | None = None) -> list[str]:
    smiles = []
    with Path(path).open() as handle:
        for line in handle:
            value = line.strip().split(",")[0]
            if not value or value.lower() == "smiles":
                continue
            smiles.append(value)
            if limit is not None and len(smiles) >= limit:
                break
    return smiles


class SmilesDataset(Dataset):
    def __init__(self, smiles: list[str], tokenizer: SmilesTokenizer) -> None:
        self.encoded = [tokenizer.encode(smi) for smi in smiles]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> list[int]:
        return self.encoded[idx]


def make_collate(pad_id: int):
    def collate(batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x in batch)
        x = torch.full((len(batch), max_len - 1), pad_id, dtype=torch.long)
        y = torch.full((len(batch), max_len - 1), pad_id, dtype=torch.long)
        for i, ids in enumerate(batch):
            x[i, : len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long)
            y[i, : len(ids) - 1] = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

    return collate


def make_loader(
    smiles: list[str],
    tokenizer: SmilesTokenizer,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        SmilesDataset(smiles, tokenizer),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate(tokenizer.pad_id),
        generator=generator,
    )

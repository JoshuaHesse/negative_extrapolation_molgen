from __future__ import annotations

import re
from dataclasses import dataclass

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
SMILES_REGEX_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|"
    r":|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)
SMILES_REGEX = re.compile(SMILES_REGEX_PATTERN)


def tokenize_smiles(smiles: str) -> list[str]:
    tokens = SMILES_REGEX.findall(smiles)
    if "".join(tokens) != smiles:
        return list(smiles)
    return tokens


@dataclass
class SmilesTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_smiles(cls, smiles: list[str]) -> SmilesTokenizer:
        tokens = sorted({token for smi in smiles for token in tokenize_smiles(smi)})
        itos = SPECIAL_TOKENS + tokens
        stoi = {token: i for i, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    def encode(self, smiles: str) -> list[int]:
        return [self.bos_id, *[self.stoi[token] for token in tokenize_smiles(smiles)], self.eos_id]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for idx in ids:
            if idx == self.eos_id:
                break
            if idx in {self.pad_id, self.bos_id}:
                continue
            chars.append(self.itos[idx])
        return "".join(chars)

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> SmilesTokenizer:
        itos = list(data["itos"])
        return cls(stoi={token: i for i, token in enumerate(itos)}, itos=itos)

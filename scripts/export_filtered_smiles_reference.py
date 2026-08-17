from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a canonical, unique SMILES reference from filtered sample tables."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--input-file", default="base_samples.csv")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--filter-column", required=True)
    parser.add_argument("--filter-value", type=float, required=True)
    return parser.parse_args()


def canonical_unique_smiles(frame: pd.DataFrame) -> list[str]:
    if "valid" in frame.columns:
        frame = frame[frame["valid"].astype(bool)]
    smiles_column = "canonical_smiles" if "canonical_smiles" in frame.columns else "smiles"

    smiles: list[str] = []
    seen: set[str] = set()
    for value in frame[smiles_column].dropna().astype(str):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical not in seen:
            seen.add(canonical)
            smiles.append(canonical)
    return smiles


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    for seed in args.seeds:
        seed_dir = results_dir / f"seed_{seed}"
        input_path = seed_dir / args.input_file
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        frame = pd.read_csv(input_path)
        if args.filter_column not in frame.columns:
            raise ValueError(f"{input_path} has no column {args.filter_column!r}")
        selected = frame[frame[args.filter_column].eq(args.filter_value)]
        smiles = canonical_unique_smiles(selected)
        if not smiles:
            raise ValueError(
                f"Filter {args.filter_column} == {args.filter_value} selected no valid "
                f"molecules from {input_path}"
            )

        output_path = seed_dir / args.output_file
        output_path.write_text("\n".join(smiles) + "\n")
        print(
            f"seed={seed}: wrote {len(smiles)} unique valid molecules to {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()

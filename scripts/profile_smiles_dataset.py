from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from neon_molgen.data import read_smiles
from neon_molgen.scoring import score_smiles, summarize_scores
from neon_molgen.tokenizer import tokenize_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/dataset_profiles")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Read only the first N SMILES. Use this for quick checks.",
    )
    parser.add_argument(
        "--score-limit",
        type=int,
        default=100000,
        help="Score only the first N read SMILES. Use 0 to score all read SMILES.",
    )
    parser.add_argument(
        "--lengths-only",
        action="store_true",
        help="Skip RDKit atom/property scoring and report only SMILES/token lengths.",
    )
    return parser.parse_args()


def count_lines(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line in handle:
            value = line.strip().split(",")[0]
            if value and value.lower() != "smiles":
                count += 1
    return count


def length_summary(values: list[int]) -> dict[str, float]:
    series = pd.Series(values)
    return {
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def atom_counts(smiles: list[str]) -> list[int]:
    counts = []
    for smi in tqdm(smiles, desc="atom counts"):
        mol = Chem.MolFromSmiles(smi)
        counts.append(mol.GetNumHeavyAtoms() if mol is not None else 0)
    return counts


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or input_path.stem

    total_rows = count_lines(input_path)
    smiles = read_smiles(input_path, limit=args.limit)
    if not smiles:
        raise ValueError(f"No SMILES read from {input_path}")

    token_lengths = [len(tokenize_smiles(smi)) + 2 for smi in smiles]
    smiles_lengths = [len(smi) for smi in smiles]
    summary = {
        "name": name,
        "input": str(input_path),
        "total_rows_in_file": int(total_rows),
        "n_read": int(len(smiles)),
        "read_limit": args.limit,
        "score_limit": args.score_limit,
        "smiles_length": length_summary(smiles_lengths),
        "token_length_with_bos_eos": length_summary(token_lengths),
    }
    if not args.lengths_only:
        heavy_atom_counts = atom_counts(smiles)

        score_smiles_subset = smiles
        if args.score_limit and args.score_limit > 0:
            score_smiles_subset = smiles[: args.score_limit]
        scores = score_smiles(score_smiles_subset)
        scores_path = output_dir / f"{name}_scores.csv"
        scores.to_csv(scores_path, index=False)

        valid = scores[scores["valid"]]
        summary.update(
            {
                "n_scored": int(len(scores)),
                "n_valid_scored": int(len(valid)),
                "heavy_atom_count": length_summary(heavy_atom_counts),
                **summarize_scores(scores),
            }
        )
        if len(valid):
            summary.update(
                {
                    "mw_p10": float(valid["mw"].quantile(0.10)),
                    "mw_p90": float(valid["mw"].quantile(0.90)),
                    "logp_p10": float(valid["logp"].quantile(0.10)),
                    "logp_p90": float(valid["logp"].quantile(0.90)),
                    "tpsa_p10": float(valid["tpsa"].quantile(0.10)),
                    "tpsa_p90": float(valid["tpsa"].quantile(0.90)),
                }
            )
    summary_path = output_dir / f"{name}_profile.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    flat_summary = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (str, int, float)) or value is None or np.isscalar(value)
    }
    pd.DataFrame([flat_summary]).to_csv(output_dir / f"{name}_profile.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Count unique usable scaffolds in an epoch-sensitivity experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--reference-smiles",
        default=None,
        help=(
            "Optional training/reference SMILES file used to require novelty. "
            "If omitted, usable molecules are valid, unique, and liability-free."
        ),
    )
    parser.add_argument("--liability-column", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def as_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def scaffold_metrics(
    frame: pd.DataFrame,
    *,
    reference_smiles: set[str] | None,
    liability_column: str,
    n_sampled: int | None = None,
) -> dict[str, float | int]:
    required = {
        "valid",
        "canonical_smiles",
        "murcko_scaffold",
        liability_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sample table is missing columns: {sorted(missing)}")

    valid_unique = frame.loc[
        as_boolean(frame["valid"]) & frame["canonical_smiles"].notna()
    ].drop_duplicates("canonical_smiles")
    novel = valid_unique
    if reference_smiles is not None:
        novel = valid_unique.loc[
            ~valid_unique["canonical_smiles"].astype(str).isin(reference_smiles)
        ]
    liability_values = pd.to_numeric(novel[liability_column], errors="coerce")
    usable = novel.loc[liability_values.eq(0)].copy()

    scaffolds = usable["murcko_scaffold"].fillna("").astype(str)
    scaffold_counts = scaffolds.loc[scaffolds.ne("")].value_counts()
    n_observed = len(frame)
    n_sampled = n_observed if n_sampled is None else int(n_sampled)
    n_usable = len(usable)
    n_scaffolds = len(scaffold_counts)
    return {
        "n_sampled": n_sampled,
        "n_observed": n_observed,
        "n_unique_usable_molecules": n_usable,
        "n_usable_acyclic_molecules": int(scaffolds.eq("").sum()),
        "n_unique_usable_scaffolds": n_scaffolds,
        "unique_usable_scaffold_yield": n_scaffolds / n_sampled if n_sampled else np.nan,
        "unique_scaffold_fraction_among_usable": (
            n_scaffolds / n_usable if n_usable else np.nan
        ),
        "top10_usable_scaffold_fraction": (
            float(scaffold_counts.iloc[:10].sum() / scaffold_counts.sum())
            if len(scaffold_counts)
            else np.nan
        ),
    }


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary = pd.read_csv(results_dir / "epoch_sensitivity_summary.csv")
    reference_smiles = (
        set(Path(args.reference_smiles).read_text().splitlines())
        if args.reference_smiles
        else None
    )
    rows = []
    record_columns = ["model", "epoch", "direction_kind"]
    if "n_sampled" in summary:
        record_columns.append("n_sampled")
    for record in summary[record_columns].to_dict("records"):
        sample_path = results_dir / f"{record['model']}_samples.csv"
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        rows.append(
            {
                **record,
                **scaffold_metrics(
                    pd.read_csv(sample_path),
                    reference_smiles=reference_smiles,
                    liability_column=args.liability_column,
                    n_sampled=record.get("n_sampled"),
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    output_path = (
        Path(args.output) if args.output else results_dir / "usable_scaffold_metrics.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(metrics.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

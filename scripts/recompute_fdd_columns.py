from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_distribution_distance import (  # noqa: E402
    discover_seed_dirs,
    fdd,
    model_name_from_sample,
    parse_extra_references,
    physchem_frechet_10d,
    read_sample_smiles,
    reference_smiles,
    sample_files,
    sample_smiles,
    seed_from_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh paper-compatible FDD columns in existing distribution-distance tables without recalculating FCD."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--metrics", required=True, help="Existing distribution_distance_metrics.csv to update in place.")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--reference-smiles", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    metrics_path = Path(args.metrics)
    metrics = pd.read_csv(metrics_path)
    requested_models = set(args.models) if args.models else None
    requested_rows = set(
        zip(
            metrics["seed"].astype(int),
            metrics["model"].astype(str),
            metrics["reference"].astype(str),
            strict=True,
        )
    )
    extra_references = parse_extra_references(args.reference_smiles)

    refreshed = {}
    for seed_dir in discover_seed_dirs(results_dir):
        seed = seed_from_dir(seed_dir)
        refs = reference_smiles(seed_dir, extra_references)
        for sample_path in sample_files(seed_dir, requested_models):
            model = model_name_from_sample(sample_path)
            model_smiles = read_sample_smiles(sample_path)
            for ref_name, ref_smiles in refs.items():
                key = (seed, model, ref_name)
                if key not in requested_rows:
                    continue
                row = metrics[
                    metrics["seed"].astype(int).eq(seed)
                    & metrics["model"].astype(str).eq(model)
                    & metrics["reference"].astype(str).eq(ref_name)
                ].iloc[0]
                n_compared = int(row["n_compared"])
                if n_compared <= 1:
                    refreshed[key] = (float("nan"), float("nan"))
                    continue
                left_key = f"{seed}:{model}:model:{ref_name}:{n_compared}"
                right_key = f"{seed}:{model}:reference:{ref_name}:{n_compared}"
                sampled_model = sample_smiles(model_smiles, n_compared, key=left_key)
                sampled_ref = (
                    sampled_model
                    if model == ref_name == "base"
                    else sample_smiles(ref_smiles, n_compared, key=right_key)
                )
                refreshed[key] = (fdd(sampled_model, sampled_ref), physchem_frechet_10d(sampled_model, sampled_ref))

    for index, row in metrics.iterrows():
        key = (int(row["seed"]), str(row["model"]), str(row["reference"]))
        if key in refreshed:
            metrics.at[index, "fdd"] = refreshed[key][0]
            metrics.at[index, "physchem_frechet_10d"] = refreshed[key][1]
    metrics.to_csv(metrics_path, index=False)

    numeric = [
        column
        for column in metrics.select_dtypes(include="number").columns
        if column not in {"seed", "n_model_available", "n_reference_available", "n_compared"}
    ]
    summary = metrics.groupby(["model", "reference"])[numeric].agg(["mean", "std", "count"])
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns]
    summary.reset_index().to_csv(metrics_path.with_name("distribution_distance_summary.csv"), index=False)
    print(f"Refreshed {len(refreshed)} FDD comparisons in {metrics_path}")


if __name__ == "__main__":
    main()

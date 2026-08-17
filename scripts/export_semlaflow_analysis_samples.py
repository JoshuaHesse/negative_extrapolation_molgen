#!/usr/bin/env python3
"""Export SemlaFlow sample folders to the common SMILES-analysis layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_MODELS = [
    "base",
    "random_tuned",
    "positive_tuned",
    "bad_tuned",
    "full_model_neon_lambda_2p5",
    "full_model_random_neon_lambda_2p5",
    "full_model_norm_matched_random_corrected_neon_lambda_2p5",
    "full_model_positive_corrected_neon_lambda_2p5",
    "full_model_neon_lambda_4",
    "full_model_random_neon_lambda_4",
    "full_model_norm_matched_random_corrected_neon_lambda_4",
    "full_model_positive_corrected_neon_lambda_4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective-column", default="four_liability_hit")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser.parse_args()


def sample_dir(seed_dir: Path, model: str) -> Path | None:
    candidates = [
        seed_dir / "joint" / "samples" / model,
        seed_dir / "random_reference" / "samples" / model,
        seed_dir / "positive_tuning" / "samples" / model,
    ]
    return next((path for path in candidates if (path / "scores.csv").is_file()), None)


def load_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "valid" not in frame.columns:
        frame["valid"] = frame["canonical_smiles"].notna()
    columns = ["canonical_smiles", "valid"]
    extra = [column for column in frame.columns if column not in columns]
    return frame[[*columns, *extra]].copy()


def write_smi(smiles: pd.Series, path: Path) -> int:
    values = smiles.dropna().astype(str)
    values = values[values.ne("")]
    unique = sorted(values.unique())
    path.write_text("\n".join(unique) + ("\n" if unique else ""))
    return len(unique)


def export_seed(seed_dir: Path, output_seed_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    output_seed_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for model in args.models:
        directory = sample_dir(seed_dir, model)
        if directory is None:
            continue
        scores = load_scores(directory / "scores.csv")
        scores.to_csv(output_seed_dir / f"{model}_samples.csv", index=False)
        exported.append(model)

    base_path = output_seed_dir / "base_samples.csv"
    if not base_path.is_file():
        raise FileNotFoundError(f"Could not export base_samples.csv for {seed_dir}")
    base = pd.read_csv(base_path)
    valid_base = base[base["valid"].fillna(False).astype(bool)].copy()
    if args.objective_column not in valid_base.columns:
        raise ValueError(f"{base_path} is missing objective column {args.objective_column!r}")

    hits = valid_base[valid_base[args.objective_column].fillna(False).astype(bool)]
    nonhits = valid_base[~valid_base[args.objective_column].fillna(False).astype(bool)]
    n_good = write_smi(nonhits["canonical_smiles"], output_seed_dir / "good_smiles.smi")
    n_bad = write_smi(hits["canonical_smiles"], output_seed_dir / "bad_smiles.smi")
    write_smi(nonhits["canonical_smiles"], output_seed_dir / "liability_free_base.smi")

    return {
        "seed": seed_dir.name.removeprefix("seed_"),
        "n_models": len(exported),
        "models": ",".join(exported),
        "n_base_valid": int(len(valid_base)),
        "n_base_good": int(n_good),
        "n_base_bad": int(n_bad),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = int(seed_dir.name.removeprefix("seed_"))
        if args.seeds is not None and seed not in args.seeds:
            continue
        rows.append(export_seed(seed_dir, output_dir / seed_dir.name, args))
    if not rows:
        raise ValueError(f"No seed directories found under {root}")
    pd.DataFrame(rows).to_csv(output_dir / "export_summary.csv", index=False)
    print(f"Wrote SemlaFlow analysis sample export to {output_dir}")


if __name__ == "__main__":
    main()

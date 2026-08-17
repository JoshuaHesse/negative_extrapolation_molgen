#!/usr/bin/env python3
"""Run PoseBusters on saved SemlaFlow samples without regenerating molecules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import posebusters as pb
from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=SAMPLE_DIR",
        help="Model label and directory containing one SDF, rdkit_profile.csv, and scores.csv.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--objective-column",
        default="brenk_top5_oxygen_nitrogen_single_bond_hit",
    )
    parser.add_argument("--posebusters-config", default="mol")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --model value {spec!r}; expected NAME=SAMPLE_DIR")
    name, directory = spec.split("=", 1)
    if not name:
        raise ValueError(f"Empty model name in {spec!r}")
    return name, Path(directory)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def analyze_model(
    name: str,
    sample_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    model_output = output_dir / name
    model_output.mkdir(parents=True, exist_ok=True)
    summary_path = model_output / "summary.json"
    if args.resume and summary_path.is_file():
        print(f"Resume: reusing {summary_path}")
        return json.loads(summary_path.read_text())

    sdf_paths = sorted(sample_dir.glob("*.sdf"))
    if len(sdf_paths) != 1:
        raise ValueError(f"Expected one SDF in {sample_dir}, found {len(sdf_paths)}")
    rdkit_profile = pd.read_csv(sample_dir / "rdkit_profile.csv")
    scores = pd.read_csv(sample_dir / "scores.csv")
    if len(rdkit_profile) != len(scores):
        raise ValueError(
            f"Profile/score mismatch for {name}: {len(rdkit_profile)} != {len(scores)}"
        )
    if args.objective_column not in scores:
        raise ValueError(f"Missing {args.objective_column!r} in {sample_dir / 'scores.csv'}")

    sdf_molecules = list(Chem.SDMolSupplier(str(sdf_paths[0]), removeHs=False, sanitize=False))
    if any(mol is None for mol in sdf_molecules):
        raise ValueError(f"Could not load every SDF record for {name}")
    present_mask = bool_series(rdkit_profile["rdkit_mol_present"])
    present_positions = rdkit_profile.index[present_mask].to_numpy()
    if len(sdf_molecules) != len(present_positions):
        raise ValueError(
            f"SDF/profile present mismatch for {name}: {len(sdf_molecules)} != {len(present_positions)}"
        )

    buster = pb.PoseBusters(config=args.posebusters_config, max_workers=args.workers)
    checks = buster.bust(sdf_molecules, None, None).reset_index(drop=True)
    checks.to_csv(model_output / "posebusters_profile.csv", index=False)
    pass_values = checks.fillna(False).astype(bool).all(axis=1).to_numpy()

    scores = scores.copy()
    scores["posebusters_all_checks"] = False
    scores.loc[present_positions, "posebusters_all_checks"] = pass_values
    scores.to_csv(model_output / "scores_with_posebusters.csv", index=False)

    valid = bool_series(scores["valid"])
    all_checks = bool_series(scores["posebusters_all_checks"])
    n_valid = int(valid.sum())
    n_valid_posebusters_pass = int((valid & all_checks).sum())
    valid_unique = scores.loc[valid].drop_duplicates("canonical_smiles")
    liability_free = pd.to_numeric(
        valid_unique[args.objective_column], errors="coerce"
    ).fillna(1).eq(0)
    posebusters_pass = bool_series(valid_unique["posebusters_all_checks"])
    usable_3d = valid_unique.loc[liability_free & posebusters_pass]
    n_sampled = len(scores)

    summary: dict[str, object] = {
        "model": name,
        "sample_dir": str(sample_dir),
        "n_sampled": n_sampled,
        "n_sdf_records": len(sdf_molecules),
        "n_posebusters_pass": int(pass_values.sum()),
        "posebusters_all_checks_fraction_of_sdf": float(pass_values.mean()),
        "posebusters_all_checks_yield": float(pass_values.sum() / n_sampled),
        "n_rdkit_valid": n_valid,
        "n_rdkit_valid_posebusters_pass": n_valid_posebusters_pass,
        # Conditional 3D-quality guardrail, separated from upstream graph validity.
        "posebusters_all_checks_fraction_of_rdkit_valid": (
            float(n_valid_posebusters_pass / n_valid) if n_valid else float("nan")
        ),
        "n_valid_unique_liability_free_posebusters_pass": int(len(usable_3d)),
        # Fixed-budget yield after molecular validity, uniqueness, liability
        # removal, and all PoseBusters checks.
        "usable_3d_yield": float(len(usable_3d) / n_sampled),
    }
    for column in checks.columns:
        summary[f"posebusters_{column}_fraction"] = float(
            checks[column].fillna(False).astype(bool).mean()
        )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(
        f"{name}: PB pass={summary['posebusters_all_checks_yield']:.4f}, "
        f"usable 3D yield={summary['usable_3d_yield']:.4f}"
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        analyze_model(name, sample_dir, output_dir, args)
        for name, sample_dir in map(parse_model_spec, args.model)
    ]
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "posebusters_summary.csv", index=False)
    print(summary[[
        "model",
        "n_sampled",
        "posebusters_all_checks_yield",
        "usable_3d_yield",
    ]].to_string(index=False))
    print(f"Wrote {output_dir / 'posebusters_summary.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate seed-paired SemlaFlow four-liability experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def add_conditional_posebusters_metrics(frame: pd.DataFrame, posebusters_dir: Path) -> pd.DataFrame:
    """Recover conditional PoseBusters rates from existing per-sample masks."""
    frame = frame.copy()
    for index, model in frame["model"].items():
        scores_path = posebusters_dir / str(model) / "scores_with_posebusters.csv"
        if not scores_path.is_file():
            continue
        scores = pd.read_csv(scores_path, usecols=["valid", "posebusters_all_checks"])
        valid = bool_series(scores["valid"])
        passes = bool_series(scores["posebusters_all_checks"])
        n_valid = int(valid.sum())
        n_valid_pass = int((valid & passes).sum())
        frame.at[index, "n_rdkit_valid"] = n_valid
        frame.at[index, "n_rdkit_valid_posebusters_pass"] = n_valid_pass
        frame.at[index, "posebusters_all_checks_fraction_of_rdkit_valid"] = (
            n_valid_pass / n_valid if n_valid else float("nan")
        )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser.parse_args()


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def read_seed(seed_dir: Path) -> pd.DataFrame:
    random_summary = pd.read_csv(seed_dir / "random_reference" / "summary.csv")
    joint_summary = pd.read_csv(seed_dir / "joint" / "summary.csv")

    random_control = random_summary[random_summary["model"].eq("random_tuned")].copy()
    if len(random_control) != 1:
        raise ValueError(f"Expected one random_tuned row in {seed_dir}")

    required_joint = {"base", "positive_tuned", "bad_tuned"}
    missing = required_joint - set(joint_summary["model"])
    if missing:
        raise ValueError(f"{seed_dir} is missing required joint models: {sorted(missing)}")

    rows = [joint_summary, random_control]

    # Later task-arithmetic passes can overwrite joint/summary.csv with only
    # the models generated in that pass. The per-model summary.json files are
    # the source of truth for completed sampled models and let us recover older
    # rows, such as random-corrected NE, without rerunning sampling.
    present_models = set(pd.concat(rows, ignore_index=True)["model"])
    recovered = []
    for summary_path in sorted((seed_dir / "joint" / "samples").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        model = summary.get("model")
        if model and model not in present_models:
            recovered.append(summary)
            present_models.add(model)
    if recovered:
        rows.append(pd.DataFrame(recovered))

    frame = pd.concat(rows, ignore_index=True)
    frame.insert(0, "seed", int(seed_dir.name.removeprefix("seed_")))
    return frame


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column for column in frame.select_dtypes(include="number").columns if column != "seed"
    ]
    summary = frame.groupby("model", sort=False)[numeric_columns].agg(["mean", "std", "count"])
    return flatten_columns(summary.reset_index())


def add_base_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column for column in frame.select_dtypes(include="number").columns if column != "seed"
    ]
    base = (
        frame[frame["model"].eq("base")][["seed", *numeric_columns]]
        .drop_duplicates("seed")
        .set_index("seed")
    )
    rows = []
    for _, row in frame.iterrows():
        seed = int(row["seed"])
        output = {"seed": seed, "model": row["model"]}
        for column in numeric_columns:
            output[f"delta_{column}"] = row[column] - base.at[seed, column]
        rows.append(output)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(
        path
        for path in root.glob("seed_*")
        if (path / "random_reference" / "summary.csv").is_file()
        and (path / "joint" / "summary.csv").is_file()
        and (args.seeds is None or int(path.name.removeprefix("seed_")) in args.seeds)
    )
    if not seed_dirs:
        raise ValueError(f"No completed seed runs found under {root}")

    metrics = pd.concat([read_seed(seed_dir) for seed_dir in seed_dirs], ignore_index=True)
    summary = aggregate(metrics)
    deltas = add_base_deltas(metrics)
    delta_summary = aggregate(deltas)

    metrics.to_csv(output_dir / "replicate_metrics.csv", index=False)
    summary.to_csv(output_dir / "aggregate_summary.csv", index=False)
    deltas.to_csv(output_dir / "delta_metrics.csv", index=False)
    delta_summary.to_csv(output_dir / "delta_summary.csv", index=False)

    posebusters_frames = []
    for seed_dir in seed_dirs:
        path = seed_dir / "joint" / "analysis" / "posebusters" / "posebusters_summary.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame = add_conditional_posebusters_metrics(frame, path.parent)
        frame.insert(0, "seed", int(seed_dir.name.removeprefix("seed_")))
        posebusters_frames.append(frame)
    if posebusters_frames:
        posebusters = pd.concat(posebusters_frames, ignore_index=True)
        posebusters.to_csv(output_dir / "posebusters_replicate_metrics.csv", index=False)
        aggregate(posebusters).to_csv(
            output_dir / "posebusters_aggregate_summary.csv", index=False
        )

    print(
        summary[
            [
                "model",
                "valid_fraction_mean",
                "four_liability_hit_fraction_mean",
                "usable_yield_mean",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote SemlaFlow replicate summaries to {output_dir}")


if __name__ == "__main__":
    main()

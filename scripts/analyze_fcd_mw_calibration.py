from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from scripts.analyze_distribution_distance import (
    build_fcd,
    discover_seed_dirs,
    fcd_distance,
    fdd,
    physchem_frechet_10d,
    read_sample_smiles,
    sample_smiles,
    seed_from_dir,
)

MW_BANDS = [
    ("base_resample", None, None, "Unfiltered baseline resample"),
    ("mw_bottom_10pct", 0.00, 0.10, "MW bottom 10%"),
    ("mw_bottom_20pct", 0.00, 0.20, "MW bottom 20%"),
    ("mw_20_40pct", 0.20, 0.40, "MW 20-40% band"),
    ("mw_40_60pct", 0.40, 0.60, "MW 40-60% band"),
    ("mw_60_80pct", 0.60, 0.80, "MW 60-80% band"),
    ("mw_top_20pct", 0.80, 1.00, "MW top 20%"),
    ("mw_top_10pct", 0.90, 1.00, "MW top 10%"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate FCD/FDD magnitude by comparing interpretable molecular-weight "
            "bands from baseline generator samples to the unfiltered baseline distribution."
        )
    )
    parser.add_argument("--results-dir", required=True, help="Experiment directory containing seed_*/base_samples.csv files.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to RESULTS_DIR/analysis/fcd_mw_calibration.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Maximum molecules per compared set.")
    parser.add_argument("--min-size", type=int, default=500, help="Minimum molecules required for a band to be analyzed.")
    parser.add_argument("--device", default="cpu", help="Device for fcd_torch, e.g. cpu or cuda:0.")
    parser.add_argument("--jobs", type=int, default=1, help="Worker count passed to fcd_torch when supported.")
    parser.add_argument("--flush-every", type=int, default=10, help="Write partial metrics after this many completed rows.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-fcd", action="store_true", help="Skip ChemNet FCD and compute descriptor controls only.")
    return parser.parse_args()


def mol_weight(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


def molecular_weight_frame(smiles: list[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles:
        mw = mol_weight(smi)
        if mw is not None:
            rows.append({"smiles": smi, "mw": mw})
    return pd.DataFrame(rows)


def select_band(frame: pd.DataFrame, q_low: float | None, q_high: float | None) -> tuple[list[str], float, float]:
    if q_low is None or q_high is None:
        return frame["smiles"].tolist(), float(frame["mw"].min()), float(frame["mw"].max())
    low = float(frame["mw"].quantile(q_low))
    high = float(frame["mw"].quantile(q_high))
    if q_low <= 0.0:
        selected = frame[frame["mw"].le(high)]
    elif q_high >= 1.0:
        selected = frame[frame["mw"].ge(low)]
    else:
        selected = frame[frame["mw"].ge(low) & frame["mw"].le(high)]
    return selected["smiles"].tolist(), low, high


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mw_mean": np.nan, "mw_p10": np.nan, "mw_p50": np.nan, "mw_p90": np.nan}
    array = np.asarray(values, dtype=float)
    return {
        "mw_mean": float(np.mean(array)),
        "mw_p10": float(np.quantile(array, 0.10)),
        "mw_p50": float(np.quantile(array, 0.50)),
        "mw_p90": float(np.quantile(array, 0.90)),
    }


def summarize(metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.select_dtypes(include="number").columns
        if column not in {"seed", "n_base_available", "n_band_available", "n_compared", "n_reference"}
    ]
    summary = metrics.groupby(["band", "band_label", "q_low", "q_high"], dropna=False)[numeric].agg(["mean", "std", "count"])
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "mw_fcd_calibration_summary.csv", index=False)
    return summary


def write_report(metrics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, fcd_available: bool, fcd_error: str | None) -> None:
    lines = [
        "# Molecular-Weight FCD Calibration",
        "",
        "This control compares molecular-weight bands sampled from baseline generator outputs against an unfiltered baseline reference. It provides experiment-local anchors for interpreting FCD values: the same generator is used throughout, but the query set is increasingly biased toward low- or high-MW chemistry.",
        "",
        f"FCD available: `{fcd_available}`",
    ]
    if fcd_error:
        lines.append(f"FCD note: `{fcd_error}`")
    lines.extend([
        "",
        "## Files",
        "",
        "- `mw_fcd_calibration_metrics.csv`: per-seed band distances.",
        "- `mw_fcd_calibration_summary.csv`: mean/std/count across seeds.",
        "",
        "## Summary",
        "",
    ])
    display_cols = [
        "band_label",
        "fcd_mean",
        "fcd_std",
        "fdd_mean",
        "fdd_std",
        "mw_mean_mean",
        "mw_p10_mean",
        "mw_p90_mean",
    ]
    available_cols = [col for col in display_cols if col in summary.columns]
    if available_cols:
        lines.append("| " + " | ".join(available_cols) + " |")
        lines.append("| " + " | ".join("---" for _ in available_cols) + " |")
        for _, row in summary[available_cols].iterrows():
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append("" if np.isnan(value) else f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("_No summary columns available._")
    (output_dir / "mw_fcd_calibration_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis" / "fcd_mw_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    fcd_model = None
    fcd_error = None
    if not args.skip_fcd:
        fcd_model, fcd_error = build_fcd(args.device, args.jobs)
        if fcd_error:
            print(f"FCD unavailable: {fcd_error}")
    fcd_cache: dict[str, object] = {}

    rows = []
    seed_dirs = discover_seed_dirs(
        results_dir,
        set(args.seeds) if args.seeds is not None else None,
    )
    for seed_dir in seed_dirs:
        seed = seed_from_dir(seed_dir)
        base_path = seed_dir / "base_samples.csv"
        if not base_path.exists():
            print(f"Skipping {seed_dir}: no base_samples.csv")
            continue
        base_smiles = read_sample_smiles(base_path)
        base_frame = molecular_weight_frame(base_smiles)
        if len(base_frame) < args.min_size:
            print(f"Skipping seed={seed}: only {len(base_frame)} valid baseline molecules")
            continue

        reference = sample_smiles(
            base_frame["smiles"].tolist(),
            args.sample_size,
            key=f"mw_calibration::{seed}::reference",
        )
        reference_mw = [mw for mw in (mol_weight(smi) for smi in reference) if mw is not None]

        for band, q_low, q_high, band_label in MW_BANDS:
            selected, mw_cut_low, mw_cut_high = select_band(base_frame, q_low, q_high)
            if len(selected) < args.min_size:
                print(f"Skipping seed={seed} band={band}: n={len(selected)} < min_size={args.min_size}")
                continue
            query = sample_smiles(selected, args.sample_size, key=f"mw_calibration::{seed}::{band}")
            query_mw = [mw for mw in (mol_weight(smi) for smi in query) if mw is not None]
            row = {
                "seed": seed,
                "band": band,
                "band_label": band_label,
                "q_low": q_low,
                "q_high": q_high,
                "mw_cut_low": mw_cut_low,
                "mw_cut_high": mw_cut_high,
                "n_base_available": int(len(base_frame)),
                "n_band_available": int(len(selected)),
                "n_reference": int(len(reference)),
                "n_compared": int(len(query)),
                "reference_mw_mean": float(np.mean(reference_mw)) if reference_mw else np.nan,
                "reference_mw_p10": float(np.quantile(reference_mw, 0.10)) if reference_mw else np.nan,
                "reference_mw_p90": float(np.quantile(reference_mw, 0.90)) if reference_mw else np.nan,
                **numeric_summary(query_mw),
                "fdd": fdd(query, reference),
                "physchem_frechet_10d": physchem_frechet_10d(query, reference),
                "fcd": fcd_distance(
                    fcd_model,
                    query,
                    reference,
                    fcd_cache,
                    key_a=f"seed_{seed}::{band}",
                    key_b=f"seed_{seed}::reference",
                ),
            }
            rows.append(row)
            if args.flush_every and len(rows) % args.flush_every == 0:
                pd.DataFrame(rows).to_csv(output_dir / "mw_fcd_calibration_metrics.partial.csv", index=False)
                print(f"completed {len(rows)} band comparisons; latest seed={seed} band={band}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "mw_fcd_calibration_metrics.csv", index=False)
    summary = summarize(metrics, output_dir) if not metrics.empty else pd.DataFrame()
    write_report(metrics, summary, output_dir, fcd_model is not None, fcd_error)
    print(f"Wrote molecular-weight FCD calibration to {output_dir}")


if __name__ == "__main__":
    main()

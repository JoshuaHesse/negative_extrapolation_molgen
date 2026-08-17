#!/usr/bin/env python3
"""Count unique liability-free scaffolds per fixed SemlaFlow sampling budget."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MODEL_DIRS = {
    "base": "base",
    "positive_tuned": "positive_tuned",
    "full_model_neon_lambda_2p5": "standard_ne_2p5",
    "full_model_positive_corrected_neon_lambda_2p5": "positive_corrected_ne_2p5",
}
MODEL_ORDER = list(MODEL_DIRS)
MODEL_LABELS = {
    "base": "Base",
    "positive_tuned": "Positive FT",
    "full_model_neon_lambda_2p5": "Standard NE",
    "full_model_positive_corrected_neon_lambda_2p5": "Positive-corrected NE",
}
MODEL_COLORS = {
    "base": "#7A8089",
    "positive_tuned": "#EE9B00",
    "full_model_neon_lambda_2p5": "#1A759F",
    "full_model_positive_corrected_neon_lambda_2p5": "#2D6A4F",
}
FOCAL = "full_model_positive_corrected_neon_lambda_2p5"
FIXED_SCAFFOLD_TARGET = 2500
COMPARATORS = [
    "base",
    "positive_tuned",
    "full_model_neon_lambda_2p5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/external/semlaflow/four_liability_joint_replicates",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser.parse_args()


def bool_series(frame: pd.DataFrame, column: str, *, missing: bool) -> pd.Series:
    if column not in frame:
        return pd.Series(missing, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(missing)
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(float(missing)).astype(bool)


def scaffold_metrics(frame: pd.DataFrame, mask: pd.Series, scope: str) -> dict[str, float | int | str]:
    usable = frame.loc[mask].dropna(subset=["canonical_smiles"]).drop_duplicates(
        "canonical_smiles"
    )
    scaffolds = usable["murcko_scaffold"].dropna().astype(str)
    scaffolded = scaffolds[scaffolds.ne("")]
    counts = scaffolded.value_counts()
    n_sampled = len(frame)
    n_usable = len(usable)
    n_scaffolds = len(counts)
    observed_scaffolds: set[str] = set()
    samples_to_target = np.nan
    for sampled_index, (keep, scaffold) in enumerate(
        zip(mask, frame["murcko_scaffold"].fillna("").astype(str), strict=True), start=1
    ):
        if keep and scaffold:
            observed_scaffolds.add(scaffold)
        if len(observed_scaffolds) >= FIXED_SCAFFOLD_TARGET:
            samples_to_target = sampled_index
            break
    return {
        "scope": scope,
        "n_sampled": n_sampled,
        "n_unique_usable_molecules": n_usable,
        "n_usable_acyclic_molecules": int((usable["murcko_scaffold"].fillna("") == "").sum()),
        "n_unique_usable_scaffolds": n_scaffolds,
        "fixed_scaffold_target": FIXED_SCAFFOLD_TARGET,
        "samples_to_fixed_scaffold_target": samples_to_target,
        "unique_usable_scaffold_yield": n_scaffolds / n_sampled if n_sampled else np.nan,
        "unique_scaffold_fraction_among_usable": n_scaffolds / n_usable if n_usable else np.nan,
        "top10_usable_scaffold_fraction": (
            float(counts.iloc[:10].sum() / counts.sum()) if len(counts) else np.nan
        ),
    }


def load_metrics(root: Path, seeds: list[int] | None = None) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        if seeds is not None and seed not in seeds:
            continue
        pb_root = seed_dir / "joint" / "analysis" / "posebusters"
        for model, directory in MODEL_DIRS.items():
            path = pb_root / directory / "scores_with_posebusters.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            valid = bool_series(frame, "valid", missing=False)
            liability = bool_series(frame, "four_liability_hit", missing=True)
            posebusters = bool_series(frame, "posebusters_all_checks", missing=False)
            masks = {
                "2d_usable": valid & ~liability,
                "3d_usable": valid & ~liability & posebusters,
            }
            for scope, mask in masks.items():
                row = {"seed": seed, "model": model}
                row.update(scaffold_metrics(frame, mask, scope))
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in metrics.select_dtypes("number") if column != "seed"]
    summary = metrics.groupby(["model", "scope"], sort=False)[numeric].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def holm_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(float)
    result = np.full(len(p), np.nan)
    order = np.argsort(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        result[index] = running
    return pd.Series(result, index=values.index)


def paired_statistics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    endpoint_specs = [
        ("2d_usable", "unique_usable_scaffold_yield"),
        ("3d_usable", "unique_usable_scaffold_yield"),
        ("2d_usable", "samples_to_fixed_scaffold_target"),
        ("3d_usable", "samples_to_fixed_scaffold_target"),
    ]
    for scope, endpoint in endpoint_specs:
        subset = metrics[metrics["scope"].eq(scope)]
        wide = subset.pivot(index="seed", columns="model", values=endpoint)
        for comparator in COMPARATORS:
            paired = wide[[FOCAL, comparator]].dropna()
            difference = (paired[FOCAL] - paired[comparator]).to_numpy()
            n = len(difference)
            sd = float(np.std(difference, ddof=1))
            sem = sd / math.sqrt(n)
            critical = float(stats.t.ppf(0.975, n - 1))
            test = stats.ttest_rel(paired[FOCAL], paired[comparator])
            rows.append(
                {
                    "scope": scope,
                    "endpoint": endpoint,
                    "focal_model": FOCAL,
                    "comparator_model": comparator,
                    "n_pairs": n,
                    "focal_mean": float(paired[FOCAL].mean()),
                    "comparator_mean": float(paired[comparator].mean()),
                    "paired_mean_difference": float(np.mean(difference)),
                    "ci95_low": float(np.mean(difference) - critical * sem),
                    "ci95_high": float(np.mean(difference) + critical * sem),
                    "cohen_dz": float(np.mean(difference) / sd),
                    "paired_t_p": float(test.pvalue),
                }
            )
    output = pd.DataFrame(rows)
    output["paired_t_p_holm"] = output.groupby(["scope", "endpoint"], group_keys=False)[
        "paired_t_p"
    ].apply(holm_adjust)
    return output


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(colors="black", width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    for panel, (ax, scope, title) in enumerate(
        zip(axes[:2], ("2d_usable", "3d_usable"), ("2D usable", "PoseBusters-qualified"), strict=True)
    ):
        subset = metrics[metrics["scope"].eq(scope)]
        values = [
            100 * subset.loc[subset["model"].eq(model), "unique_usable_scaffold_yield"].to_numpy()
            for model in MODEL_ORDER
        ]
        artists = ax.boxplot(
            values,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            boxprops={"edgecolor": "black", "linewidth": 0.8},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
        )
        for patch, model in zip(artists["boxes"], MODEL_ORDER, strict=True):
            patch.set_facecolor(MODEL_COLORS[model])
        ax.set_xticks(
            range(1, len(MODEL_ORDER) + 1),
            [MODEL_LABELS[model] for model in MODEL_ORDER],
            rotation=25,
            ha="right",
        )
        ax.set_title(title)
        ax.text(
            0.015,
            0.98,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
            ha="left",
            va="top",
        )
        style_axis(ax)
    axes[0].set_ylabel("Unique usable scaffolds / sampled molecules (%)")
    axes[0].set_ylim(50.5, 83.5)
    axes[1].set_ylim(50.5, 83.5)

    ax = axes[2]
    subset = metrics[metrics["scope"].eq("3d_usable")]
    values = [
        subset.loc[
            subset["model"].eq(model), "samples_to_fixed_scaffold_target"
        ].to_numpy()
        for model in MODEL_ORDER
    ]
    artists = ax.boxplot(
        values,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    for patch, model in zip(artists["boxes"], MODEL_ORDER, strict=True):
        patch.set_facecolor(MODEL_COLORS[model])
    ax.set_xticks(
        range(1, len(MODEL_ORDER) + 1),
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=25,
        ha="right",
    )
    ax.set_title("Fixed-target efficiency")
    ax.set_ylabel(f"Samples to {FIXED_SCAFFOLD_TARGET:,} 3D usable scaffolds")
    ax.text(
        0.015, 0.98, "c", transform=ax.transAxes, fontweight="bold", fontsize=11,
        ha="left", va="top"
    )
    style_axis(ax)
    base = output_dir / "semlaflow_unique_usable_scaffold_yield"
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(
            f"{value:.6g}" if isinstance(value, float) else str(value) for value in row
        ) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "analysis" / "usable_scaffolds"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(root, args.seeds)
    summary = aggregate(metrics)
    paired = paired_statistics(metrics)
    metrics.to_csv(output_dir / "usable_scaffold_metrics.csv", index=False)
    summary.to_csv(output_dir / "usable_scaffold_summary.csv", index=False)
    paired.to_csv(output_dir / "usable_scaffold_paired_statistics.csv", index=False)
    plot(metrics, output_dir)
    report_columns = [
        "model",
        "scope",
        "n_unique_usable_molecules_mean",
        "n_unique_usable_scaffolds_mean",
        "unique_usable_scaffold_yield_mean",
        "unique_scaffold_fraction_among_usable_mean",
        "top10_usable_scaffold_fraction_mean",
        "samples_to_fixed_scaffold_target_mean",
    ]
    report = [
        "# SemlaFlow Unique Usable Scaffold Analysis",
        "",
        "Murcko scaffold counts exclude usable acyclic molecules with an empty scaffold.",
        "Yields use the original number sampled as denominator.",
        "",
        markdown_table(summary[report_columns]),
        "",
        "## Paired statistics",
        "",
        markdown_table(paired),
        "",
    ]
    (output_dir / "usable_scaffold_report.md").write_text("\n".join(report))
    print(summary[report_columns].to_string(index=False))
    print(f"Wrote usable scaffold analysis to {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OBJECTIVES = {
    "reactive": {
        "label": "Reactive motifs",
        "primary": "reactive_hit_fraction",
        "full": "guacamol_transformer_reactive_removal",
        "last_block": "guacamol_transformer_reactive_lastblock_final",
    },
    "chelator": {
        "label": "Metal-binding motifs",
        "primary": "chelator_hit_fraction",
        "full": "guacamol_transformer_chelator_removal",
        "last_block": "guacamol_transformer_chelator_lastblock_final",
    },
}

LAMBDA_RE = re.compile(r"lambda_(\d+(?:\.\d+)?)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/objectives")
    parser.add_argument("--output-dir", default="results/liability_summary/scope_ablation")
    return parser.parse_args()


def neon_lambda(model: str) -> float | None:
    if not model.startswith("neon"):
        return None
    match = LAMBDA_RE.search(model)
    if match is None:
        return None
    return float(match.group(1))


def load_scope_rows(results_root: Path) -> pd.DataFrame:
    rows = []
    for objective, spec in OBJECTIVES.items():
        for scope, run_name in [("full_model", spec["full"]), ("last_block_output", spec["last_block"])]:
            path = results_root / run_name / "aggregate_summary.csv"
            if not path.exists():
                continue
            summary = pd.read_csv(path)
            primary = f"{spec['primary']}_mean"
            base = summary[summary["model"].eq("base")].iloc[0]
            positive = summary[summary["model"].eq("positive")].iloc[0]
            rows.append(
                {
                    "objective": objective,
                    "objective_label": spec["label"],
                    "scope": scope,
                    "model": "positive",
                    "lambda": np.nan,
                    "primary_mean": positive[primary],
                    "primary_relative_reduction": (base[primary] - positive[primary]) / base[primary],
                    "valid_fraction_mean": positive["valid_fraction_mean"],
                    "validity_delta": positive["valid_fraction_mean"] - base["valid_fraction_mean"],
                    "liability_hit_fraction_mean": positive.get("liability_hit_fraction_mean", np.nan),
                    "base_primary_mean": base[primary],
                    "base_valid_fraction_mean": base["valid_fraction_mean"],
                }
            )
            for _, row in summary.iterrows():
                lam = neon_lambda(str(row["model"]))
                if lam is None:
                    continue
                rows.append(
                    {
                        "objective": objective,
                        "objective_label": spec["label"],
                        "scope": scope,
                        "model": row["model"],
                        "lambda": lam,
                        "primary_mean": row[primary],
                        "primary_relative_reduction": (base[primary] - row[primary]) / base[primary],
                        "valid_fraction_mean": row["valid_fraction_mean"],
                        "validity_delta": row["valid_fraction_mean"] - base["valid_fraction_mean"],
                        "liability_hit_fraction_mean": row.get("liability_hit_fraction_mean", np.nan),
                        "base_primary_mean": base[primary],
                        "base_valid_fraction_mean": base["valid_fraction_mean"],
                    }
                )
    return pd.DataFrame(rows)


def plot_scope(rows: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {"full_model": "#E45756", "last_block_output": "#54A24B", "positive": "#4C78A8"}
    for ax, (objective, spec) in zip(axes, OBJECTIVES.items(), strict=False):
        subset = rows[rows["objective"].eq(objective)].copy()
        for scope in ["full_model", "last_block_output"]:
            part = subset[subset["scope"].eq(scope) & subset["lambda"].notna()].sort_values("lambda")
            ax.plot(
                part["valid_fraction_mean"],
                part["primary_relative_reduction"],
                marker="o",
                linewidth=2,
                label=scope.replace("_", " "),
                color=colors[scope],
            )
            for _, row in part.iterrows():
                ax.text(
                    row["valid_fraction_mean"],
                    row["primary_relative_reduction"] + 0.025,
                    f"{row['lambda']:.2g}",
                    fontsize=8,
                    ha="center",
                )
        positive = subset[subset["model"].eq("positive")].head(1)
        if not positive.empty:
            row = positive.iloc[0]
            ax.scatter(
                row["valid_fraction_mean"],
                row["primary_relative_reduction"],
                marker="s",
                s=80,
                color=colors["positive"],
                edgecolor="black",
                linewidth=0.4,
                label="positive FT",
            )
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_title(spec["label"])
        ax.set_xlabel("Validity")
        ax.set_ylabel("Relative target reduction")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(output_dir / "transformer_scope_ablation.png", dpi=220)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(rows: pd.DataFrame, output_dir: Path) -> None:
    display = rows[
        rows["model"].eq("positive")
        | rows["lambda"].isin([0.25, 0.5, 0.75, 1.0])
    ][
        [
            "objective_label",
            "scope",
            "model",
            "lambda",
            "primary_relative_reduction",
            "valid_fraction_mean",
            "validity_delta",
            "liability_hit_fraction_mean",
        ]
    ].copy()
    report = "# Transformer Scope-Ablation Summary\n\n"
    report += (
        "This uses the existing full-model Transformer NEON runs and compares them "
        "against last-block/output NEON. Full-model NEON can achieve strong target "
        "removal, but the validity cliff is severe. Restricting the update to the "
        "last transformer block and output layer preserves most of the target effect "
        "while keeping validity in a usable range.\n\n"
    )
    report += markdown_table(display)
    report += "\n\nFigure: `transformer_scope_ablation.png`.\n"
    (output_dir / "transformer_scope_ablation_report.md").write_text(report)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_scope_rows(Path(args.results_root))
    rows.to_csv(output_dir / "transformer_scope_ablation_metrics.csv", index=False)
    plot_scope(rows, output_dir)
    write_report(rows, output_dir)
    print(f"Wrote scope ablation analysis to {output_dir}")


if __name__ == "__main__":
    main()

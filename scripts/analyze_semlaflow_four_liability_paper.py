#!/usr/bin/env python3
"""Create paper statistics and a consolidated figure for the SemlaFlow case study."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MODEL_ORDER = [
    "base",
    "full_model_random_neon_lambda_2p5",
    "positive_tuned",
    "full_model_neon_lambda_2p5",
    "full_model_norm_matched_random_corrected_neon_lambda_2p5",
    "full_model_positive_corrected_neon_lambda_2p5",
]
MODEL_LABELS = {
    "base": "Base",
    "full_model_random_neon_lambda_2p5": "Random NE",
    "positive_tuned": "Positive FT",
    "full_model_neon_lambda_2p5": "Standard NE",
    "full_model_norm_matched_random_corrected_neon_lambda_2p5": "Random-corrected NE",
    "full_model_positive_corrected_neon_lambda_2p5": "Positive-corrected NE",
}
MODEL_COLORS = {
    "base": "#7A8089",
    "full_model_random_neon_lambda_2p5": "#A3A3A3",
    "positive_tuned": "#EE9B00",
    "full_model_neon_lambda_2p5": "#1A759F",
    "full_model_norm_matched_random_corrected_neon_lambda_2p5": "#52B69A",
    "full_model_positive_corrected_neon_lambda_2p5": "#2D6A4F",
}
POSE_MODEL_MAP = {
    "base": "base",
    "random_tuned": "random_tuned",
    "positive_tuned": "positive_tuned",
    "bad_tuned": "bad_tuned",
    "standard_ne_2p5": "full_model_neon_lambda_2p5",
    "random_ne_2p5": "full_model_random_neon_lambda_2p5",
    "norm_matched_random_corrected_ne_2p5": "full_model_norm_matched_random_corrected_neon_lambda_2p5",
    "positive_corrected_ne_2p5": "full_model_positive_corrected_neon_lambda_2p5",
    "standard_ne_4": "full_model_neon_lambda_4",
    "random_ne_4": "full_model_random_neon_lambda_4",
    "norm_matched_random_corrected_ne_4": "full_model_norm_matched_random_corrected_neon_lambda_4",
    "positive_corrected_ne_4": "full_model_positive_corrected_neon_lambda_4",
}
FOCAL_MODEL = "full_model_positive_corrected_neon_lambda_2p5"
COMPARATORS = [
    "base",
    "positive_tuned",
    "full_model_norm_matched_random_corrected_neon_lambda_2p5",
]
EFFICIENCY_MODELS = [
    "base",
    "positive_tuned",
    "full_model_norm_matched_random_corrected_neon_lambda_2p5",
    "full_model_positive_corrected_neon_lambda_2p5",
]
PAPER_SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        default="results/external/semlaflow/four_liability_joint_replicates/analysis",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=PAPER_SEEDS)
    return parser.parse_args()


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return pd.Series(adjusted, index=p_values.index)
    order = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return float("nan")
    observed = abs(float(np.mean(differences)))
    if observed == 0:
        return 1.0
    extreme = 0
    total = 2 ** len(differences)
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += permuted >= observed - 1e-15
    return float(extreme / total)


def paired_test(
    frame: pd.DataFrame,
    *,
    metric: str,
    focal: str,
    comparator: str,
) -> dict[str, float | str]:
    wide = frame.pivot(index="seed", columns="model", values=metric)
    paired = wide[[focal, comparator]].dropna()
    differences = (paired[focal] - paired[comparator]).to_numpy(dtype=float)
    n = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    t_result = stats.ttest_rel(paired[focal], paired[comparator]) if n > 1 else None
    shapiro_p = float(stats.shapiro(differences).pvalue) if 3 <= n <= 5000 else float("nan")
    return {
        "metric": metric,
        "focal_model": focal,
        "comparator_model": comparator,
        "n_pairs": n,
        "focal_mean": float(paired[focal].mean()),
        "comparator_mean": float(paired[comparator].mean()),
        "paired_mean_difference": mean,
        "paired_difference_sd": sd,
        "ci95_low": mean - critical * sem,
        "ci95_high": mean + critical * sem,
        "cohen_dz": mean / sd if sd > 0 else float("nan"),
        "paired_t_statistic": float(t_result.statistic) if t_result else float("nan"),
        "paired_t_p": float(t_result.pvalue) if t_result else float("nan"),
        "exact_sign_flip_p": exact_sign_flip_p(differences),
        "difference_shapiro_p": shapiro_p,
    }


def load_tables(
    analysis_dir: Path,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(analysis_dir / "replicate_metrics.csv")
    pose = pd.read_csv(analysis_dir / "posebusters_replicate_metrics.csv")
    pose["model"] = pose["model"].map(POSE_MODEL_MAP).fillna(pose["model"])
    diversity = pd.read_csv(analysis_dir / "diversity" / "diversity_metrics.csv")
    distance = pd.read_csv(
        analysis_dir / "distribution_distance_fcd" / "distribution_distance_metrics.csv"
    )
    if len(distance) and not distance["fcd"].notna().all():
        raise ValueError("FCD table contains missing values; rerun distribution-distance analysis")
    selected = set(seeds)
    metrics = metrics[metrics["seed"].isin(selected)].copy()
    pose = pose[pose["seed"].isin(selected)].copy()
    diversity = diversity[diversity["seed"].isin(selected)].copy()
    distance = distance[distance["seed"].isin(selected)].copy()
    return metrics, pose, diversity, distance


def statistics_table(metrics: pd.DataFrame, pose: pd.DataFrame) -> pd.DataFrame:
    endpoint_sources = {
        "four_liability_hit_fraction": metrics,
        "usable_yield": metrics,
        "posebusters_all_checks_yield": pose,
        "usable_3d_yield": pose,
    }
    rows = []
    for metric, frame in endpoint_sources.items():
        for comparator in COMPARATORS:
            rows.append(
                paired_test(
                    frame,
                    metric=metric,
                    focal=FOCAL_MODEL,
                    comparator=comparator,
                )
            )
    output = pd.DataFrame(rows)
    output["paired_t_p_holm"] = output.groupby("metric", group_keys=False)[
        "paired_t_p"
    ].apply(holm_adjust)
    output["exact_sign_flip_p_holm"] = output.groupby("metric", group_keys=False)[
        "exact_sign_flip_p"
    ].apply(holm_adjust)
    return output


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(colors="black", width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def boxplot(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str, *, percent: bool) -> None:
    data = []
    positions = []
    colors = []
    for position, model in enumerate(MODEL_ORDER):
        values = frame.loc[frame["model"].eq(model), metric].dropna().to_numpy(dtype=float)
        if percent:
            values = 100.0 * values
        data.append(values)
        positions.append(position)
        colors.append(MODEL_COLORS[model])
    artists = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    for patch, color in zip(artists["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    ax.set_xticks(positions, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    style_axis(ax)


def mean_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().to_numpy(dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan")
    sem = float(stats.sem(values))
    return mean, float(stats.t.ppf(0.975, len(values) - 1) * sem)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def create_figure(
    metrics: pd.DataFrame,
    pose: pd.DataFrame,
    diversity: pd.DataFrame,
    distance: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)

    boxplot(
        axes[0, 0], metrics, "four_liability_hit_fraction", "Liability-hit fraction (%)", percent=True
    )
    boxplot(
        axes[0, 1], pose, "posebusters_all_checks_yield", "PoseBusters pass yield (%)", percent=True
    )
    boxplot(axes[0, 2], pose, "usable_3d_yield", "3D usable yield (%)", percent=True)
    boxplot(
        axes[1, 0], diversity, "unique_scaffold_fraction", "Unique scaffolds / valid molecules (%)", percent=True
    )
    boxplot(
        axes[1, 1],
        diversity,
        "pairwise_tanimoto_distance_mean",
        "Mean pairwise Tanimoto distance",
        percent=False,
    )

    ax = axes[1, 2]
    shift_models = [
        "base",
        "positive_tuned",
        "full_model_neon_lambda_2p5",
        "full_model_norm_matched_random_corrected_neon_lambda_2p5",
        "full_model_positive_corrected_neon_lambda_2p5",
    ]
    references = ["base", "liability_free_base"]
    for model in shift_models:
        means = []
        errors = []
        for reference in references:
            values = distance.loc[
                distance["model"].eq(model) & distance["reference"].eq(reference), "fcd"
            ]
            mean, error = mean_ci(values)
            means.append(mean)
            errors.append(error)
        ax.errorbar(
            [0, 1],
            means,
            yerr=errors,
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xticks([0, 1], ["Raw base", "Liability-free base"])
    ax.set_ylabel("FCD to reference")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)

    for label, ax in zip("abcdef", axes.flat, strict=True):
        add_panel_label(ax, label)

    figure_base = output_dir / "semlaflow_four_liability_consolidated"
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(figure_base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def sampling_efficiency(
    metrics: pd.DataFrame,
    pose: pd.DataFrame,
    *,
    target_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    two_d = metrics[["seed", "model", "usable_yield"]]
    three_d = pose[["seed", "model", "usable_3d_yield"]]
    frame = two_d.merge(three_d, on=["seed", "model"], how="inner")
    frame = frame[frame["model"].isin(EFFICIENCY_MODELS)].copy()
    frame["target_candidates"] = target_candidates
    frame["samples_required_2d"] = target_candidates / frame["usable_yield"]
    frame["samples_required_3d"] = target_candidates / frame["usable_3d_yield"]

    for endpoint in ("2d", "3d"):
        required = f"samples_required_{endpoint}"
        wide = frame.pivot(index="seed", columns="model", values=required)
        frame = frame.merge(
            wide["base"].rename(f"base_{required}"), left_on="seed", right_index=True
        )
        frame = frame.merge(
            wide["positive_tuned"].rename(f"positive_ft_{required}"),
            left_on="seed",
            right_index=True,
        )
        frame[f"fraction_saved_vs_base_{endpoint}"] = (
            1.0 - frame[required] / frame[f"base_{required}"]
        )
        frame[f"fraction_saved_vs_positive_ft_{endpoint}"] = (
            1.0 - frame[required] / frame[f"positive_ft_{required}"]
        )

    numeric = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in {"seed", "target_candidates"}
    ]
    summary = frame.groupby("model", sort=False)[numeric].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    for endpoint in ("2d", "3d"):
        std_col = f"samples_required_{endpoint}_std"
        count_col = f"samples_required_{endpoint}_count"
        summary[f"samples_required_{endpoint}_ci95"] = (
            stats.t.ppf(0.975, summary[count_col] - 1)
            * summary[std_col]
            / np.sqrt(summary[count_col])
        )
    return frame, summary


def create_sampling_efficiency_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    x = np.arange(len(EFFICIENCY_MODELS), dtype=float)
    width = 0.34
    summary = summary.set_index("model").loc[EFFICIENCY_MODELS]
    specifications = [
        ("2d", -width / 2, "Valid, unique, liability-free"),
        ("3d", width / 2, "+ PoseBusters pass"),
    ]
    for endpoint, offset, label in specifications:
        means = summary[f"samples_required_{endpoint}_mean"].to_numpy()
        errors = summary[f"samples_required_{endpoint}_ci95"].to_numpy()
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=2,
            color=[MODEL_COLORS[model] for model in EFFICIENCY_MODELS],
            edgecolor="black",
            linewidth=0.7,
            alpha=1.0 if endpoint == "2d" else 0.55,
            label=label,
        )
        if endpoint == "3d":
            for bar in bars:
                bar.set_hatch("//")
    ax.axhline(100000, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, [MODEL_LABELS[model] for model in EFFICIENCY_MODELS], rotation=20, ha="right")
    ax.set_ylabel("Molecules sampled for 100,000 usable candidates")
    ax.legend(frameon=False)
    style_axis(ax)
    figure_base = output_dir / "semlaflow_sampling_efficiency_100k"
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(figure_base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(
    statistics_frame: pd.DataFrame,
    efficiency_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    display = statistics_frame.copy()
    for column in [
        "focal_mean",
        "comparator_mean",
        "paired_mean_difference",
        "ci95_low",
        "ci95_high",
        "cohen_dz",
        "paired_t_p_holm",
        "exact_sign_flip_p_holm",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    efficiency_display = efficiency_summary.copy()
    efficiency_columns = [
        "model",
        "samples_required_2d_mean",
        "samples_required_3d_mean",
        "fraction_saved_vs_base_2d_mean",
        "fraction_saved_vs_positive_ft_2d_mean",
        "fraction_saved_vs_base_3d_mean",
        "fraction_saved_vs_positive_ft_3d_mean",
    ]
    efficiency_display = efficiency_display[efficiency_columns]
    efficiency_header = "| " + " | ".join(efficiency_display.columns) + " |"
    efficiency_separator = "| " + " | ".join("---" for _ in efficiency_display.columns) + " |"
    efficiency_rows = [
        "| " + " | ".join(
            f"{value:.6g}" if isinstance(value, float) else str(value) for value in row
        ) + " |"
        for row in efficiency_display.itertuples(index=False, name=None)
    ]
    lines = [
        "# SemlaFlow Four-Liability Paper Analysis",
        "",
        "The focal method is positive-corrected NE at lambda 2.5. Comparisons are paired by seed.",
        "Holm adjustment is applied across the three specified comparisons within each endpoint.",
        "Paired t-tests are primary; exact paired sign-flip tests are reported as a sensitivity analysis.",
        "These development-stage statistics quantify consistency across paired seeds and are not an independent external validation.",
        "",
        header,
        separator,
        *rows,
        "",
        "## Fixed-budget sampling efficiency",
        "",
        "Required sample counts are calculated per seed as target / yield and then summarized.",
        "Relative sample reduction equals relative generation-time reduction only under equal throughput.",
        "Adaptation time and measured generation throughput are not available in the retained artifacts.",
        "",
        efficiency_header,
        efficiency_separator,
        *efficiency_rows,
        "",
    ]
    (output_dir / "paper_analysis_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "paper_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, pose, diversity, distance = load_tables(analysis_dir, args.seeds)
    stats_frame = statistics_table(metrics, pose)
    stats_frame.to_csv(output_dir / "paired_statistics.csv", index=False)

    metrics[metrics["model"].isin(MODEL_ORDER)].to_csv(
        output_dir / "endpoint_2d_plot_data.csv", index=False
    )
    pose[pose["model"].isin(MODEL_ORDER)].to_csv(
        output_dir / "posebusters_plot_data.csv", index=False
    )
    diversity[diversity["model"].isin(MODEL_ORDER)].to_csv(
        output_dir / "diversity_plot_data.csv", index=False
    )
    distance[
        distance["model"].isin(MODEL_ORDER)
        & distance["reference"].isin(["base", "liability_free_base"])
    ].to_csv(output_dir / "fcd_plot_data.csv", index=False)

    efficiency_metrics, efficiency_summary = sampling_efficiency(
        metrics, pose, target_candidates=100000
    )
    efficiency_metrics.to_csv(output_dir / "sampling_efficiency_100k_metrics.csv", index=False)
    efficiency_summary.to_csv(output_dir / "sampling_efficiency_100k_summary.csv", index=False)

    create_figure(metrics, pose, diversity, distance, output_dir)
    create_sampling_efficiency_figure(efficiency_summary, output_dir)
    write_report(stats_frame, efficiency_summary, output_dir)
    print(stats_frame.to_string(index=False))
    print(f"Wrote SemlaFlow paper analysis to {output_dir}")


if __name__ == "__main__":
    main()

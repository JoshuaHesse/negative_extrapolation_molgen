#!/usr/bin/env python3
"""Run the prespecified confirmatory statistics used in the manuscript.

The independently trained seed is the experimental unit. All comparisons are
paired within seed and use an exact two-sided sign-flip test of the mean paired
difference. Confidence intervals describe the paired mean difference and use a
Student-t interval. Holm correction is applied within explicitly named
generator-by-endpoint families.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PAPER_SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

OBJECTIVES = {
    "reactive": {
        "label": "Reactive",
        "metric": "reactive_hit_fraction",
        "rnn_dir": "guacamol_rnn_reactive_removal",
        "transformer_dir": "guacamol_transformer_reactive_lastblock_final",
        "reinvent_dir": "reactive_replicates",
    },
    "chelator": {
        "label": "Metal-binding motif",
        "metric": "chelator_hit_fraction",
        "rnn_dir": "guacamol_rnn_chelator_removal",
        "transformer_dir": "guacamol_transformer_chelator_lastblock_final",
        "reinvent_dir": "chelator_replicates",
    },
    "charged_motif": {
        "label": "Charged motif",
        "metric": "charged_motif_hit_fraction",
        "rnn_dir": "guacamol_rnn_charged_motif_removal",
        "transformer_dir": "guacamol_transformer_charged_motif_lastblock_final",
        "reinvent_dir": "charged_motif_replicates",
    },
    "assay_interference": {
        "label": "Assay interference",
        "metric": "assay_interference_hit_fraction",
        "rnn_dir": "guacamol_rnn_assay_interference_removal",
        "transformer_dir": "guacamol_transformer_assay_interference_lastblock_final",
        "reinvent_dir": "assay_interference_replicates",
    },
}

GENERATOR_CONFIG = {
    "GuacaMol RNN": {
        "root": "results/objectives",
        "dir_key": "rnn_dir",
        "reference": "results/guacamol_rnn_base/reference_canonical_smiles.smi",
        "models": {
            "base": "base",
            "random_ne": "random_neon_lambda_1.0",
            "positive": "positive",
            "ne": "neon_lambda_1.0",
        },
    },
    "GuacaMol Transformer": {
        "root": "results/objectives",
        "dir_key": "transformer_dir",
        "reference": "results/guacamol_transformer_base/reference_canonical_smiles.smi",
        "models": {
            "base": "base",
            "random_ne": "random_neon_last_block_output_lambda_1.0",
            "positive": "positive",
            "ne": "neon_last_block_output_lambda_1.0",
        },
    },
    "REINVENT prior": {
        "root": "results/external/reinvent4",
        "dir_key": "reinvent_dir",
        "reference": None,
        "models": {
            "base": "base",
            "random_ne": "random_neon_lambda_1",
            "positive": "positive",
            "ne": "neon_lambda_1",
        },
    },
}

COMPARATORS = {
    "base": "Base",
    "random_ne": "Random NE",
    "positive": "Positive FT",
}

SEMLAFLOW_MODELS = {
    "base": "base",
    "random_ne": "full_model_random_neon_lambda_2p5",
    "positive": "positive_tuned",
    "standard_ne": "full_model_neon_lambda_2p5",
    "random_corrected_ne": "full_model_norm_matched_random_corrected_neon_lambda_2p5",
    "positive_corrected_ne": "full_model_positive_corrected_neon_lambda_2p5",
}

SEMLAFLOW_COMPARATORS = {
    "base": "Base",
    "random_ne": "Random NE",
    "positive": "Positive FT",
    "standard_ne": "Standard NE",
    "random_corrected_ne": "Random-corrected NE",
}

OBJECTIVE_ORDER = ["Reactive", "Metal-binding motif", "Charged motif", "Assay interference"]
GENERATOR_ORDER = ["GuacaMol RNN", "GuacaMol Transformer", "REINVENT prior"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/paper_statistics")
    )
    parser.add_argument(
        "--paper-table-dir",
        type=Path,
        default=Path("results/publication/tables"),
    )
    parser.add_argument(
        "--paper-figure-dir",
        type=Path,
        default=Path("results/publication/figures"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def validate_seed_rows(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    selected = frame[frame["seed"].isin(PAPER_SEEDS)].copy()
    observed = sorted(selected["seed"].astype(int).unique())
    if observed != PAPER_SEEDS:
        raise ValueError(f"{context}: expected seeds {PAPER_SEEDS}, found {observed}")
    if selected["seed"].duplicated().any():
        duplicates = selected.loc[selected["seed"].duplicated(False), "seed"].tolist()
        raise ValueError(f"{context}: duplicate seed rows {duplicates}")
    return selected


def load_reference(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        return {line.strip() for line in handle if line.strip()}


def novelty_aware_usable_yield(
    sample_path: Path,
    *,
    liability_column: str,
    reference: set[str],
) -> float:
    required = ["canonical_smiles", "valid", liability_column]
    frame = pd.read_csv(sample_path, usecols=lambda column: column in required)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{sample_path}: missing columns {sorted(missing)}")
    valid = frame["valid"].fillna(False).astype(bool)
    liability_free = pd.to_numeric(frame[liability_column], errors="coerce").eq(0)
    canonical = frame["canonical_smiles"].fillna("").astype(str)
    usable = canonical[valid & liability_free]
    usable = usable[usable.ne("") & ~usable.isin(reference)]
    return float(usable.nunique() / len(frame)) if len(frame) else math.nan


def load_liability_endpoints(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    references: dict[str, set[str]] = {}
    for generator, generator_config in GENERATOR_CONFIG.items():
        reference_path = generator_config["reference"]
        if reference_path is not None:
            references[generator] = load_reference(root / str(reference_path))
        for objective, objective_config in OBJECTIVES.items():
            result_dir = (
                root
                / str(generator_config["root"])
                / str(objective_config[str(generator_config["dir_key"])])
            )
            metrics = read_csv(result_dir / "objective_metrics.csv")
            liability_metric = str(objective_config["metric"])
            liability_column = liability_metric.removesuffix("_fraction")
            for method_key, model_name in generator_config["models"].items():
                model_metrics = validate_seed_rows(
                    metrics[metrics["model"].astype(str).eq(model_name)],
                    context=f"{generator}/{objective}/{model_name}",
                )
                for row in model_metrics.itertuples(index=False):
                    seed = int(row.seed)
                    if generator == "REINVENT prior":
                        usable_yield = float(row.usable_yield)
                    else:
                        usable_yield = novelty_aware_usable_yield(
                            result_dir / f"seed_{seed}" / f"{model_name}_samples.csv",
                            liability_column=liability_column,
                            reference=references[generator],
                        )
                    rows.extend(
                        [
                            {
                                "experiment": "liability",
                                "generator": generator,
                                "objective": objective,
                                "objective_label": objective_config["label"],
                                "metric": "target_hit_fraction",
                                "metric_label": "Target liability hit fraction",
                                "endpoint_role": "primary",
                                "better": "lower",
                                "seed": seed,
                                "method_key": method_key,
                                "model": model_name,
                                "value": float(getattr(row, liability_metric)),
                            },
                            {
                                "experiment": "liability",
                                "generator": generator,
                                "objective": objective,
                                "objective_label": objective_config["label"],
                                "metric": "usable_yield",
                                "metric_label": "Usable yield",
                                "endpoint_role": "primary",
                                "better": "higher",
                                "seed": seed,
                                "method_key": method_key,
                                "model": model_name,
                                "value": usable_yield,
                            },
                        ]
                    )
    return pd.DataFrame(rows)


def load_qed_endpoints(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    configurations = {
        "GuacaMol RNN": (
            root / "results/objectives/guacamol_rnn_qed/objective_metrics.csv",
            {
                "base": "base",
                "random_ne": "random_neon_lambda_1.0",
                "positive": "positive",
                "ne": "neon_lambda_1.0",
            },
        ),
        "GuacaMol Transformer": (
            root / "results/objectives/guacamol_transformer_qed/objective_metrics.csv",
            {
                "base": "base",
                "random_ne": "random_neon_lambda_1.0",
                "positive": "positive",
                "ne": "neon_lambda_1.0",
            },
        ),
    }
    metrics_to_load = {
        "qed_ge_0.9_fraction": ("QED >= 0.9 fraction", "primary", "higher"),
        "qed_mean": ("Mean QED", "secondary", "higher"),
    }
    for generator, (path, models) in configurations.items():
        metrics = read_csv(path)
        for method_key, model_name in models.items():
            model_metrics = validate_seed_rows(
                metrics[metrics["model"].astype(str).eq(model_name)],
                context=f"{generator}/QED/{model_name}",
            )
            for _, row in model_metrics.iterrows():
                for metric, (label, role, better) in metrics_to_load.items():
                    rows.append(
                        {
                            "experiment": "qed",
                            "generator": generator,
                            "objective": "qed",
                            "objective_label": "QED",
                            "metric": metric,
                            "metric_label": label,
                            "endpoint_role": role,
                            "better": better,
                            "seed": int(row["seed"]),
                            "method_key": method_key,
                            "model": model_name,
                            "value": float(row[metric]),
                        }
                    )
    return pd.DataFrame(rows)


def load_semlaflow_endpoints(root: Path) -> pd.DataFrame:
    analysis_dir = (
        root
        / "results/external/semlaflow/four_liability_joint_replicates/analysis/paper_analysis"
    )
    endpoints_2d = read_csv(analysis_dir / "endpoint_2d_plot_data.csv")
    posebusters = read_csv(analysis_dir / "posebusters_plot_data.csv")
    rows: list[dict[str, object]] = []
    for method_key, model_name in SEMLAFLOW_MODELS.items():
        model_2d = validate_seed_rows(
            endpoints_2d[endpoints_2d["model"].astype(str).eq(model_name)],
            context=f"SemlaFlow/{model_name}/2D",
        )
        model_3d = validate_seed_rows(
            posebusters[posebusters["model"].astype(str).eq(model_name)],
            context=f"SemlaFlow/{model_name}/3D",
        )
        merged = model_2d.merge(
            model_3d[["seed", "usable_3d_yield"]], on="seed", validate="one_to_one"
        )
        for row in merged.itertuples(index=False):
            rows.extend(
                [
                    {
                        "experiment": "semlaflow",
                        "generator": "SemlaFlow",
                        "objective": "joint_liability",
                        "objective_label": "Joint liability",
                        "metric": "target_hit_fraction",
                        "metric_label": "Joint liability hit fraction",
                        "endpoint_role": "primary",
                        "better": "lower",
                        "seed": int(row.seed),
                        "method_key": method_key,
                        "model": model_name,
                        "value": float(row.four_liability_hit_fraction),
                    },
                    {
                        "experiment": "semlaflow",
                        "generator": "SemlaFlow",
                        "objective": "joint_liability",
                        "objective_label": "Joint liability",
                        "metric": "usable_3d_yield",
                        "metric_label": "3D usable yield",
                        "endpoint_role": "primary",
                        "better": "higher",
                        "seed": int(row.seed),
                        "method_key": method_key,
                        "model": model_name,
                        "value": float(row.usable_3d_yield),
                    },
                ]
            )
    return pd.DataFrame(rows)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    signs = itertools.product((-1.0, 1.0), repeat=len(differences))
    extreme = sum(
        abs(float(np.mean(differences * np.asarray(sign)))) >= observed - 1e-15
        for sign in signs
    )
    return float(extreme / (2 ** len(differences)))


def holm_adjust(values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    running_max = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        running_max = max(running_max, min(1.0, (count - rank) * float(value)))
        adjusted.loc[index] = running_max
    return adjusted


def paired_summary(
    endpoint_values: pd.DataFrame,
    *,
    focal: str,
    comparators: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [
        "experiment",
        "generator",
        "objective",
        "objective_label",
        "metric",
        "metric_label",
        "endpoint_role",
        "better",
    ]
    for keys, group in endpoint_values.groupby(group_columns, observed=True):
        metadata = dict(zip(group_columns, keys, strict=True))
        pivot = group.pivot(index="seed", columns="method_key", values="value")
        if focal not in pivot:
            continue
        for comparator, comparator_label in comparators.items():
            if comparator not in pivot:
                raise ValueError(f"Missing comparator {comparator} for {metadata}")
            paired = pivot[[focal, comparator]].dropna()
            if sorted(paired.index.astype(int)) != PAPER_SEEDS:
                raise ValueError(
                    f"Incomplete pairing for {metadata}, comparator={comparator}: "
                    f"{sorted(paired.index.astype(int))}"
                )
            difference = (paired[focal] - paired[comparator]).to_numpy(dtype=float)
            n_pairs = len(difference)
            mean_difference = float(difference.mean())
            difference_sd = float(difference.std(ddof=1))
            sem = difference_sd / math.sqrt(n_pairs)
            margin = float(stats.t.ppf(0.975, n_pairs - 1) * sem)
            rows.append(
                {
                    **metadata,
                    "focal_method": focal,
                    "comparator": comparator,
                    "comparator_label": comparator_label,
                    "n_pairs": n_pairs,
                    "focal_mean": float(paired[focal].mean()),
                    "focal_sd": float(paired[focal].std(ddof=1)),
                    "comparator_mean": float(paired[comparator].mean()),
                    "comparator_sd": float(paired[comparator].std(ddof=1)),
                    "mean_difference": mean_difference,
                    "difference_sd": difference_sd,
                    "ci95_low": mean_difference - margin,
                    "ci95_high": mean_difference + margin,
                    "exact_sign_flip_p": exact_sign_flip_p(difference),
                    "all_differences_same_direction": bool(
                        np.all(difference > 0) or np.all(difference < 0)
                    ),
                }
            )
    output = pd.DataFrame(rows)
    output["family"] = (
        output["experiment"].astype(str)
        + "__"
        + output["generator"].astype(str)
        + "__"
        + output["metric"].astype(str)
    )
    output["holm_exact_p"] = output.groupby("family", group_keys=False)[
        "exact_sign_flip_p"
    ].apply(holm_adjust)
    return output


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(character, character) for character in text)


def format_p(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.4f}"


def effect_text(
    row: pd.Series, *, scale: float = 100.0, digits: int = 2
) -> str:
    return (
        f"{scale * row.mean_difference:+.{digits}f} "
        f"[{scale * row.ci95_low:+.{digits}f}, "
        f"{scale * row.ci95_high:+.{digits}f}]"
    )


def write_primary_liability_table(statistics: pd.DataFrame, path: Path) -> None:
    subset = statistics[
        statistics["experiment"].eq("liability")
        & statistics["comparator"].eq("positive")
    ].copy()
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Primary paired comparison of NE with positive fine-tuning. Effects are NE minus positive fine-tuning in percentage points with 95\% confidence intervals. Negative target-hit effects and positive usable-yield effects favor NE. Exact two-sided sign-flip p-values are Holm-adjusted within each generator and endpoint across all twelve prespecified comparisons.}",
        r"\label{tab:confirmatory-liability-primary}",
        r"\begin{tabularx}{\textwidth}{ll>{\centering\arraybackslash}X>{\centering\arraybackslash}X}",
        r"\toprule",
        r"Generator & Objective & Target-hit difference; $p_{\mathrm{Holm}}$ & Usable-yield difference; $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for generator in GENERATOR_ORDER:
        generator_rows = subset[subset["generator"].eq(generator)]
        for objective_label in OBJECTIVE_ORDER:
            objective_rows = generator_rows[
                generator_rows["objective_label"].eq(objective_label)
            ]
            target = objective_rows[
                objective_rows["metric"].eq("target_hit_fraction")
            ].iloc[0]
            usable = objective_rows[objective_rows["metric"].eq("usable_yield")].iloc[0]
            lines.append(
                f"{latex_escape(generator)} & {latex_escape(objective_label)} & "
                f"{effect_text(target)}; {format_p(target.holm_exact_p)} & "
                f"{effect_text(usable)}; {format_p(usable.holm_exact_p)} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qed_table(statistics: pd.DataFrame, path: Path) -> None:
    subset = statistics[statistics["experiment"].eq("qed")].copy()
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Paired QED-control comparisons. Effects are NE minus comparator with 95\% confidence intervals. QED $\geq 0.9$ effects are percentage points; mean-QED effects are in QED units. Exact two-sided sign-flip p-values are Holm-adjusted separately by architecture and endpoint.}",
        r"\label{tab:confirmatory-qed}",
        r"\begin{tabularx}{\textwidth}{ll>{\centering\arraybackslash}X>{\centering\arraybackslash}X}",
        r"\toprule",
        r"Architecture & Comparator & QED $\geq 0.9$ difference; $p_{\mathrm{Holm}}$ & Mean-QED difference; $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for generator in ["GuacaMol RNN", "GuacaMol Transformer"]:
        for comparator in COMPARATORS:
            rows = subset[
                subset["generator"].eq(generator)
                & subset["comparator"].eq(comparator)
            ]
            high = rows[rows["metric"].eq("qed_ge_0.9_fraction")].iloc[0]
            mean = rows[rows["metric"].eq("qed_mean")].iloc[0]
            lines.append(
                f"{latex_escape(generator)} & {latex_escape(COMPARATORS[comparator])} & "
                f"{effect_text(high)}; {format_p(high.holm_exact_p)} & "
                f"{effect_text(mean, scale=1.0, digits=3)}; "
                f"{format_p(mean.holm_exact_p)} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_semlaflow_table(statistics: pd.DataFrame, path: Path) -> None:
    subset = statistics[statistics["experiment"].eq("semlaflow")].copy()
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Primary paired SemlaFlow comparisons for positive-corrected NE at $\lambda=2.5$. Effects are positive-corrected NE minus comparator in percentage points with 95\% confidence intervals. Negative liability-hit and positive 3D-usable-yield effects favor positive-corrected NE. Exact two-sided sign-flip p-values are Holm-adjusted separately for each endpoint.}",
        r"\label{tab:confirmatory-semlaflow}",
        r"\begin{tabularx}{\textwidth}{l>{\centering\arraybackslash}X>{\centering\arraybackslash}X}",
        r"\toprule",
        r"Comparator & Liability-hit difference; $p_{\mathrm{Holm}}$ & 3D usable-yield difference; $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for comparator, label in SEMLAFLOW_COMPARATORS.items():
        rows = subset[subset["comparator"].eq(comparator)]
        target = rows[rows["metric"].eq("target_hit_fraction")].iloc[0]
        usable = rows[rows["metric"].eq("usable_3d_yield")].iloc[0]
        lines.append(
            f"{latex_escape(label)} & {effect_text(target)}; "
            f"{format_p(target.holm_exact_p)} & {effect_text(usable)}; "
            f"{format_p(usable.holm_exact_p)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(statistics: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Confirmatory statistical analysis",
        "",
        "- Experimental unit: independently trained seed (n=10 paired seeds).",
        "- Test: exact two-sided paired sign-flip test of the mean difference.",
        "- Interval: 95% Student-t confidence interval for the paired mean difference.",
        "- Multiplicity: Holm correction within each generator-by-endpoint family.",
        "- Development seed 11 and exploratory lambda/scope/epoch analyses are excluded.",
        "",
        "## Liability erasure: NE versus positive fine-tuning",
        "",
        "Effects below are percentage-point differences (NE minus positive FT).",
        "",
        "| Generator | Objective | Target-hit difference [95% CI] | Holm p | Usable-yield difference [95% CI] | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    liability = statistics[
        statistics["experiment"].eq("liability")
        & statistics["comparator"].eq("positive")
    ]
    for generator in GENERATOR_ORDER:
        for objective in OBJECTIVE_ORDER:
            rows = liability[
                liability["generator"].eq(generator)
                & liability["objective_label"].eq(objective)
            ]
            target = rows[rows["metric"].eq("target_hit_fraction")].iloc[0]
            usable = rows[rows["metric"].eq("usable_yield")].iloc[0]
            lines.append(
                f"| {generator} | {objective} | {effect_text(target)} | "
                f"{target.holm_exact_p:.4f} | {effect_text(usable)} | "
                f"{usable.holm_exact_p:.4f} |"
            )
    lines.extend(
        [
            "",
            "## SemlaFlow positive-corrected NE",
            "",
            "Effects are percentage-point differences relative to each comparator.",
            "",
            "| Comparator | Liability-hit difference [95% CI] | Holm p | 3D usable-yield difference [95% CI] | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    semlaflow = statistics[statistics["experiment"].eq("semlaflow")]
    for comparator, label in SEMLAFLOW_COMPARATORS.items():
        rows = semlaflow[semlaflow["comparator"].eq(comparator)]
        target = rows[rows["metric"].eq("target_hit_fraction")].iloc[0]
        usable = rows[rows["metric"].eq("usable_3d_yield")].iloc[0]
        lines.append(
            f"| {label} | {effect_text(target)} | {target.holm_exact_p:.4f} | "
            f"{effect_text(usable)} | {usable.holm_exact_p:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / args.output_dir
    table_dir = root / args.paper_table_dir
    figure_dir = root / args.paper_figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("Loading liability endpoints...")
    liability_values = load_liability_endpoints(root)
    print("Loading QED endpoints...")
    qed_values = load_qed_endpoints(root)
    print("Loading SemlaFlow endpoints...")
    semlaflow_values = load_semlaflow_endpoints(root)

    endpoint_values = pd.concat(
        [liability_values, qed_values, semlaflow_values], ignore_index=True
    )
    liability_stats = paired_summary(
        liability_values, focal="ne", comparators=COMPARATORS
    )
    qed_stats = paired_summary(qed_values, focal="ne", comparators=COMPARATORS)
    semlaflow_stats = paired_summary(
        semlaflow_values,
        focal="positive_corrected_ne",
        comparators=SEMLAFLOW_COMPARATORS,
    )
    statistics = pd.concat(
        [liability_stats, qed_stats, semlaflow_stats], ignore_index=True
    )

    endpoint_values.to_csv(output_dir / "confirmatory_endpoint_values.csv", index=False)
    statistics.to_csv(output_dir / "confirmatory_paired_comparisons.csv", index=False)
    statistics[statistics["experiment"].eq("liability")].to_csv(
        output_dir / "liability_paired_comparisons.csv", index=False
    )
    statistics[statistics["experiment"].eq("qed")].to_csv(
        output_dir / "qed_paired_comparisons.csv", index=False
    )
    statistics[statistics["experiment"].eq("semlaflow")].to_csv(
        output_dir / "semlaflow_paired_comparisons.csv", index=False
    )

    write_primary_liability_table(
        statistics, table_dir / "si_confirmatory_liability_primary.tex"
    )
    write_qed_table(statistics, table_dir / "si_confirmatory_qed.tex")
    write_semlaflow_table(statistics, table_dir / "si_confirmatory_semlaflow.tex")
    write_report(statistics, output_dir / "confirmatory_statistics_report.md")
    print(f"Wrote confirmatory statistical analysis to {output_dir}")


if __name__ == "__main__":
    main()

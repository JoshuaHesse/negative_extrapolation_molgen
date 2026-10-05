from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from neon_molgen.paper_data import (
    load_mw_calibration_comparisons,
    load_reference,
    novelty_aware_usable_yield,
    replace_reinvent_base,
)
from neon_molgen.paper_data import (
    replace_reinvent_base_diversity as replace_reinvent_base_diversity,
)

ROOT = Path.cwd()
if not (ROOT / "results").exists() and (ROOT.parent / "results").exists():
    ROOT = ROOT.parent
RESULTS = ROOT / "results"

DEVELOPMENT_SEED = 11
PAPER_SEEDS_10 = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

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

METHOD_LABELS = {
    "base": "Base",
    "bad": "Bad FT",
    "positive": "Positive FT",
    "neon_lambda_0.25": "NE 0.25",
    "neon_lambda_0.5": "NE 0.5",
    "neon_lambda_0.75": "NE 0.75",
    "neon_lambda_1": "NE 1.0",
    "neon_lambda_1.0": "NE 1.0",
    "random_neon_lambda_0.25": "Random NE 0.25",
    "random_neon_lambda_0.5": "Random NE 0.5",
    "random_neon_lambda_0.75": "Random NE 0.75",
    "random_neon_lambda_1": "Random NE 1.0",
    "random_neon_lambda_1.0": "Random NE 1.0",
}


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(path)
    return pd.DataFrame()


def _add_common_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" in out.columns:
        out["method_label"] = out["model"].map(METHOD_LABELS).fillna(out["model"])
    if "arm" in out.columns:
        out["method_label"] = out["arm"].map(METHOD_LABELS).fillna(out["arm"])
    return out


def _filter_paper_seeds(
    df: pd.DataFrame,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Exclude development seeds from any per-seed paper table."""
    if df.empty or "seed" not in df.columns:
        return df.copy()
    selected = PAPER_SEEDS_10 if seeds is None else seeds
    return df[df["seed"].isin(selected)].copy()


def _summarize_filtered_metrics(
    metrics: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Rebuild cached summaries after filtering their underlying seed rows."""
    if metrics.empty:
        return pd.DataFrame()
    numeric_cols = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {*group_cols, "seed"}
    ]
    summary = (
        metrics.groupby(group_cols, dropna=False, observed=True)[numeric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    return summary


def load_guacamol_liability(*, seeds: list[int] | None = None, objectives=None) -> pd.DataFrame:
    """Per-seed custom RNN/Transformer liability results from objective_metrics.csv."""
    rows = []
    references = {}
    seeds = PAPER_SEEDS_10 if seeds is None else seeds
    for objective, meta in OBJECTIVES.items():
        if objectives is not None and objective not in objectives:
            continue
        for arch, dir_key in [("RNN", "rnn_dir"), ("Transformer", "transformer_dir")]:
            path = RESULTS / "objectives" / meta[dir_key] / "objective_metrics.csv"
            df = _filter_paper_seeds(_read_csv(path), seeds)
            if "usable_yield" not in df:
                df["usable_yield"] = np.nan
            missing = df["usable_yield"].isna()
            if missing.any():
                if arch not in references:
                    base = "guacamol_rnn_base" if arch == "RNN" else "guacamol_transformer_base"
                    references[arch] = load_reference(RESULTS / base / "reference_canonical_smiles.smi")
                for index, row in df[missing].iterrows():
                    sample_path = path.parent / f"seed_{int(row['seed'])}" / f"{row['model']}_samples.csv"
                    df.at[index, "usable_yield"] = novelty_aware_usable_yield(
                        sample_path,
                        liability_column=meta["metric"].removesuffix("_fraction"),
                        reference=references[arch],
                    )
            df["objective"] = objective
            df["objective_label"] = meta["label"]
            df["architecture"] = arch
            df["target_metric"] = meta["metric"]
            df["target_hit_fraction"] = df[meta["metric"]]
            rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    data = data[data["seed"].isin(seeds)].copy()
    return _add_common_labels(data)


def load_reinvent_liability(*, seeds: list[int] | None = None, objectives=None) -> pd.DataFrame:
    """Edited results with the independently sampled 10k-attempt base evaluation."""
    rows = []
    seeds = PAPER_SEEDS_10 if seeds is None else seeds
    for objective, meta in OBJECTIVES.items():
        if objectives is not None and objective not in objectives:
            continue
        path = RESULTS / "external/reinvent4" / meta["reinvent_dir"] / "objective_metrics.csv"
        df = _read_csv(path)
        df = replace_reinvent_base(df, ROOT, objective=objective, seeds=seeds)
        df["objective"] = objective
        df["objective_label"] = meta["label"]
        df["target_metric"] = meta["metric"]
        df["target_hit_fraction"] = df[meta["metric"]]
        # REINVENT writes only accepted rows, so valid_fraction is RDKit
        # validity among written rows. This fixed-budget yield is the
        # relevant guardrail against the requested sampling budget.
        df["output_yield_fraction"] = df["n_valid"] / df["n_sampled"]
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    seeds = PAPER_SEEDS_10 if seeds is None else seeds
    data = data[data["seed"].isin(seeds)].copy()
    return _add_common_labels(data)


def load_fcd_mw_calibration(
    *,
    generator: str = "reinvent",
    mode: str = "common_size",
    seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load molecular-weight band calibration for FCD/FDD interpretation.

    Generate with `make paper-fcd-mw-calibration-size-check`. The default uses
    the same query/reference count for every MW band within a generator/seed.
    Draws share a saved pool and may overlap; these are not universal thresholds.
    """
    seeds = PAPER_SEEDS_10 if seeds is None else seeds
    comparisons = load_mw_calibration_comparisons(ROOT, seeds=seeds)
    if generator not in {"rnn", "transformer", "reinvent"} or mode not in {"original", "pair_matched", "common_size"}:
        raise ValueError("Unknown calibration generator or mode")
    metrics = comparisons[comparisons.generator.eq(generator) & comparisons["mode"].eq(mode)].copy()
    summary = _summarize_filtered_metrics(
        metrics,
        ["band", "band_label"],
    )
    return metrics, summary


def load_guacamol_qed(*, seeds: list[int] | None = None) -> pd.DataFrame:
    """Load GuacaMol QED control objective results for RNN and Transformer."""
    rows = []
    for architecture, folder in [
        ("RNN", "guacamol_rnn_qed"),
        ("Transformer", "guacamol_transformer_qed"),
    ]:
        path = RESULTS / "objectives" / folder / "objective_metrics.csv"
        df = _read_csv(path)
        df = df.copy()
        df["architecture"] = architecture
        rows.append(df)
    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    seeds = PAPER_SEEDS_10 if seeds is None else seeds
    if not data.empty:
        data = data[data["seed"].isin(seeds)].copy()
    return _add_common_labels(data)


def summarize_mean_ci(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    """Mean and pointwise 95% Student-t CI for descriptive plotting."""
    summary = (
        df.groupby(group_cols, observed=True)[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n"].clip(lower=1))
    summary["t_critical"] = summary["n"].map(
        lambda n: float(stats.t.ppf(0.975, n - 1)) if n > 1 else np.nan
    )
    summary["ci95"] = summary["t_critical"] * summary["sem"]
    return summary


if __name__ == "__main__":
    guacamol_liability = load_guacamol_liability()
    reinvent_liability = load_reinvent_liability()
    print("GuacaMol liability:", guacamol_liability.shape)
    print("REINVENT liability:", reinvent_liability.shape)

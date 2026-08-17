from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch

from neon_molgen.checkpoint import load_checkpoint, save_checkpoint, save_json
from neon_molgen.data import read_smiles
from neon_molgen.mlflow_utils import (
    log_artifacts,
    log_metrics_from_mapping,
    log_params,
    log_table_metrics,
    start_run,
)
from neon_molgen.model import clone_model, negative_extrapolate, parameter_update_norm
from neon_molgen.sample import sample_smiles
from neon_molgen.scoring import (
    add_reference_scaffold_scores,
    canonicalize_smiles_set,
    scaffold_counts_from_smiles,
    score_smiles,
    summarize_scores,
)
from neon_molgen.train import train_model_with_history

DEFAULT_NEON_SCOPE = {
    "name": "",
    "include": None,
    "exclude": None,
}
RANDOM_CONTROL_REFRESH_DEFINITION = "trained_random_global_scope_l2_v1"


def select_tail(
    scores: pd.DataFrame,
    metric: str,
    tail: str,
    fraction: float,
) -> tuple[pd.DataFrame, float]:
    if metric not in scores.columns:
        raise ValueError(f"Selection metric '{metric}' is not available.")
    if tail not in {"low", "high"}:
        raise ValueError(f"Selection tail must be 'low' or 'high', got '{tail}'.")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"Selection fraction must be between 0 and 1, got {fraction}.")

    quantile = fraction if tail == "low" else 1.0 - fraction
    cutoff = float(scores[metric].quantile(quantile))
    comparison = scores[metric].le(cutoff) if tail == "low" else scores[metric].ge(cutoff)
    return scores[comparison].copy(), cutoff


def save_and_log_training_history(
    config: dict,
    history: list[dict[str, float]],
    *,
    stage: str,
    seed: int,
    output_dir: Path,
) -> Path:
    del config  # Kept in the signature for compatibility with existing call sites.
    frame = pd.DataFrame(history)
    frame.insert(0, "seed", seed)
    frame.insert(1, "stage", stage)
    path = output_dir / f"{stage}_training_history.csv"
    frame.to_csv(path, index=False)
    return path


def score_cached_samples(frame: pd.DataFrame) -> pd.DataFrame:
    if "smiles" not in frame:
        raise ValueError("Cached sample file is missing required 'smiles' column.")
    return score_smiles(frame["smiles"].fillna("").astype(str).tolist())


REFERENCE_SCAFFOLD_METRICS = {
    "reference_scaffold_count",
    "reference_scaffold_fraction",
    "reference_scaffold_hit",
    "common_reference_scaffold_hit",
}


def needs_reference_scaffolds(config: dict) -> bool:
    selection = config.get("selection", {})
    return bool(config.get("reference_scaffolds", {}).get("enabled")) or selection.get(
        "metric"
    ) in REFERENCE_SCAFFOLD_METRICS


def load_or_build_reference_scaffold_counts(
    base_seed_dir: Path,
    reference_canonical_smiles: set[str],
) -> dict[str, int]:
    cache_path = base_seed_dir.parent / "reference_scaffold_counts.csv"
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        return dict(zip(frame["murcko_scaffold"], frame["count"], strict=False))
    counts = scaffold_counts_from_smiles(reference_canonical_smiles)
    frame = pd.DataFrame(
        [
            {"murcko_scaffold": scaffold, "count": int(count)}
            for scaffold, count in counts.most_common()
        ]
    )
    frame.to_csv(cache_path, index=False)
    return dict(counts)


def add_reference_scaffolds_if_needed(
    scores: pd.DataFrame,
    config: dict,
    reference_scaffold_counts: dict[str, int],
) -> pd.DataFrame:
    if not needs_reference_scaffolds(config):
        return scores
    options = config.get("reference_scaffolds", {})
    return add_reference_scaffold_scores(
        scores,
        reference_scaffold_counts,
        common_scaffold_min_count=int(options.get("common_scaffold_min_count", 25)),
    )


def compare_series(series: pd.Series, operator: str, value: float | int | bool) -> pd.Series:
    if operator == "==":
        return series.eq(value)
    if operator == "!=":
        return series.ne(value)
    if operator == ">=":
        return series.ge(value)
    if operator == ">":
        return series.gt(value)
    if operator == "<=":
        return series.le(value)
    if operator == "<":
        return series.lt(value)
    raise ValueError(f"Unsupported threshold operator: {operator}")


def sample_frame(frame: pd.DataFrame, n: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed).copy()


def finalize_selected_sets(
    bad_selected: pd.DataFrame,
    good_selected: pd.DataFrame,
    selection: dict,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    min_smiles = int(selection.get("min_smiles", 0))
    max_smiles = selection.get("max_smiles")
    max_smiles = int(max_smiles) if max_smiles is not None else None
    balance = bool(selection.get("balance", False))

    n_bad_raw = len(bad_selected)
    n_good_raw = len(good_selected)
    if n_bad_raw < min_smiles or n_good_raw < min_smiles:
        raise ValueError(
            "Underpowered objective selection: "
            f"bad={n_bad_raw}, good={n_good_raw}, min_smiles={min_smiles}"
        )

    if balance:
        n_selected = min(n_bad_raw, n_good_raw)
        if max_smiles is not None:
            n_selected = min(n_selected, max_smiles)
        bad_selected = sample_frame(bad_selected, n_selected, seed=seed + 101)
        good_selected = sample_frame(good_selected, n_selected, seed=seed + 102)
    elif max_smiles is not None:
        bad_selected = sample_frame(bad_selected, max_smiles, seed=seed + 101)
        good_selected = sample_frame(good_selected, max_smiles, seed=seed + 102)

    return (
        bad_selected.reset_index(drop=True),
        good_selected.reset_index(drop=True),
        {
            "balance": balance,
            "max_smiles": max_smiles,
            "min_smiles": min_smiles,
            "n_bad_raw": int(n_bad_raw),
            "n_good_raw": int(n_good_raw),
            "n_bad_after_cap": int(len(bad_selected)),
            "n_good_after_cap": int(len(good_selected)),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/objectives/guacamol_rnn_qed_from_base.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--refresh-random-controls",
        action="store_true",
        help=(
            "Retrain the random-data direction, replace norm-matched random-NE rows, "
            "and preserve existing non-random results. If the original bad checkpoint "
            "was removed, reconstruct it deterministically and refresh standard NE too."
        ),
    )
    parser.add_argument(
        "--delete-refreshed-checkpoints",
        action="store_true",
        help=(
            "After refreshed samples and metadata are written, delete transient random "
            "and extrapolated checkpoints. The bad checkpoint is retained as the audit "
            "reference for future norm matching."
        ),
    )
    return parser.parse_args()


def neon_scopes(config: dict) -> list[dict]:
    scopes = config.get("neon", {}).get("parameter_scopes")
    if not scopes:
        return [DEFAULT_NEON_SCOPE]
    normalized = []
    for scope in scopes:
        name = str(scope["name"]).strip()
        if not name:
            raise ValueError("NEON parameter scope names must be non-empty.")
        normalized.append(
            {
                "name": name,
                "include": scope.get("include"),
                "exclude": scope.get("exclude"),
            }
        )
    return normalized


def scoped_model_name(prefix: str, scope: dict, scale: float) -> str:
    scope_name = scope.get("name", "")
    if scope_name:
        return f"{prefix}_{scope_name}_lambda_{scale}"
    return f"{prefix}_lambda_{scale}"


def discover_seed_dirs(base_results_dir: Path, seeds: list[int] | None) -> list[tuple[int, Path]]:
    if seeds:
        pairs = [(int(seed), base_results_dir / f"seed_{int(seed)}") for seed in seeds]
    else:
        pairs = [
            (int(path.name.removeprefix("seed_")), path)
            for path in sorted(base_results_dir.glob("seed_*"))
            if path.is_dir()
        ]
    missing = [str(path) for _, path in pairs if not (path / "base.pt").exists()]
    if missing:
        raise FileNotFoundError("Missing base checkpoint(s):\n" + "\n".join(missing))
    return pairs


def load_reference_canonical_smiles(base_seed_dir: Path, base_config: dict) -> set[str]:
    candidate_paths = [
        Path(base_config.get("reference_canonical_smiles_path", "")),
        base_seed_dir.parent / "reference_canonical_smiles.smi",
    ]
    for path in candidate_paths:
        if str(path) and path.exists():
            return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    train_smiles = read_smiles(
        base_config["data"]["train_smiles"],
        limit=base_config["data"].get("limit"),
    )
    return canonicalize_smiles_set(train_smiles)


def select_objective_sets(
    valid_scores: pd.DataFrame,
    selection: dict,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict]:
    selection_mode = selection.get("mode", "tail")
    metric = selection.get("metric", "bad_score")
    metadata: dict = {"mode": selection_mode, "metric": metric}
    forced_random_selected = None

    if selection_mode == "binary":
        bad_column = selection["bad_where"]
        good_column = selection.get("good_where", bad_column)
        if bad_column not in valid_scores:
            raise ValueError(f"Binary bad_where column '{bad_column}' not present in scores.")
        if good_column not in valid_scores:
            raise ValueError(f"Binary good_where column '{good_column}' not present in scores.")
        bad_value = selection.get("bad_value", 1.0)
        good_value = selection.get("good_value", 0.0)
        bad_selected = valid_scores[valid_scores[bad_column].eq(bad_value)].copy()
        good_selected = valid_scores[valid_scores[good_column].eq(good_value)].copy()
        bad_selected, good_selected, cap_metadata = finalize_selected_sets(
            bad_selected,
            good_selected,
            {**selection, "balance": selection.get("balance", True)},
            seed=seed,
        )
        metadata.update(
            {
                "bad_where": bad_column,
                "bad_value": bad_value,
                "good_where": good_column,
                "good_value": good_value,
                **cap_metadata,
            }
        )
    elif selection_mode == "threshold":
        metric = selection["metric"]
        if metric not in valid_scores:
            raise ValueError(f"Threshold metric '{metric}' not present in scores.")
        bad_operator = selection["bad_operator"]
        good_operator = selection["good_operator"]
        bad_value = selection["bad_value"]
        good_value = selection["good_value"]
        bad_selected = valid_scores[
            compare_series(valid_scores[metric], bad_operator, bad_value)
        ].copy()
        good_selected = valid_scores[
            compare_series(valid_scores[metric], good_operator, good_value)
        ].copy()
        bad_selected, good_selected, cap_metadata = finalize_selected_sets(
            bad_selected,
            good_selected,
            {**selection, "balance": selection.get("balance", True)},
            seed=seed,
        )
        metadata.update(
            {
                "bad_operator": bad_operator,
                "bad_value": bad_value,
                "good_operator": good_operator,
                "good_value": good_value,
                **cap_metadata,
            }
        )
    elif "fraction" in selection:
        bad_tail = selection.get("bad_tail", "high")
        good_tail = selection.get("good_tail", "low")
        fraction = float(selection["fraction"])
        bad_selected, bad_cutoff = select_tail(valid_scores, metric, bad_tail, fraction)
        good_selected, good_cutoff = select_tail(valid_scores, metric, good_tail, fraction)
        bad_selected, good_selected, cap_metadata = finalize_selected_sets(
            bad_selected,
            good_selected,
            selection,
            seed=seed,
        )
        metadata.update(
            {
                "bad_tail": bad_tail,
                "good_tail": good_tail,
                "fraction": fraction,
                "bad_cutoff": float(bad_cutoff),
                "good_cutoff": float(good_cutoff),
                **cap_metadata,
            }
        )
    else:
        bad_tail = "high"
        good_tail = "low"
        bad_quantile = selection.get("bad_score_high_quantile", selection.get("bad_quantile", 0.9))
        good_quantile = selection.get("bad_score_low_quantile", selection.get("good_quantile", 0.1))
        metric = selection.get("metric", "bad_score")
        bad_cutoff = float(valid_scores[metric].quantile(bad_quantile))
        good_cutoff = float(valid_scores[metric].quantile(good_quantile))
        bad_selected = valid_scores[valid_scores[metric].ge(bad_cutoff)].copy()
        good_selected = valid_scores[valid_scores[metric].le(good_cutoff)].copy()
        bad_selected, good_selected, cap_metadata = finalize_selected_sets(
            bad_selected,
            good_selected,
            selection,
            seed=seed,
        )
        metadata.update(
            {
                "bad_tail": bad_tail,
                "good_tail": good_tail,
                "bad_quantile": float(bad_quantile),
                "good_quantile": float(good_quantile),
                "bad_cutoff": bad_cutoff,
                "good_cutoff": good_cutoff,
                **cap_metadata,
            }
        )

    return bad_selected, good_selected, forced_random_selected, metadata


def selection_summary_from_frames(
    *,
    metadata: dict,
    valid_scores: pd.DataFrame,
    bad_selected: pd.DataFrame,
    good_selected: pd.DataFrame,
    random_selected: pd.DataFrame,
) -> dict:
    summary = {
        **metadata,
        "n_valid_initial": int(len(valid_scores)),
        "n_bad_smiles": int(len(bad_selected)),
        "n_good_smiles": int(len(good_selected)),
        "n_random_smiles": int(len(random_selected)),
        "n_unique_bad_canonical_smiles": int(bad_selected["canonical_smiles"].nunique()),
        "n_unique_good_canonical_smiles": int(good_selected["canonical_smiles"].nunique()),
        "n_unique_random_canonical_smiles": int(random_selected["canonical_smiles"].nunique())
        if len(random_selected)
        else 0,
    }
    for name, frame in [
        ("bad", bad_selected),
        ("good", good_selected),
        ("random", random_selected),
    ]:
        if len(frame):
            for column in [
                "qed",
                "bad_score",
                "logp",
                "mw",
                "sa_score",
                "alert_hit",
                "liability_hit",
                "flat_lipophilic_score",
                "aromatic_rings",
                "fsp3",
                "reference_scaffold_count",
                "reference_scaffold_fraction",
                "reference_scaffold_hit",
                "common_reference_scaffold_hit",
            ]:
                if column in frame:
                    summary[f"{name}_mean_{column}"] = float(frame[column].mean())
        else:
            for column in [
                "qed",
                "bad_score",
                "logp",
                "mw",
                "sa_score",
                "alert_hit",
                "liability_hit",
                "flat_lipophilic_score",
                "aromatic_rings",
                "fsp3",
                "reference_scaffold_count",
                "reference_scaffold_fraction",
                "reference_scaffold_hit",
                "common_reference_scaffold_hit",
            ]:
                summary[f"{name}_mean_{column}"] = None
    return summary


def log_training_epoch(config: dict, row: dict[str, float], *, stage: str) -> None:
    log_metrics_from_mapping(
        config,
        {
            "loss_mean": row["loss_mean"],
            "n_batches": row["n_batches"],
        },
        prefix=f"train/{stage}/",
        step=int(row["epoch"]),
    )


def liability_free_column_from_config(config: dict) -> str | None:
    selection = config.get("selection", {})
    bad_where = selection.get("bad_where")
    if isinstance(bad_where, str) and bad_where.endswith("_hit"):
        return bad_where
    if config.get("objective", {}).get("name") == "alert_removal":
        return "alert_hit"
    return "liability_hit"


def evaluate_model(
    name: str,
    model,
    tokenizer,
    config: dict,
    *,
    device: str,
    output_dir: Path,
    reference_canonical_smiles: set[str],
    reference_scaffold_counts: dict[str, int],
    run_config: dict,
) -> dict:
    sampling = config["sampling"]
    smiles = sample_smiles(
        model,
        tokenizer,
        n_samples=sampling["eval_samples"],
        max_len=sampling["max_len"],
        temperature=sampling["temperature"],
        batch_size=sampling["batch_size"],
        device=device,
    )
    scores = score_smiles(smiles)
    scores = add_reference_scaffolds_if_needed(
        scores,
        run_config,
        reference_scaffold_counts,
    )
    scores.to_csv(output_dir / f"{name}_samples.csv", index=False)
    return {
        "model": name,
        **summarize_scores(
            scores,
            reference_canonical_smiles=reference_canonical_smiles,
            liability_free_column=liability_free_column_from_config(run_config),
            n_sampled=sampling["eval_samples"],
        ),
    }


def run_seed(
    config: dict,
    *,
    seed: int,
    base_seed_dir: Path,
    output_dir: Path,
    device: str,
    refresh_random_controls: bool = False,
    delete_refreshed_checkpoints: bool = False,
) -> pd.DataFrame:
    seed_output_dir = output_dir / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    objective = config["objective"]["name"]
    run_config = copy.deepcopy(config)
    run_config["seed"] = seed
    run_config["output_dir"] = str(seed_output_dir)

    model, tokenizer, base_config = load_checkpoint(base_seed_dir / "base.pt", device=device)
    reference_canonical_smiles = load_reference_canonical_smiles(base_seed_dir, base_config)
    reference_scaffold_counts = load_or_build_reference_scaffold_counts(
        base_seed_dir,
        reference_canonical_smiles,
    )
    architecture = base_config["model"].get("architecture", "rnn")
    with start_run(run_config, f"objective__{objective}__{architecture}__seed_{seed}"):
        log_params(
            run_config,
            {
                "workflow": "objective_from_base",
                "architecture": architecture,
                "objective": objective,
                "base_seed_dir": str(base_seed_dir),
            },
        )
        initial_path = base_seed_dir / "initial_samples.csv"
        if not initial_path.exists():
            raise FileNotFoundError(f"Missing cached initial sample pool: {initial_path}")
        initial_scores = score_cached_samples(pd.read_csv(initial_path))
        initial_scores = add_reference_scaffolds_if_needed(
            initial_scores,
            run_config,
            reference_scaffold_counts,
        )
        initial_scores.to_csv(seed_output_dir / "initial_samples.csv", index=False)
        valid_scores = initial_scores[initial_scores["valid"]].copy()
        if len(valid_scores) == 0:
            raise ValueError(f"Cached initial pool has zero valid molecules: {initial_path}")

        bad_selected, good_selected, forced_random_selected, selection_metadata = (
            select_objective_sets(valid_scores, config["selection"], seed=seed)
        )
        if len(bad_selected) == 0 or len(good_selected) == 0:
            raise ValueError("Selection produced an empty bad or good fine-tuning set.")

        random_selected = pd.DataFrame()
        if config.get("controls", {}).get("random_bad_set", False):
            if forced_random_selected is not None:
                random_selected = forced_random_selected
            else:
                random_selected = valid_scores.sample(n=len(bad_selected), random_state=seed)

        bad_smiles = bad_selected["smiles"].tolist()
        good_smiles = good_selected["smiles"].tolist()
        random_smiles = random_selected["smiles"].tolist() if len(random_selected) else []

        selection_summary = selection_summary_from_frames(
            metadata=selection_metadata,
            valid_scores=valid_scores,
            bad_selected=bad_selected,
            good_selected=good_selected,
            random_selected=random_selected,
        )
        save_json(seed_output_dir / "selection_summary.json", selection_summary)
        log_metrics_from_mapping(run_config, selection_summary, prefix="selection/")
        print(json.dumps(selection_summary, indent=2))

        (seed_output_dir / "bad_smiles.smi").write_text("\n".join(bad_smiles) + "\n")
        (seed_output_dir / "good_smiles.smi").write_text("\n".join(good_smiles) + "\n")
        if random_smiles:
            (seed_output_dir / "random_smiles.smi").write_text("\n".join(random_smiles) + "\n")

        training = config["training"]
        bad_checkpoint_path = seed_output_dir / "bad.pt"
        refresh_standard_neon = refresh_random_controls and not bad_checkpoint_path.is_file()
        previously_refreshed_standard_neon = False
        if refresh_random_controls:
            for norm_path in seed_output_dir.glob("random_neon_norm_matching*.json"):
                try:
                    norm_metadata = json.loads(norm_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                previously_refreshed_standard_neon |= bool(
                    norm_metadata.get("standard_neon_refreshed", False)
                )
        standard_neon_refreshed_for_run = (
            refresh_standard_neon or previously_refreshed_standard_neon
        )
        if refresh_random_controls and bad_checkpoint_path.is_file():
            bad_model, _, _ = load_checkpoint(bad_checkpoint_path, device=device)
            bad_direction_source = "stored_bad_checkpoint"
            print(f"Reusing original bad checkpoint for norm matching: {bad_checkpoint_path}")
        else:
            bad_direction_source = (
                "deterministically_reconstructed_bad_checkpoint"
                if refresh_random_controls
                else "fresh_bad_training"
            )
            if refresh_random_controls:
                print(
                    "Original bad checkpoint is unavailable; deterministically "
                    "reconstructing the bad-set update from the saved selection."
                )
            bad_model = clone_model(model)
            bad_model, bad_history = train_model_with_history(
                bad_model,
                bad_smiles,
                tokenizer,
                epochs=training["bad_epochs"],
                batch_size=training["batch_size"],
                learning_rate=training["learning_rate"],
                device=device,
                desc=f"{objective} bad seed={seed}",
                seed=seed + 1,
                on_epoch_end=lambda row: log_training_epoch(run_config, row, stage="bad"),
            )
            save_and_log_training_history(
                run_config,
                bad_history,
                stage="bad",
                seed=seed,
                output_dir=seed_output_dir,
            )
            save_checkpoint(bad_checkpoint_path, bad_model, tokenizer, base_config)

        positive_model = None
        if not refresh_random_controls:
            positive_model = clone_model(model)
            positive_model, positive_history = train_model_with_history(
                positive_model,
                good_smiles,
                tokenizer,
                epochs=training["positive_epochs"],
                batch_size=training["batch_size"],
                learning_rate=training["learning_rate"],
                device=device,
                desc=f"{objective} positive seed={seed}",
                seed=seed + 2,
                on_epoch_end=lambda row: log_training_epoch(run_config, row, stage="positive"),
            )
            save_and_log_training_history(
                run_config,
                positive_history,
                stage="positive",
                seed=seed,
                output_dir=seed_output_dir,
            )
            save_checkpoint(seed_output_dir / "positive.pt", positive_model, tokenizer, base_config)

        random_model = None
        if random_smiles:
            random_model = clone_model(model)
            random_model, random_history = train_model_with_history(
                random_model,
                random_smiles,
                tokenizer,
                epochs=training["bad_epochs"],
                batch_size=training["batch_size"],
                learning_rate=training["learning_rate"],
                device=device,
                desc=f"{objective} random seed={seed}",
                seed=seed + 3,
                on_epoch_end=lambda row: log_training_epoch(run_config, row, stage="random"),
            )
            save_and_log_training_history(
                run_config,
                random_history,
                stage="random",
                seed=seed,
                output_dir=seed_output_dir,
            )
            save_checkpoint(seed_output_dir / "random.pt", random_model, tokenizer, base_config)

        sampling_config = {
            **base_config,
            "sampling": {**base_config["sampling"], **config.get("sampling", {})},
        }
        if refresh_random_controls:
            existing_summary_path = seed_output_dir / "summary.csv"
            if not existing_summary_path.is_file():
                raise FileNotFoundError(
                    f"Cannot refresh random controls without {existing_summary_path}."
                )
            existing = pd.read_csv(existing_summary_path)
            keep = ~existing["model"].astype(str).str.startswith("random_neon")
            keep &= existing["model"].ne("random_finetune")
            if refresh_standard_neon:
                keep &= ~existing["model"].astype(str).str.startswith("neon")
            summaries = existing.loc[keep].drop(columns=["seed"], errors="ignore").to_dict("records")
        else:
            base_scores = score_cached_samples(pd.read_csv(base_seed_dir / "base_samples.csv"))
            base_scores = add_reference_scaffolds_if_needed(
                base_scores,
                run_config,
                reference_scaffold_counts,
            )
            base_scores.to_csv(seed_output_dir / "base_samples.csv", index=False)
            summaries = [
                {
                    "model": "base",
                    **summarize_scores(
                        base_scores,
                        reference_canonical_smiles=reference_canonical_smiles,
                        liability_free_column=liability_free_column_from_config(run_config),
                        n_sampled=len(base_scores),
                    ),
                },
                evaluate_model(
                    "positive",
                    positive_model,
                    tokenizer,
                    sampling_config,
                    device=device,
                    output_dir=seed_output_dir,
                    reference_canonical_smiles=reference_canonical_smiles,
                    reference_scaffold_counts=reference_scaffold_counts,
                    run_config=run_config,
                ),
                evaluate_model(
                    "bad",
                    bad_model,
                    tokenizer,
                    sampling_config,
                    device=device,
                    output_dir=seed_output_dir,
                    reference_canonical_smiles=reference_canonical_smiles,
                    reference_scaffold_counts=reference_scaffold_counts,
                    run_config=run_config,
                ),
            ]
        if random_model is not None:
            summaries.append(
                evaluate_model(
                    "random_finetune",
                    random_model,
                    tokenizer,
                    sampling_config,
                    device=device,
                    output_dir=seed_output_dir,
                    reference_canonical_smiles=reference_canonical_smiles,
                    reference_scaffold_counts=reference_scaffold_counts,
                    run_config=run_config,
                )
            )

        for scope in neon_scopes(config):
            if random_model is not None:
                scope_tag = f"_{scope['name']}" if scope["name"] else ""
                bad_update_norm = parameter_update_norm(
                    model,
                    bad_model,
                    include_patterns=scope["include"],
                    exclude_patterns=scope["exclude"],
                )
                random_update_norm = parameter_update_norm(
                    model,
                    random_model,
                    include_patterns=scope["include"],
                    exclude_patterns=scope["exclude"],
                )
                if random_update_norm == 0.0:
                    raise ValueError("Random fine-tuning produced a zero parameter update.")
                save_json(
                    seed_output_dir / f"random_neon_norm_matching{scope_tag}.json",
                    {
                        "definition": "global_scope_l2_v1",
                        "scope": scope["name"] or "full_model",
                        "bad_direction_source": bad_direction_source,
                        "standard_neon_refreshed": standard_neon_refreshed_for_run,
                        "bad_update_norm": bad_update_norm,
                        "random_update_norm": random_update_norm,
                        "random_direction_scale": bad_update_norm / random_update_norm,
                    },
                )
            for scale in config["neon"]["lambda_values"]:
                if not refresh_random_controls or refresh_standard_neon:
                    neon_name = scoped_model_name("neon", scope, scale)
                    neon_model = negative_extrapolate(
                        model,
                        bad_model,
                        float(scale),
                        include_patterns=scope["include"],
                        exclude_patterns=scope["exclude"],
                    )
                    save_checkpoint(seed_output_dir / f"{neon_name}.pt", neon_model, tokenizer, base_config)
                    summaries.append(
                        evaluate_model(
                            neon_name,
                            neon_model,
                            tokenizer,
                            sampling_config,
                            device=device,
                            output_dir=seed_output_dir,
                            reference_canonical_smiles=reference_canonical_smiles,
                            reference_scaffold_counts=reference_scaffold_counts,
                            run_config=run_config,
                        )
                    )
                if random_model is not None:
                    random_neon_name = scoped_model_name("random_neon", scope, scale)
                    random_neon_model = negative_extrapolate(
                        model,
                        random_model,
                        float(scale),
                        include_patterns=scope["include"],
                        exclude_patterns=scope["exclude"],
                        norm_match_to=bad_model,
                    )
                    save_checkpoint(
                        seed_output_dir / f"{random_neon_name}.pt",
                        random_neon_model,
                        tokenizer,
                        base_config,
                    )
                    summaries.append(
                        evaluate_model(
                            random_neon_name,
                            random_neon_model,
                            tokenizer,
                            sampling_config,
                            device=device,
                            output_dir=seed_output_dir,
                            reference_canonical_smiles=reference_canonical_smiles,
                            reference_scaffold_counts=reference_scaffold_counts,
                            run_config=run_config,
                        )
                    )

        summary = pd.DataFrame(summaries)
        summary.insert(0, "seed", seed)
        summary.to_csv(seed_output_dir / "summary.csv", index=False)
        save_json(seed_output_dir / "config.json", run_config)
        log_table_metrics(run_config, summary, row_key="model")
        artifact_paths = [
            seed_output_dir / "summary.csv",
            seed_output_dir / "selection_summary.json",
            seed_output_dir / "config.json",
        ]
        artifact_paths.extend(seed_output_dir.glob("*_samples.csv"))
        artifact_paths.extend(seed_output_dir.glob("*_training_history.csv"))
        log_artifacts(run_config, list(artifact_paths))
        if refresh_random_controls and delete_refreshed_checkpoints:
            transient_paths = [seed_output_dir / "random.pt"]
            transient_paths.extend(seed_output_dir.glob("random_neon*.pt"))
            if standard_neon_refreshed_for_run:
                transient_paths.extend(seed_output_dir.glob("neon*.pt"))
            for path in transient_paths:
                path.unlink(missing_ok=True)
        if refresh_random_controls:
            refresh_marker = seed_output_dir / "random_control_refresh_complete.json"
            save_json(
                refresh_marker,
                {
                    "definition": RANDOM_CONTROL_REFRESH_DEFINITION,
                    "seed": seed,
                    "objective": objective,
                    "standard_neon_refreshed": standard_neon_refreshed_for_run,
                    "deleted_transient_checkpoints": delete_refreshed_checkpoints,
                },
            )
            log_artifacts(run_config, [refresh_marker])
        print(summary.to_string(index=False))
        return summary


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame.reset_index()


def summarize_objective(metrics: pd.DataFrame, output_dir: Path) -> None:
    metrics.to_csv(output_dir / "objective_metrics.csv", index=False)
    numeric_columns = [
        column for column in metrics.select_dtypes(include="number").columns if column != "seed"
    ]
    aggregate = metrics.groupby("model")[numeric_columns].agg(["mean", "std", "count"])
    aggregate = flatten_columns(aggregate)
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)

    delta_metrics = [
        "valid_fraction",
        "unique_fraction",
        "novel_fraction",
        "unique_novel_fraction",
        "usable_yield",
        "n_sampled",
        "n_valid",
        "n_valid_unique",
        "n_valid_unique_novel",
        "n_valid_unique_novel_liability_free",
        "qed_mean",
        "qed_median",
        "qed_ge_0.8_fraction",
        "qed_ge_0.9_fraction",
        "bad_score_mean",
        "logp_le_2_fraction",
        "logp_gt_2_fraction",
        "sa_score_mean",
        "sa_score_p90",
        "aromatic_rings_mean",
        "rings_mean",
        "fsp3_mean",
        "formal_charge_abs_mean",
        "hbd_mean",
        "hba_mean",
        "flat_lipophilic_score_mean",
        "alert_hit_fraction",
        "pains_hit_fraction",
        "brenk_hit_fraction",
        "nih_hit_fraction",
        "zinc_hit_fraction",
        "liability_hit_fraction",
        "reactive_hit_fraction",
        "unstable_hit_fraction",
        "chelator_hit_fraction",
        "charged_motif_hit_fraction",
        "assay_interference_hit_fraction",
        "reference_scaffold_count_mean",
        "reference_scaffold_count_p90",
        "reference_scaffold_hit_fraction",
        "common_reference_scaffold_hit_fraction",
        "novel_reference_scaffold_fraction",
    ]
    delta_metrics = [metric for metric in delta_metrics if metric in metrics.columns]
    base = metrics[metrics["model"].eq("base")][["seed", *delta_metrics]]
    comparable = metrics[~metrics["model"].eq("base")].copy()
    delta = comparable.merge(base, on="seed", suffixes=("", "_base"))
    for metric in delta_metrics:
        delta[f"delta_{metric}"] = delta[metric] - delta[f"{metric}_base"]
    delta.to_csv(output_dir / "delta_metrics.csv", index=False)
    delta_columns = [column for column in delta.columns if column.startswith("delta_")]
    delta_summary = delta.groupby("model")[delta_columns].agg(["mean", "std", "count"])
    delta_summary = flatten_columns(delta_summary)
    delta_summary.to_csv(output_dir / "delta_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    base_results_dir = Path(config["base_results_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config)

    seeds = args.seeds or config.get("seeds") or config.get("replicates", {}).get("seeds")
    seed_dirs = discover_seed_dirs(base_results_dir, seeds)

    summaries = []
    for index, (seed, seed_dir) in enumerate(seed_dirs, start=1):
        seed_output_dir = output_dir / f"seed_{seed}"
        summary_path = seed_output_dir / "summary.csv"
        refresh_marker_path = seed_output_dir / "random_control_refresh_complete.json"
        refresh_is_complete = False
        if refresh_marker_path.is_file():
            marker = json.loads(refresh_marker_path.read_text())
            refresh_is_complete = (
                marker.get("definition") == RANDOM_CONTROL_REFRESH_DEFINITION
            )
        if (
            args.skip_existing
            and summary_path.exists()
            and (not args.refresh_random_controls or refresh_is_complete)
        ):
            print(
                f"[{index}/{len(seed_dirs)}] skipping existing "
                f"objective={config['objective']['name']} seed={seed}"
            )
            summaries.append(pd.read_csv(summary_path))
            continue
        print(f"[{index}/{len(seed_dirs)}] objective={config['objective']['name']} seed={seed}")
        summaries.append(
            run_seed(
                config,
                seed=seed,
                base_seed_dir=seed_dir,
                output_dir=output_dir,
                device=args.device,
                refresh_random_controls=args.refresh_random_controls,
                delete_refreshed_checkpoints=args.delete_refreshed_checkpoints,
            )
        )
    metrics = pd.concat(summaries, ignore_index=True)
    summarize_objective(metrics, output_dir)


if __name__ == "__main__":
    main()

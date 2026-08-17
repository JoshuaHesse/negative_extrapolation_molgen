#!/usr/bin/env python3
"""Development-seed sensitivity analysis for molecular-model fine-tuning length."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neon_molgen.checkpoint import load_checkpoint, save_json
from neon_molgen.model import (
    clone_model,
    matches_any_pattern,
    negative_extrapolate,
    parameter_update_norm,
)
from neon_molgen.scoring import summarize_scores
from neon_molgen.train import train_model_with_history

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_objective_from_base import (  # noqa: E402
    add_reference_scaffolds_if_needed,
    evaluate_model,
    liability_free_column_from_config,
    load_or_build_reference_scaffold_counts,
    load_reference_canonical_smiles,
    neon_scopes,
    score_cached_samples,
    select_objective_sets,
    selection_summary_from_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train bad and positive models continuously, compare raw with "
            "common-norm negative extrapolation in a configured parameter scope, "
            "and track positive fine-tuning."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/objectives/guacamol_rnn_reactive_removal_from_base.json",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--reference-epoch", type=int, default=2)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--eval-samples", type=int, default=5000)
    parser.add_argument(
        "--scope-name",
        default=None,
        help=(
            "Name of a parameter scope from config.neon.parameter_scopes. "
            "The sole configured scope is selected automatically when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/development/guacamol_rnn_reactive_epoch_sensitivity_seed_11",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[int]:
    epochs = sorted(set(args.epochs))
    if not epochs or epochs[0] < 1:
        raise ValueError("--epochs must contain positive integers.")
    if args.reference_epoch not in epochs:
        raise ValueError("--reference-epoch must be included in --epochs.")
    if args.eval_samples < 1:
        raise ValueError("--eval-samples must be positive.")
    return epochs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_parameter_scope(config: dict, requested_name: str | None) -> dict:
    scopes = neon_scopes(config)
    if requested_name is not None:
        matches = [scope for scope in scopes if scope["name"] == requested_name]
        if len(matches) != 1:
            available = [scope["name"] or "full_model" for scope in scopes]
            raise ValueError(
                f"Unknown --scope-name {requested_name!r}; available scopes: {available}"
            )
        return matches[0]
    if len(scopes) != 1:
        available = [scope["name"] or "full_model" for scope in scopes]
        raise ValueError(
            "Multiple parameter scopes are configured; select one with --scope-name. "
            f"Available scopes: {available}"
        )
    return scopes[0]


def clone_state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def model_from_state(
    base_model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> torch.nn.Module:
    model = clone_model(base_model).cpu()
    model.load_state_dict(state)
    model.eval()
    return model


def update_cosine_similarity(
    base_model: torch.nn.Module,
    tuned_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> float:
    base_state = base_model.state_dict()
    tuned_state = tuned_model.state_dict()
    reference_state = reference_model.state_dict()
    dot = 0.0
    tuned_squared = 0.0
    reference_squared = 0.0
    for key, base_value in base_state.items():
        included = include_patterns is None or matches_any_pattern(key, include_patterns)
        excluded = matches_any_pattern(key, exclude_patterns)
        if not torch.is_floating_point(base_value) or not included or excluded:
            continue
        tuned_delta = (tuned_state[key] - base_value).detach().float()
        reference_delta = (reference_state[key] - base_value).detach().float()
        dot += float(torch.sum(tuned_delta * reference_delta).cpu())
        tuned_squared += float(torch.sum(tuned_delta * tuned_delta).cpu())
        reference_squared += float(torch.sum(reference_delta * reference_delta).cpu())
    denominator = math.sqrt(tuned_squared * reference_squared)
    return dot / denominator if denominator else float("nan")


def evaluate_with_fixed_sampling_seed(
    *,
    name: str,
    model: torch.nn.Module,
    tokenizer,
    sampling_config: dict,
    device: str,
    output_dir: Path,
    reference_canonical_smiles: set[str],
    reference_scaffold_counts: dict[str, int],
    run_config: dict,
    sampling_seed: int,
) -> dict:
    seed_everything(sampling_seed)
    result = evaluate_model(
        name,
        model,
        tokenizer,
        sampling_config,
        device=device,
        output_dir=output_dir,
        reference_canonical_smiles=reference_canonical_smiles,
        reference_scaffold_counts=reference_scaffold_counts,
        run_config=run_config,
    )
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_report(
    summary: pd.DataFrame,
    direction_stats: pd.DataFrame,
    history: pd.DataFrame,
    output_dir: Path,
    objective_metric: str,
    architecture: str,
    scope_name: str,
) -> None:
    columns = [
        column
        for column in [
            "model",
            "epoch",
            "direction_kind",
            objective_metric,
            "valid_fraction",
            "usable_yield",
            "unique_fraction",
        ]
        if column in summary
    ]
    lines = [
        f"# {architecture.capitalize()} fine-tuning epoch sensitivity",
        "",
        "This is a development-seed diagnostic. It is not part of the confirmatory seed set.",
        f"Negative extrapolation scope: `{scope_name or 'full_model'}`.",
        "",
        "## Training history",
        "",
        history.to_csv(index=False).strip(),
        "",
        "## Direction diagnostics",
        "",
        direction_stats.to_csv(index=False).strip(),
        "",
        "## Generation endpoints",
        "",
        summary[columns].to_csv(index=False).strip(),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    epochs = validate_args(args)
    config = json.loads(Path(args.config).read_text())
    scope = select_parameter_scope(config, args.scope_name)
    include_patterns = scope.get("include")
    exclude_patterns = scope.get("exclude")
    base_seed_dir = Path(config["base_results_dir"]) / f"seed_{args.seed}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (base_seed_dir / "base.pt").is_file():
        raise FileNotFoundError(base_seed_dir / "base.pt")

    run_config = copy.deepcopy(config)
    run_config["seed"] = args.seed
    run_config["workflow"] = "model_negative_direction_epoch_sensitivity"
    run_config["epoch_sensitivity"] = {
        "epochs": epochs,
        "reference_epoch": args.reference_epoch,
        "lambda_value": args.lambda_value,
        "eval_samples": args.eval_samples,
        "parameter_scope": scope,
        "development_only": True,
    }
    save_json(output_dir / "config.json", run_config)

    seed_everything(args.seed)
    base_model, tokenizer, base_config = load_checkpoint(base_seed_dir / "base.pt", device="cpu")
    architecture = base_config["model"].get("architecture", "rnn")

    reference_canonical_smiles = load_reference_canonical_smiles(base_seed_dir, base_config)
    reference_scaffold_counts = load_or_build_reference_scaffold_counts(
        base_seed_dir,
        reference_canonical_smiles,
    )
    initial_scores = score_cached_samples(pd.read_csv(base_seed_dir / "initial_samples.csv"))
    initial_scores = add_reference_scaffolds_if_needed(
        initial_scores,
        run_config,
        reference_scaffold_counts,
    )
    valid_scores = initial_scores[initial_scores["valid"]].copy()
    bad_selected, good_selected, _, selection_metadata = select_objective_sets(
        valid_scores,
        config["selection"],
        seed=args.seed,
    )
    selection_summary = selection_summary_from_frames(
        metadata=selection_metadata,
        valid_scores=valid_scores,
        bad_selected=bad_selected,
        good_selected=good_selected,
        random_selected=pd.DataFrame(),
    )
    save_json(output_dir / "selection_summary.json", selection_summary)
    (output_dir / "bad_smiles.smi").write_text(
        "\n".join(bad_selected["smiles"].astype(str)) + "\n"
    )
    (output_dir / "good_smiles.smi").write_text(
        "\n".join(good_selected["smiles"].astype(str)) + "\n"
    )

    requested_epochs = set(epochs)
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    tuned_model = clone_model(base_model)

    def capture_epoch(row: dict[str, float]) -> None:
        epoch = int(row["epoch"])
        if epoch in requested_epochs:
            snapshots[epoch] = clone_state_to_cpu(tuned_model)

    training = config["training"]
    trained_bad_model, history_rows = train_model_with_history(
        tuned_model,
        bad_selected["smiles"].astype(str).tolist(),
        tokenizer,
        epochs=max(epochs),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        device=args.device,
        desc=f"{config['objective']['name']} epoch sensitivity seed={args.seed}",
        seed=args.seed + 1,
        on_epoch_end=capture_epoch,
    )
    trained_bad_model.cpu()
    del trained_bad_model
    bad_history = pd.DataFrame(history_rows)
    bad_history.insert(0, "stage", "bad")
    bad_history.to_csv(output_dir / "bad_training_history.csv", index=False)
    if set(snapshots) != requested_epochs:
        raise RuntimeError(f"Missing epoch snapshots: {requested_epochs - set(snapshots)}")

    positive_snapshots: dict[int, dict[str, torch.Tensor]] = {}
    positive_model = clone_model(base_model)

    def capture_positive_epoch(row: dict[str, float]) -> None:
        epoch = int(row["epoch"])
        if epoch in requested_epochs:
            positive_snapshots[epoch] = clone_state_to_cpu(positive_model)

    trained_positive_model, positive_history_rows = train_model_with_history(
        positive_model,
        good_selected["smiles"].astype(str).tolist(),
        tokenizer,
        epochs=max(epochs),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        device=args.device,
        desc=f"{config['objective']['name']} positive sensitivity seed={args.seed}",
        seed=args.seed + 2,
        on_epoch_end=capture_positive_epoch,
    )
    trained_positive_model.cpu()
    del trained_positive_model
    positive_history = pd.DataFrame(positive_history_rows)
    positive_history.insert(0, "stage", "positive")
    positive_history.to_csv(output_dir / "positive_training_history.csv", index=False)
    if set(positive_snapshots) != requested_epochs:
        raise RuntimeError(
            f"Missing positive epoch snapshots: {requested_epochs - set(positive_snapshots)}"
        )
    history = pd.concat([bad_history, positive_history], ignore_index=True)
    history.to_csv(output_dir / "training_history.csv", index=False)

    reference_model = model_from_state(base_model, snapshots[args.reference_epoch])
    reference_norm = parameter_update_norm(
        base_model,
        reference_model,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    direction_rows = []
    epoch_models = {}
    for epoch in epochs:
        epoch_model = model_from_state(base_model, snapshots[epoch])
        direction_norm = parameter_update_norm(
            base_model,
            epoch_model,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        direction_rows.append(
            {
                "epoch": epoch,
                "direction_norm": direction_norm,
                "reference_epoch": args.reference_epoch,
                "reference_direction_norm": reference_norm,
                "norm_matching_scale": reference_norm / direction_norm,
                "cosine_to_reference_direction": update_cosine_similarity(
                    base_model,
                    epoch_model,
                    reference_model,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                ),
                "scope_name": scope["name"] or "full_model",
            }
        )
        epoch_models[epoch] = epoch_model
    direction_stats = pd.DataFrame(direction_rows)
    direction_stats.to_csv(output_dir / "direction_metrics.csv", index=False)

    sampling_config = {
        **base_config,
        "sampling": {
            **base_config["sampling"],
            **config.get("sampling", {}),
            "eval_samples": args.eval_samples,
        },
    }
    liability_column = liability_free_column_from_config(run_config)
    objective_metric = f"{liability_column}_fraction"
    rows = []

    base_scores = score_cached_samples(
        pd.read_csv(base_seed_dir / "base_samples.csv").iloc[: args.eval_samples]
    )
    base_scores = add_reference_scaffolds_if_needed(
        base_scores,
        run_config,
        reference_scaffold_counts,
    )
    base_scores.to_csv(output_dir / "base_samples.csv", index=False)
    rows.append(
        {
            "model": "base",
            "epoch": 0,
            "direction_kind": "base",
            **summarize_scores(
                base_scores,
                reference_canonical_smiles=reference_canonical_smiles,
                liability_free_column=liability_column,
                n_sampled=args.eval_samples,
            ),
        }
    )

    sampling_seed = args.seed + 10_000
    for epoch in epochs:
        bad_model = epoch_models[epoch]
        bad_name = f"bad_epoch_{epoch}"
        bad_result = evaluate_with_fixed_sampling_seed(
            name=bad_name,
            model=bad_model,
            tokenizer=tokenizer,
            sampling_config=sampling_config,
            device=args.device,
            output_dir=output_dir,
            reference_canonical_smiles=reference_canonical_smiles,
            reference_scaffold_counts=reference_scaffold_counts,
            run_config=run_config,
            sampling_seed=sampling_seed,
        )
        rows.append(
            {
                **bad_result,
                "epoch": epoch,
                "direction_kind": "bad_finetune",
            }
        )

        raw_name = f"ne_raw_epoch_{epoch}_lambda_{args.lambda_value:g}"
        raw_ne = negative_extrapolate(
            base_model,
            bad_model,
            args.lambda_value,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        raw_result = evaluate_with_fixed_sampling_seed(
            name=raw_name,
            model=raw_ne,
            tokenizer=tokenizer,
            sampling_config=sampling_config,
            device=args.device,
            output_dir=output_dir,
            reference_canonical_smiles=reference_canonical_smiles,
            reference_scaffold_counts=reference_scaffold_counts,
            run_config=run_config,
            sampling_seed=sampling_seed,
        )
        rows.append(
            {
                **raw_result,
                "epoch": epoch,
                "direction_kind": "raw_ne",
            }
        )

        matched_name = f"ne_norm_matched_epoch_{epoch}_lambda_{args.lambda_value:g}"
        if epoch == args.reference_epoch:
            shutil.copyfile(
                output_dir / f"{raw_name}_samples.csv",
                output_dir / f"{matched_name}_samples.csv",
            )
            matched_result = {**raw_result, "model": matched_name}
        else:
            matched_ne = negative_extrapolate(
                base_model,
                bad_model,
                args.lambda_value,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                norm_match_to=reference_model,
            )
            matched_result = evaluate_with_fixed_sampling_seed(
                name=matched_name,
                model=matched_ne,
                tokenizer=tokenizer,
                sampling_config=sampling_config,
                device=args.device,
                output_dir=output_dir,
                reference_canonical_smiles=reference_canonical_smiles,
                reference_scaffold_counts=reference_scaffold_counts,
                run_config=run_config,
                sampling_seed=sampling_seed,
            )
        rows.append(
            {
                **matched_result,
                "epoch": epoch,
                "direction_kind": "norm_matched_ne",
            }
        )

    for epoch in epochs:
        positive_epoch_model = model_from_state(base_model, positive_snapshots[epoch])
        positive_name = f"positive_epoch_{epoch}"
        positive_result = evaluate_with_fixed_sampling_seed(
            name=positive_name,
            model=positive_epoch_model,
            tokenizer=tokenizer,
            sampling_config=sampling_config,
            device=args.device,
            output_dir=output_dir,
            reference_canonical_smiles=reference_canonical_smiles,
            reference_scaffold_counts=reference_scaffold_counts,
            run_config=run_config,
            sampling_seed=sampling_seed,
        )
        rows.append(
            {
                **positive_result,
                "epoch": epoch,
                "direction_kind": "positive_finetune",
            }
        )

    summary = pd.DataFrame(rows)
    summary.insert(0, "seed", args.seed)
    summary.to_csv(output_dir / "epoch_sensitivity_summary.csv", index=False)
    write_report(
        summary,
        direction_stats,
        history,
        output_dir,
        objective_metric,
        architecture,
        scope["name"],
    )
    print(summary[[
        column
        for column in [
            "model",
            "epoch",
            "direction_kind",
            objective_metric,
            "valid_fraction",
            "usable_yield",
        ]
        if column in summary
    ]].to_string(index=False))
    print(f"Wrote {architecture} epoch sensitivity analysis to {output_dir}")


if __name__ == "__main__":
    main()

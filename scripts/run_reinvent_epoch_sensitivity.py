#!/usr/bin/env python3
"""Evaluate REINVENT transfer-learning duration and NE update magnitude."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from reinvent.models.meta_data import update_model_data

from neon_molgen.scoring import summarize_scores
from scripts.reinvent_negative_extrapolate import (
    global_delta_norm,
    load_checkpoint,
    network_state,
)
from scripts.run_reinvent_reactive_replicates import (
    run_command,
    score_sample_csv,
    write_sampling_config,
    write_tl_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-seed REINVENT reactive-liability epoch-sensitivity "
            "analysis from an existing replicate selection."
        )
    )
    parser.add_argument("--prior", default="external/REINVENT4/priors/reinvent.prior")
    parser.add_argument(
        "--source-dir",
        default="results/external/reinvent4/reactive_replicates/seed_11",
    )
    parser.add_argument(
        "--output-dir",
        default="results/development/reinvent_reactive_epoch_sensitivity_seed_11",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--reference-epoch", type=int, default=2)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--eval-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--keep-models",
        action="store_true",
        help="Retain transfer-learning and extrapolated checkpoints after scoring.",
    )
    return parser.parse_args()


def checkpoint_for_epoch(model_path: Path, epoch: int, max_epoch: int) -> Path:
    checkpoint_path = Path(f"{model_path}.{epoch}.chkpt")
    if checkpoint_path.is_file():
        return checkpoint_path
    if epoch == max_epoch and model_path.is_file():
        return model_path
    raise FileNotFoundError(
        f"REINVENT did not write the expected epoch-{epoch} checkpoint: "
        f"{checkpoint_path}"
    )


def ensure_transfer_series(
    *,
    prior: Path,
    smiles_file: Path,
    output_model: Path,
    label: str,
    output_dir: Path,
    seed: int,
    max_epoch: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    skip_existing: bool,
) -> None:
    expected = [Path(f"{output_model}.{epoch}.chkpt") for epoch in range(1, max_epoch + 1)]
    if skip_existing and all(path.is_file() for path in expected):
        return
    config_path = output_dir / "configs" / f"{label}_tl.toml"
    write_tl_config(
        config_path,
        prior=prior,
        smiles_file=smiles_file,
        output_model_file=output_model,
        tb_logdir=output_dir / f"tb_{label}",
        json_output=output_dir / "models" / f"{label}_tl.json",
        device=device,
        epochs=max_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_every_n_epochs=1,
    )
    run_command(
        [
            "reinvent",
            "--seed",
            str(seed),
            "-l",
            str(output_dir / "models" / f"{label}_tl.log"),
            str(config_path),
        ]
    )


def floating_keys(state: dict[str, torch.Tensor]) -> list[str]:
    return [key for key, value in state.items() if torch.is_tensor(value) and value.is_floating_point()]


def make_ne_checkpoint(
    *,
    base_path: Path,
    bad_path: Path,
    output_path: Path,
    scale: float,
    target_norm: float | None,
    epoch: int,
) -> dict[str, float]:
    base_checkpoint = load_checkpoint(base_path)
    bad_checkpoint = load_checkpoint(bad_path)
    base_state = network_state(base_checkpoint)
    bad_state = network_state(bad_checkpoint)
    if set(base_state) != set(bad_state):
        raise ValueError("REINVENT base and bad checkpoints have different tensor keys.")

    scoped_keys = floating_keys(base_state)
    deltas = {key: bad_state[key] - base_state[key] for key in scoped_keys}
    raw_norm = global_delta_norm(deltas, scoped_keys)
    if raw_norm == 0.0:
        raise ValueError(f"Epoch {epoch} produced a zero bad-set update.")
    norm_scale = 1.0 if target_norm is None else target_norm / raw_norm

    output_checkpoint = load_checkpoint(base_path)
    output_state = network_state(output_checkpoint)
    for key, base_value in base_state.items():
        if key in deltas:
            output_state[key] = base_value - scale * norm_scale * deltas[key]
        else:
            output_state[key] = base_value
    output_checkpoint = update_model_data(
        output_checkpoint,
        comment=(
            f"REINVENT epoch sensitivity: epoch={epoch}, lambda={scale:g}, "
            f"direction_scale={norm_scale:.8g}"
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output_path)
    return {
        "direction_raw_norm": raw_norm,
        "direction_target_norm": raw_norm if target_norm is None else target_norm,
        "direction_scale": norm_scale,
        "direction_applied_norm": raw_norm * norm_scale,
    }


def sample_and_score(
    *,
    model_path: Path,
    model: str,
    output_dir: Path,
    seed: int,
    eval_samples: int,
    device: str,
    skip_existing: bool,
) -> tuple[dict, Path]:
    scored_path = output_dir / f"{model}_samples.csv"
    if skip_existing and scored_path.is_file():
        scores = pd.read_csv(scored_path)
        summary = summarize_scores(
            scores,
            liability_free_column="reactive_hit",
            n_sampled=eval_samples,
        )
        summary["model"] = model
        return summary, scored_path

    raw_path = output_dir / "raw_samples" / f"{model}.csv"
    config_path = output_dir / "configs" / f"sample_{model}.toml"
    write_sampling_config(
        config_path,
        model_file=model_path,
        output_file=raw_path,
        json_output=output_dir / "raw_samples" / f"{model}.json",
        num_smiles=eval_samples,
        device=device,
    )
    run_command(
        [
            "reinvent",
            "--seed",
            str(seed + 1000),
            "-l",
            str(output_dir / "raw_samples" / f"{model}.log"),
            str(config_path),
        ]
    )
    scores, summary = score_sample_csv(
        raw_path,
        model=model,
        n_sampled=eval_samples,
        liability_free_column="reactive_hit",
    )
    scores.to_csv(scored_path, index=False)
    return summary, scored_path


def base_summary(source_dir: Path, output_dir: Path, eval_samples: int) -> dict:
    source_path = source_dir / "baseline" / "prior_samples_scored.csv"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    scores = pd.read_csv(source_path).iloc[:eval_samples].copy()
    scores.insert(0, "model", "base")
    scores.to_csv(output_dir / "base_samples.csv", index=False)
    summary = summarize_scores(
        scores,
        liability_free_column="reactive_hit",
        n_sampled=eval_samples,
    )
    summary["model"] = "base"
    summary["epoch"] = 0
    summary["direction_kind"] = "base"
    return summary


def main() -> None:
    args = parse_args()
    epochs = sorted(set(args.epochs))
    if not epochs or min(epochs) < 1:
        raise ValueError("--epochs must contain positive integers.")
    if args.reference_epoch not in epochs:
        raise ValueError("--reference-epoch must be present in --epochs.")

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    expected_models = [
        f"{kind}_epoch_{epoch}"
        for epoch in epochs
        for kind in ["bad", "positive", "standard_ne", "epoch_norm_ne"]
    ]
    metadata_path = output_dir / "run_metadata.json"
    summary_path = output_dir / "epoch_sensitivity_summary.csv"
    if args.skip_existing and metadata_path.is_file() and summary_path.is_file():
        previous = json.loads(metadata_path.read_text())
        settings_match = (
            previous.get("seed") == args.seed
            and previous.get("epochs") == epochs
            and previous.get("reference_epoch") == args.reference_epoch
            and previous.get("lambda_value") == args.lambda_value
            and previous.get("eval_samples") == args.eval_samples
            and previous.get("batch_size") == args.batch_size
            and previous.get("learning_rate") == args.learning_rate
        )
        samples_complete = (output_dir / "base_samples.csv").is_file() and all(
            (output_dir / f"{model}_samples.csv").is_file()
            for model in expected_models
        )
        if settings_match and samples_complete:
            print(f"REINVENT epoch-sensitivity outputs are already complete in {output_dir}")
            return
    prior = Path(args.prior)
    bad_smiles = source_dir / "selection" / "bad_smiles.smi"
    good_smiles = source_dir / "selection" / "good_smiles.smi"
    for required in [prior, bad_smiles, good_smiles]:
        if not required.is_file():
            raise FileNotFoundError(required)

    max_epoch = max(epochs)
    bad_model = model_dir / "bad_tuned.model"
    positive_model = model_dir / "positive_tuned.model"
    ensure_transfer_series(
        prior=prior,
        smiles_file=bad_smiles,
        output_model=bad_model,
        label="bad",
        output_dir=output_dir,
        seed=args.seed,
        max_epoch=max_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        skip_existing=args.skip_existing,
    )
    ensure_transfer_series(
        prior=prior,
        smiles_file=good_smiles,
        output_model=positive_model,
        label="positive",
        output_dir=output_dir,
        seed=args.seed,
        max_epoch=max_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        skip_existing=args.skip_existing,
    )

    base_state = network_state(load_checkpoint(prior))
    reference_state = network_state(
        load_checkpoint(checkpoint_for_epoch(bad_model, args.reference_epoch, max_epoch))
    )
    scoped_keys = floating_keys(base_state)
    reference_deltas = {key: reference_state[key] - base_state[key] for key in scoped_keys}
    reference_norm = global_delta_norm(reference_deltas, scoped_keys)

    summaries = [base_summary(source_dir, output_dir, args.eval_samples)]
    direction_rows = []
    for epoch in epochs:
        bad_epoch = checkpoint_for_epoch(bad_model, epoch, max_epoch)
        positive_epoch = checkpoint_for_epoch(positive_model, epoch, max_epoch)
        specs = [
            ("bad_finetune", f"bad_epoch_{epoch}", bad_epoch, None),
            ("positive_finetune", f"positive_epoch_{epoch}", positive_epoch, None),
        ]

        standard_model = model_dir / f"standard_ne_epoch_{epoch}.model"
        standard_meta = make_ne_checkpoint(
            base_path=prior,
            bad_path=bad_epoch,
            output_path=standard_model,
            scale=args.lambda_value,
            target_norm=None,
            epoch=epoch,
        )
        direction_rows.append({"epoch": epoch, "direction_kind": "standard_ne", **standard_meta})
        specs.append(("standard_ne", f"standard_ne_epoch_{epoch}", standard_model, standard_meta))

        normalized_model = model_dir / f"epoch_norm_ne_epoch_{epoch}.model"
        normalized_meta = make_ne_checkpoint(
            base_path=prior,
            bad_path=bad_epoch,
            output_path=normalized_model,
            scale=args.lambda_value,
            target_norm=reference_norm,
            epoch=epoch,
        )
        direction_rows.append({"epoch": epoch, "direction_kind": "epoch_norm_ne", **normalized_meta})
        specs.append(("epoch_norm_ne", f"epoch_norm_ne_epoch_{epoch}", normalized_model, normalized_meta))

        for direction_kind, model, model_path, direction_meta in specs:
            summary, _ = sample_and_score(
                model_path=model_path,
                model=model,
                output_dir=output_dir,
                seed=args.seed,
                eval_samples=args.eval_samples,
                device=args.device,
                skip_existing=args.skip_existing,
            )
            summary["epoch"] = epoch
            summary["direction_kind"] = direction_kind
            if direction_meta:
                summary.update(direction_meta)
            summaries.append(summary)
        if not args.keep_models:
            standard_model.unlink(missing_ok=True)
            normalized_model.unlink(missing_ok=True)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(summary_path, index=False)
    pd.DataFrame(direction_rows).to_csv(output_dir / "direction_norms.csv", index=False)
    metadata = {
        "source_dir": str(source_dir),
        "prior": str(prior),
        "seed": args.seed,
        "epochs": epochs,
        "reference_epoch": args.reference_epoch,
        "reference_direction_norm": reference_norm,
        "lambda_value": args.lambda_value,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "usable_definition": "valid, canonical-unique, reactive-liability-free; novelty unavailable for the external prior",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if not args.keep_models:
        for path in model_dir.glob("*.model*"):
            path.unlink()

    display_columns = [
        "model",
        "epoch",
        "direction_kind",
        "reactive_hit_fraction",
        "valid_fraction",
        "usable_yield",
    ]
    print(summary_frame[display_columns].to_string(index=False))
    print(f"Wrote REINVENT epoch-sensitivity results to {output_dir}")


if __name__ == "__main__":
    main()

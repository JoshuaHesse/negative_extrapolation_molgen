"""Generate a budget-matched base evaluation without modifying training pools."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from neon_molgen.scoring import summarize_scores
from scripts.run_reinvent_reactive_replicates import (
    run_command,
    score_sample_csv,
    write_analysis_sample,
    write_sampling_config,
)

PAPER_SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
OBJECTIVES = {
    "reactive": "reactive_hit",
    "chelator": "chelator_hit",
    "charged_motif": "charged_motif_hit",
    "assay_interference": "assay_interference_hit",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def summarize_objectives(scores: pd.DataFrame, seed: int, n_sampled: int) -> pd.DataFrame:
    if len(scores) > n_sampled:
        raise ValueError("Written molecules cannot exceed the requested generation budget.")
    summaries = []
    for objective, column in OBJECTIVES.items():
        summary = summarize_scores(scores, liability_free_column=column, n_sampled=n_sampled)
        summary.update(seed=seed, model="base", objective=objective)
        summaries.append(summary)
    return pd.DataFrame(summaries)


def evaluate_seed(args: argparse.Namespace, seed: int, prior_hash: str) -> pd.DataFrame:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "definition": "reinvent_base_fixed_budget_v1",
        "seed": seed,
        "sampling_seed": seed + 1000,
        "n_sampled": args.eval_samples,
        "prior_sha256": prior_hash,
        "unique_molecules": True,
        "randomize_smiles": True,
        "device": args.device,
    }
    manifest_path = seed_dir / "run_config.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != config:
            raise ValueError(f"Configuration changed for {seed_dir}; use a new output directory.")
    elif any(seed_dir.iterdir()):
        raise ValueError(f"Refusing to reuse {seed_dir} without a matching run_config.json.")
    else:
        write_json(manifest_path, config)

    raw_path = seed_dir / "prior_samples.csv"
    sampling_manifest = seed_dir / "sampling_complete.json"
    if sampling_manifest.exists():
        if not raw_path.exists() or file_hash(raw_path) != json.loads(sampling_manifest.read_text())["sha256"]:
            raise ValueError(f"Completed sampling output is missing or changed: {raw_path}")
        print(f"[{seed}] reusing completed base sampling", flush=True)
    else:
        pending_path = seed_dir / "prior_samples.pending.csv"
        config_path = seed_dir / "sample_base.toml"
        write_sampling_config(
            config_path,
            model_file=args.prior,
            output_file=pending_path,
            json_output=seed_dir / "sample_base.json",
            num_smiles=args.eval_samples,
            device=args.device,
        )
        print(f"[{seed}] sampling {args.eval_samples} base attempts (no fine-tuning)", flush=True)
        run_command([
            "reinvent", "--seed", str(seed + 1000),
            "-l", str(seed_dir / "sample_base.log"), str(config_path),
        ])
        raw = pd.read_csv(pending_path)
        if "SMILES" not in raw or not 0 < len(raw) <= args.eval_samples:
            raise ValueError(f"Unexpected REINVENT sampling output: {pending_path}")
        pending_path.replace(raw_path)
        write_json(sampling_manifest, {"sha256": file_hash(raw_path), "n_written": len(raw)})

    # REINVENT saves accepted rows only. Keep the requested budget, not the
    # number of written rows, as the denominator for output and usable yield.
    scores, _ = score_sample_csv(
        raw_path, model="base", n_sampled=args.eval_samples, liability_free_column="reactive_hit",
    )
    write_csv(seed_dir / "scores_base.csv", scores)
    write_analysis_sample(seed_dir / "base_samples.csv", scores)
    metrics = summarize_objectives(scores, seed, args.eval_samples)
    write_csv(seed_dir / "objective_metrics.csv", metrics)
    print(f"[{seed}] wrote matched-budget base metrics for all four objectives", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=Path("external/REINVENT4/priors/reinvent.prior"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/external/reinvent4/base_evaluation_10000"))
    parser.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    parser.add_argument("--eval-samples", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.eval_samples <= 0 or not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("Provide a positive sampling budget and distinct seeds.")
    if not args.prior.is_file():
        parser.error(f"Missing pretrained REINVENT prior: {args.prior}")
    prior_hash = file_hash(args.prior)
    frames = [evaluate_seed(args, seed, prior_hash) for seed in args.seeds]
    metrics = pd.concat(frames, ignore_index=True)
    write_csv(args.output_dir / "objective_metrics.csv", metrics)
    write_json(args.output_dir / "evaluation_complete.json", {
        "definition": "reinvent_base_fixed_budget_v1",
        "seeds": args.seeds,
        "n_sampled": args.eval_samples,
        "prior_sha256": prior_hash,
        "objectives": OBJECTIVES,
        "note": "Base-only reevaluation; original 50k training pools and edited models are unchanged.",
    })
    print(f"Wrote base-only evaluation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

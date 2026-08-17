from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neon_molgen.checkpoint import save_checkpoint, save_json
from neon_molgen.data import read_smiles
from neon_molgen.mlflow_utils import log_artifacts, log_metrics_from_mapping, log_params, start_run
from neon_molgen.model import build_model
from neon_molgen.sample import sample_smiles
from neon_molgen.scoring import canonicalize_smiles_set, score_smiles, summarize_scores
from neon_molgen.tokenizer import SmilesTokenizer
from neon_molgen.train import train_model_with_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/guacamol_rnn_base.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--skip-global-summary-update",
        action="store_true",
        help=(
            "Write per-seed results without replacing the root-level base summary. "
            "Use this when adding development checkpoints to a confirmatory base directory."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame.reset_index()


def load_or_create_reference(train_smiles: list[str], output_dir: Path) -> set[str]:
    path = output_dir / "reference_canonical_smiles.smi"
    if path.exists():
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    canonical = canonicalize_smiles_set(train_smiles)
    path.write_text("\n".join(sorted(canonical)) + "\n")
    return canonical


def sample_and_score(
    model,
    tokenizer,
    sampling: dict,
    *,
    n_samples: int,
    device: str,
    output_path: Path,
) -> pd.DataFrame:
    smiles = sample_smiles(
        model,
        tokenizer,
        n_samples=n_samples,
        max_len=sampling["max_len"],
        temperature=sampling["temperature"],
        batch_size=sampling["batch_size"],
        device=device,
    )
    scores = score_smiles(smiles)
    scores.to_csv(output_path, index=False)
    return scores


def run_seed(
    config: dict,
    *,
    seed: int,
    train_smiles: list[str],
    reference_canonical_smiles: set[str],
    output_dir: Path,
    device: str,
) -> dict:
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    run_config = copy.deepcopy(config)
    run_config["seed"] = seed
    run_config["output_dir"] = str(seed_dir)
    run_config["base_output_dir"] = str(output_dir)
    run_config["reference_canonical_smiles_path"] = str(
        output_dir / "reference_canonical_smiles.smi"
    )
    architecture = run_config["model"].get("architecture", "rnn")

    with start_run(run_config, f"base__{architecture}__seed_{seed}"):
        log_params(run_config, {"workflow": "base_training", "architecture": architecture})
        set_seed(seed)
        tokenizer = SmilesTokenizer.from_smiles(train_smiles)
        model = build_model(len(tokenizer.itos), run_config["model"])
        n_parameters = count_parameters(model)
        model, history = train_model_with_history(
            model,
            train_smiles,
            tokenizer,
            epochs=run_config["training"]["epochs"],
            batch_size=run_config["training"]["batch_size"],
            learning_rate=run_config["training"]["learning_rate"],
            device=device,
            desc=f"base seed={seed}",
            seed=seed,
            on_epoch_end=lambda row: log_metrics_from_mapping(
                run_config,
                {
                    "loss_mean": row["loss_mean"],
                    "n_batches": row["n_batches"],
                },
                prefix="train/base/",
                step=int(row["epoch"]),
            ),
        )
        history_frame = pd.DataFrame(history)
        history_frame.insert(0, "seed", seed)
        history_path = seed_dir / "base_training_history.csv"
        history_frame.to_csv(history_path, index=False)
        checkpoint_path = seed_dir / "base.pt"
        save_checkpoint(checkpoint_path, model, tokenizer, run_config)

        sampling = run_config["sampling"]
        initial_scores = sample_and_score(
            model,
            tokenizer,
            sampling,
            n_samples=sampling["initial_samples"],
            device=device,
            output_path=seed_dir / "initial_samples.csv",
        )
        base_scores = sample_and_score(
            model,
            tokenizer,
            sampling,
            n_samples=sampling["eval_samples"],
            device=device,
            output_path=seed_dir / "base_samples.csv",
        )
        summary = {
            "seed": seed,
            "model": "base",
            "architecture": architecture,
            "n_train": len(train_smiles),
            "vocab_size": len(tokenizer.itos),
            "n_parameters": n_parameters,
            "epochs": run_config["training"]["epochs"],
            "final_train_loss": float(history_frame["loss_mean"].iloc[-1]),
            "initial_valid_fraction": float(initial_scores["valid"].mean())
            if len(initial_scores)
            else np.nan,
            **summarize_scores(
                base_scores,
                reference_canonical_smiles=reference_canonical_smiles,
            ),
        }
        pd.DataFrame([summary]).to_csv(seed_dir / "summary.csv", index=False)
        save_json(seed_dir / "config.json", run_config)
        log_metrics_from_mapping(run_config, summary)
        log_artifacts(
            run_config,
            [
                checkpoint_path,
                history_path,
                seed_dir / "initial_samples.csv",
                seed_dir / "base_samples.csv",
                seed_dir / "summary.csv",
                seed_dir / "config.json",
            ],
        )
        print(pd.DataFrame([summary]).to_string(index=False))
        return summary


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or config.get("replicates", {}).get("seeds")
    if not seeds:
        raise ValueError("Provide seeds with --seeds or config['replicates']['seeds'].")

    train_smiles = read_smiles(config["data"]["train_smiles"], limit=config["data"].get("limit"))
    if not train_smiles:
        raise ValueError("No training SMILES were loaded.")
    reference_canonical_smiles = load_or_create_reference(train_smiles, output_dir)
    save_json(output_dir / "config.json", config)

    summaries = []
    for index, seed in enumerate(seeds, start=1):
        seed = int(seed)
        seed_dir = output_dir / f"seed_{seed}"
        summary_path = seed_dir / "summary.csv"
        if args.skip_existing and summary_path.exists():
            print(f"[{index}/{len(seeds)}] skipping existing base seed={seed}")
            summaries.append(pd.read_csv(summary_path).iloc[0].to_dict())
            continue
        print(f"[{index}/{len(seeds)}] training base seed={seed}")
        summaries.append(
            run_seed(
                config,
                seed=seed,
                train_smiles=train_smiles,
                reference_canonical_smiles=reference_canonical_smiles,
                output_dir=output_dir,
                device=args.device,
            )
        )

    summary = pd.DataFrame(summaries)
    if not args.skip_global_summary_update:
        summary.to_csv(output_dir / "base_summary.csv", index=False)
        numeric = [col for col in summary.select_dtypes(include="number").columns if col != "seed"]
        aggregate = summary.groupby("architecture")[numeric].agg(["mean", "std", "count"])
        aggregate = flatten_columns(aggregate)
        aggregate.to_csv(output_dir / "base_summary_aggregate.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

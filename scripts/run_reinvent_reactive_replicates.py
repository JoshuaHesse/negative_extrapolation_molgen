from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from neon_molgen.scoring import score_smiles, summarize_scores
from scripts.reinvent_score_samples import read_smiles_table


def _sample_frame(frame: pd.DataFrame, n: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed).copy()


def select_binary_sets(
    scores: pd.DataFrame,
    selection: dict,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select size-matched liability-containing and liability-free molecules."""
    bad_column = selection["bad_where"]
    good_column = selection["good_where"]
    bad_value = selection.get("bad_value", 1.0)
    good_value = selection.get("good_value", 0.0)
    min_smiles = int(selection.get("min_smiles", 0))
    max_smiles = selection.get("max_smiles")
    max_smiles = int(max_smiles) if max_smiles is not None else None
    balance = bool(selection.get("balance", True))

    valid = scores[scores["valid"].astype(bool)].copy()
    missing = [column for column in (bad_column, good_column) if column not in valid]
    if missing:
        raise ValueError(f"Missing selection columns: {missing}")

    bad = valid[valid[bad_column].eq(bad_value)].copy()
    good = valid[valid[good_column].eq(good_value)].copy()
    n_bad_raw, n_good_raw = len(bad), len(good)
    if n_bad_raw < min_smiles or n_good_raw < min_smiles:
        raise ValueError(
            "Underpowered objective selection: "
            f"bad={n_bad_raw}, good={n_good_raw}, min_smiles={min_smiles}"
        )

    if balance:
        n_selected = min(n_bad_raw, n_good_raw)
        if max_smiles is not None:
            n_selected = min(n_selected, max_smiles)
        bad = _sample_frame(bad, n_selected, seed=seed + 101)
        good = _sample_frame(good, n_selected, seed=seed + 102)
    elif max_smiles is not None:
        bad = _sample_frame(bad, max_smiles, seed=seed + 101)
        good = _sample_frame(good, max_smiles, seed=seed + 102)

    metadata = {
        "mode": "binary",
        "bad_column": bad_column,
        "bad_value": bad_value,
        "good_column": good_column,
        "good_value": good_value,
        "balance": balance,
        "max_smiles": max_smiles,
        "min_smiles": min_smiles,
        "n_valid": int(len(valid)),
        "n_bad_raw": int(n_bad_raw),
        "n_good_raw": int(n_good_raw),
        "n_bad_selected": int(len(bad)),
        "n_good_selected": int(len(good)),
    }
    return bad.reset_index(drop=True), good.reset_index(drop=True), metadata


REINVENT_PRIOR_TOKENS = {
    "#",
    "$",
    "%",
    "%10",
    "(",
    ")",
    "-",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "=",
    "Br",
    "C",
    "Cl",
    "F",
    "N",
    "O",
    "S",
    "[N+]",
    "[N-]",
    "[O-]",
    "[S+]",
    "[n+]",
    "[nH]",
    "^",
    "c",
    "n",
    "o",
    "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run REINVENT4 liability-removal NEON replicates from a pretrained prior."
    )
    parser.add_argument("--config", default="configs/reinvent/reinvent4_external_model.json")
    parser.add_argument("--prior", default="external/REINVENT4/priors/reinvent.prior")
    parser.add_argument("--output-dir", default="results/external/reinvent4/reactive_replicates")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
    )
    parser.add_argument("--baseline-samples", type=int, default=50000)
    parser.add_argument("--eval-samples", type=int, default=10000)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tl-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-random-controls", action="store_true")
    parser.add_argument(
        "--delete-random-control-models",
        action="store_true",
        help=(
            "Delete transient random-NE and transfer-learning checkpoint files after "
            "sampling. Retain random_tuned.model as the learned control direction."
        ),
    )
    parser.add_argument(
        "--neon-scope-name",
        default="full_model",
        help="Parameter scope name passed to reinvent_negative_extrapolate.py.",
    )
    parser.add_argument(
        "--neon-include-prefixes",
        nargs="*",
        default=None,
        help="Optional REINVENT checkpoint tensor prefixes to update during NEON.",
    )
    return parser.parse_args()


def run_command(command: list[str], *, log_path: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def write_sampling_config(
    path: Path,
    *,
    model_file: Path | str,
    output_file: Path,
    json_output: Path,
    num_smiles: int,
    device: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                'run_type = "sampling"',
                f'device = "{device}"',
                f'json_out_config = "{json_output}"',
                "",
                "[parameters]",
                f'model_file = "{model_file}"',
                f'output_file = "{output_file}"',
                f"num_smiles = {num_smiles}",
                "unique_molecules = true",
                "randomize_smiles = true",
                "",
            ]
        )
    )


def write_tl_config(
    path: Path,
    *,
    prior: Path | str,
    smiles_file: Path,
    output_model_file: Path,
    tb_logdir: Path,
    json_output: Path,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    save_every_n_epochs: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_model_file.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                'run_type = "transfer_learning"',
                f'device = "{device}"',
                f'tb_logdir = "{tb_logdir}"',
                f'json_out_config = "{json_output}"',
                "",
                "[parameters]",
                f"num_epochs = {epochs}",
                f"save_every_n_epochs = {save_every_n_epochs or epochs}",
                f"batch_size = {batch_size}",
                f"sample_batch_size = {batch_size}",
                "num_refs = 0",
                f'input_model_file = "{prior}"',
                f'smiles_file = "{smiles_file}"',
                f'output_model_file = "{output_model_file}"',
                "standardize_smiles = true",
                "randomize_smiles = true",
                "max_sequence_length = 256",
                "",
                "[scheduler]",
                f"lr = {learning_rate}",
                "gamma = 0.95",
                "step = 10",
                "",
            ]
        )
    )


def format_lambda(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def model_label_from_path(path: Path) -> str:
    stem = path.stem
    if stem == "bad_tuned":
        return "bad"
    if stem == "positive_tuned":
        return "positive"
    if stem == "random_tuned":
        return "random_finetune"
    if stem.startswith("neon_lambda_"):
        return "neon_lambda_" + stem.removeprefix("neon_lambda_").replace("p", ".")
    if stem.startswith("random_neon_lambda_"):
        return "random_neon_lambda_" + stem.removeprefix("random_neon_lambda_").replace("p", ".")
    return stem


def reinvent_tokens(smiles: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(smiles):
        if smiles[i] == "[":
            j = smiles.find("]", i)
            if j == -1:
                tokens.append(smiles[i:])
                break
            tokens.append(smiles[i : j + 1])
            i = j + 1
            continue
        if smiles.startswith("%10", i):
            tokens.append("%10")
            i += 3
            continue
        if smiles.startswith("Cl", i) or smiles.startswith("Br", i):
            tokens.append(smiles[i : i + 2])
            i += 2
            continue
        tokens.append(smiles[i])
        i += 1
    return tokens


def is_reinvent_supported_smiles(smiles: str) -> bool:
    return all(token in REINVENT_PRIOR_TOKENS for token in reinvent_tokens(smiles))


def supported_tl_frame(frame: pd.DataFrame) -> pd.DataFrame:
    column = "smiles" if "smiles" in frame.columns else "canonical_smiles"
    mask = frame[column].fillna("").astype(str).map(is_reinvent_supported_smiles)
    filtered = frame[mask].copy()
    return filtered.reset_index(drop=True)


def write_reinvent_tl_smiles(path: Path, frame: pd.DataFrame) -> None:
    """Write REINVENT-token-compatible SMILES for transfer learning."""
    column = "smiles" if "smiles" in frame.columns else "canonical_smiles"
    smiles = frame[column].dropna().astype(str)
    path.write_text("\n".join(smiles.tolist()) + "\n")


def write_balanced_tl_smiles(
    selection_dir: Path,
    bad: pd.DataFrame,
    good: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, int]:
    bad_supported = supported_tl_frame(bad)
    good_supported = supported_tl_frame(good)
    n_bad_supported_raw = len(bad_supported)
    n_good_supported_raw = len(good_supported)
    n_tl = min(len(bad_supported), len(good_supported))
    if n_tl == 0:
        raise ValueError("No REINVENT-token-compatible bad/good SMILES remain after TL filtering.")
    if len(bad_supported) > n_tl:
        bad_supported = bad_supported.sample(n=n_tl, random_state=seed + 301).reset_index(drop=True)
    if len(good_supported) > n_tl:
        good_supported = good_supported.sample(n=n_tl, random_state=seed + 302).reset_index(drop=True)
    write_reinvent_tl_smiles(selection_dir / "bad_smiles.smi", bad_supported)
    write_reinvent_tl_smiles(selection_dir / "good_smiles.smi", good_supported)
    return {
        "n_bad_tl_supported": int(len(bad_supported)),
        "n_good_tl_supported": int(len(good_supported)),
        "n_bad_tl_filtered_out": int(len(bad) - n_bad_supported_raw),
        "n_good_tl_filtered_out": int(len(good) - n_good_supported_raw),
    }


def write_random_tl_smiles(
    path: Path,
    scores: pd.DataFrame,
    *,
    target_size: int,
    seed: int,
) -> dict[str, int]:
    """Write a matched random-data TL set using the same vocabulary filter."""
    valid = scores[scores["valid"].astype(bool)].copy()
    canonical_column = "canonical_smiles" if "canonical_smiles" in valid else "smiles"
    valid = valid.drop_duplicates(canonical_column).reset_index(drop=True)
    supported = supported_tl_frame(valid)
    if len(supported) < target_size:
        raise ValueError(
            "Not enough REINVENT-token-compatible molecules for matched random TL: "
            f"available={len(supported)}, requested={target_size}."
        )
    selected = supported.sample(n=target_size, random_state=seed + 303).reset_index(drop=True)
    write_reinvent_tl_smiles(path, selected)
    return {
        "n_random_tl_supported_pool": int(len(supported)),
        "n_random_tl_selected": int(len(selected)),
    }


def refresh_selection_smiles(selection_dir: Path, scores: pd.DataFrame, *, seed: int) -> None:
    bad_path = selection_dir / "bad_selected.csv"
    good_path = selection_dir / "good_selected.csv"
    if bad_path.exists() and good_path.exists():
        metadata = write_balanced_tl_smiles(
            selection_dir,
            pd.read_csv(bad_path),
            pd.read_csv(good_path),
            seed=seed,
        )
        write_random_tl_smiles(
            selection_dir / "random_smiles.smi",
            scores,
            target_size=metadata["n_bad_tl_supported"],
            seed=seed,
        )


def objective_name(config: dict) -> str:
    return str(config.get("objective", "liability_removal"))


def liability_column(config: dict) -> str:
    selection = config.get("selection", {})
    bad_where = selection.get("bad_where")
    if isinstance(bad_where, str) and bad_where.endswith("_hit"):
        return bad_where
    if objective_name(config) == "alert_removal":
        return "alert_hit"
    return "liability_hit"


def score_sample_csv(
    path: Path,
    *,
    model: str,
    n_sampled: int,
    liability_free_column: str,
) -> tuple[pd.DataFrame, dict]:
    scores = score_smiles(read_smiles_table(path, smiles_column="SMILES"))
    scores.insert(0, "model", model)
    summary = summarize_scores(
        scores,
        liability_free_column=liability_free_column,
        n_sampled=n_sampled,
    )
    summary["model"] = model
    return scores, summary


def write_analysis_sample(path: Path, scores: pd.DataFrame) -> None:
    columns = [column for column in ["smiles", "canonical_smiles", "valid", "model"] if column in scores.columns]
    if "canonical_smiles" not in columns:
        columns = [column for column in scores.columns if column in {"smiles", "valid", "model"}]
    scores[columns].to_csv(path, index=False)


def aggregate_outputs(output_dir: Path, *, seeds: set[int] | None = None) -> None:
    summaries = []
    for path in sorted(output_dir.glob("seed_*/scores/summary.csv")):
        seed = int(path.parent.parent.name.removeprefix("seed_"))
        if seeds is not None and seed not in seeds:
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        summaries.append(frame)
    if not summaries:
        return
    metrics = pd.concat(summaries, ignore_index=True)
    metrics.to_csv(output_dir / "objective_metrics.csv", index=False)
    numeric = [column for column in metrics.select_dtypes(include="number").columns if column != "seed"]
    aggregate = metrics.groupby("model")[numeric].agg(["mean", "std", "count"])
    aggregate.columns = ["_".join(str(part) for part in column if part) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output_dir / "aggregate_summary.csv", index=False)


def ensure_transfer_model(
    args: argparse.Namespace,
    *,
    seed: int,
    label: str,
    prior: Path | str,
    smiles_file: Path,
    model_path: Path,
    config_path: Path,
    seed_dir: Path,
) -> None:
    if args.skip_existing and model_path.exists():
        return
    write_tl_config(
        config_path,
        prior=prior,
        smiles_file=smiles_file,
        output_model_file=model_path,
        tb_logdir=seed_dir / f"tb_{label}",
        json_output=model_path.parent / f"{label}_tl.json",
        device=args.device,
        epochs=args.tl_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    run_command(
        ["reinvent", "--seed", str(seed), "-l", str(model_path.parent / f"{label}_tl.log"), str(config_path)]
    )


def run_seed(args: argparse.Namespace, config: dict, seed: int) -> None:
    objective = objective_name(config)
    free_column = liability_column(config)
    primary_fraction = f"{free_column}_fraction"
    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    baseline_dir = seed_dir / "baseline"
    selection_dir = seed_dir / "selection"
    model_dir = seed_dir / "models"
    sample_dir = seed_dir / "samples"
    score_dir = seed_dir / "scores"
    config_dir = seed_dir / "configs"
    seed_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = baseline_dir / "prior_samples.csv"
    baseline_scored = baseline_dir / "prior_samples_scored.csv"
    if not (args.skip_existing and baseline_scored.exists()):
        baseline_toml = config_dir / "sample_base.toml"
        write_sampling_config(
            baseline_toml,
            model_file=args.prior,
            output_file=baseline_csv,
            json_output=baseline_dir / "sample_base.json",
            num_smiles=args.baseline_samples,
            device=args.device,
        )
        run_command(
            ["reinvent", "--seed", str(seed), "-l", str(baseline_dir / "sample_base.log"), str(baseline_toml)]
        )
        baseline_scores = score_smiles(read_smiles_table(baseline_csv, smiles_column="SMILES"))
        baseline_scores.to_csv(baseline_scored, index=False)
        pd.DataFrame(
            [
                summarize_scores(
                    baseline_scores,
                    liability_free_column=free_column,
                    n_sampled=args.baseline_samples,
                )
            ]
        ).to_csv(
            baseline_dir / "prior_samples_summary.csv",
            index=False,
        )

    selection_summary = selection_dir / "selection_summary.json"
    if not (args.skip_existing and selection_summary.exists()):
        selection_dir.mkdir(parents=True, exist_ok=True)
        scores = pd.read_csv(baseline_scored)
        bad, good, metadata = select_binary_sets(scores, config["selection"], seed=seed)
        metadata.update(
            {
                "config": str(args.config),
                "scored": str(baseline_scored),
                "objective": objective,
                "liability_free_column": free_column,
                "seed": int(seed),
            }
        )
        bad.to_csv(selection_dir / "bad_selected.csv", index=False)
        good.to_csv(selection_dir / "good_selected.csv", index=False)
        tl_metadata = write_balanced_tl_smiles(selection_dir, bad, good, seed=seed)
        metadata.update(tl_metadata)
        metadata.update(
            write_random_tl_smiles(
                selection_dir / "random_smiles.smi",
                scores,
                target_size=tl_metadata["n_bad_tl_supported"],
                seed=seed,
            )
        )
        selection_summary.write_text(json.dumps(metadata, indent=2))
    else:
        refresh_selection_smiles(selection_dir, pd.read_csv(baseline_scored), seed=seed)

    bad_model = model_dir / "bad_tuned.model"
    positive_model = model_dir / "positive_tuned.model"
    random_model = model_dir / "random_tuned.model"
    for label, smiles_name, model_path in [
        ("bad", "bad_smiles.smi", bad_model),
        ("positive", "good_smiles.smi", positive_model),
        ("random", "random_smiles.smi", random_model),
    ]:
        ensure_transfer_model(
            args,
            seed=seed,
            label=label,
            prior=args.prior,
            smiles_file=selection_dir / smiles_name,
            model_path=model_path,
            config_path=config_dir / f"{objective}_{label}_tl.toml",
            seed_dir=seed_dir,
        )

    expected_neon = [model_dir / f"neon_lambda_{format_lambda(value)}.model" for value in args.lambda_values]
    expected_random = [
        model_dir / f"random_neon_lambda_{format_lambda(value)}.model" for value in args.lambda_values
    ]
    expected_random_scores = [
        score_dir / f"scores_random_neon_lambda_{format_lambda(value)}.csv"
        for value in args.lambda_values
    ]
    random_control_marker = model_dir / "trained_random_global_norm_control.json"
    refresh_random_controls = not args.skip_random_controls and not random_control_marker.exists()
    random_expected_ok = args.skip_random_controls or (
        random_control_marker.exists()
        and (
            all(path.exists() for path in expected_random)
            or all(path.exists() for path in expected_random_scores)
        )
    )
    if not (args.skip_existing and all(path.exists() for path in expected_neon) and random_expected_ok):
        command = [
            "python",
            "scripts/reinvent_negative_extrapolate.py",
            "--base",
            str(args.prior),
            "--bad",
            str(bad_model),
            "--output-dir",
            str(model_dir),
            "--lambda-values",
            *[str(value) for value in args.lambda_values],
            "--scope-name",
            args.neon_scope_name,
        ]
        if args.neon_include_prefixes:
            command.extend(["--include-prefixes", *args.neon_include_prefixes])
        if not args.skip_random_controls:
            command.extend(["--random-controls", "--random-trained", str(random_model)])
        run_command(command)

    model_paths = [bad_model, positive_model, random_model]
    model_paths.extend(expected_neon)
    if not args.skip_random_controls:
        model_paths.extend(expected_random)

    sample_specs: list[tuple[str, Path]] = [("base", baseline_csv)]
    for model_path in model_paths:
        model = model_label_from_path(model_path)
        sample_csv = sample_dir / f"samples_{model.replace('.', 'p')}.csv"
        force_random_refresh = refresh_random_controls and model.startswith("random_neon_lambda_")
        if force_random_refresh or not (args.skip_existing and sample_csv.exists()):
            sample_toml = config_dir / f"sample_{model.replace('.', 'p')}.toml"
            write_sampling_config(
                sample_toml,
                model_file=model_path,
                output_file=sample_csv,
                json_output=sample_dir / f"sample_{model.replace('.', 'p')}.json",
                num_smiles=args.eval_samples,
                device=args.device,
            )
            run_command(
                [
                    "reinvent",
                    "--seed",
                    str(seed + 1000),
                    "-l",
                    str(sample_dir / f"sample_{model.replace('.', 'p')}.log"),
                    str(sample_toml),
                ]
            )
        sample_specs.append((model, sample_csv))

    summaries = []
    for model, sample_csv in sample_specs:
        score_path = score_dir / f"scores_{model.replace('.', 'p')}.csv"
        force_random_refresh = refresh_random_controls and model.startswith("random_neon_lambda_")
        if args.skip_existing and score_path.exists() and not force_random_refresh:
            scores = pd.read_csv(score_path)
            n_sampled = args.baseline_samples if model == "base" else args.eval_samples
            summary = summarize_scores(
                scores,
                liability_free_column=free_column,
                n_sampled=n_sampled,
            )
            summary["model"] = model
        else:
            n_sampled = args.baseline_samples if model == "base" else args.eval_samples
            scores, summary = score_sample_csv(
                sample_csv,
                model=model,
                n_sampled=n_sampled,
                liability_free_column=free_column,
            )
            score_dir.mkdir(parents=True, exist_ok=True)
            scores.to_csv(score_path, index=False)
        summaries.append(summary)
        write_analysis_sample(seed_dir / f"{model}_samples.csv", scores)

    if not args.skip_random_controls:
        random_control_marker.write_text(
            json.dumps(
                {
                    "definition": "trained_random_global_scope_l2_v1",
                    "random_checkpoint": str(random_model),
                    "scope": args.neon_scope_name,
                },
                indent=2,
            )
        )
        if args.delete_random_control_models:
            for path in expected_random:
                path.unlink(missing_ok=True)
            for path in model_dir.glob("random_tuned.model.*.chkpt"):
                path.unlink()

    shutil.copyfile(selection_dir / "bad_smiles.smi", seed_dir / "bad_smiles.smi")
    shutil.copyfile(selection_dir / "good_smiles.smi", seed_dir / "good_smiles.smi")

    summary_frame = pd.DataFrame(summaries)
    columns = ["model", *[column for column in summary_frame.columns if column != "model"]]
    summary_frame = summary_frame[columns]
    score_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(score_dir / "summary.csv", index=False)
    print(
        summary_frame[
            [
                "model",
                "n_sampled",
                "n",
                primary_fraction,
                "usable_yield",
                "liability_hit_fraction",
                "qed_mean",
            ]
        ].rename(columns={primary_fraction: "primary_hit_fraction"}).to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    run_metadata = {
        "config": args.config,
        "prior": args.prior,
        "objective": objective_name(config),
        "liability_free_column": liability_column(config),
        "seeds": args.seeds,
        "baseline_samples": args.baseline_samples,
        "eval_samples": args.eval_samples,
        "lambda_values": args.lambda_values,
        "tl_epochs": args.tl_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "skip_random_controls": args.skip_random_controls,
        "random_control_definition": (
            None
            if args.skip_random_controls
            else "trained_random_global_scope_l2_v1"
        ),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_metadata, indent=2))

    for index, seed in enumerate(args.seeds, start=1):
        print(
            f"[{index}/{len(args.seeds)}] REINVENT {objective_name(config)} seed={seed}",
            flush=True,
        )
        run_seed(args, config, seed)
        aggregate_outputs(output_dir, seeds=set(args.seeds))

    if not args.skip_analysis:
        run_command(
            [
                "python",
                "scripts/analyze_diversity.py",
                "--results-dir",
                str(output_dir),
                "--output-dir",
                str(output_dir / "analysis" / "diversity"),
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--models",
                "base",
                "bad",
                "positive",
                *[f"neon_lambda_{value:g}" for value in args.lambda_values],
                *([] if args.skip_random_controls else [f"random_neon_lambda_{value:g}" for value in args.lambda_values]),
                "--internal-sample-size",
                "2000",
                "--internal-random-pairs",
                "200000",
                "--cluster-fit-size",
                "5000",
                "--n-clusters",
                "50",
            ]
        )
        run_command(
            [
                "python",
                "scripts/analyze_distribution_distance.py",
                "--results-dir",
                str(output_dir),
                "--output-dir",
                str(output_dir / "analysis" / "distribution_distance"),
                "--sample-size",
                "5000",
                "--min-size",
                "500",
                "--device",
                "cpu",
                "--jobs",
                "8",
                "--models",
                "base",
                "bad",
                "positive",
                *[f"neon_lambda_{value:g}" for value in args.lambda_values],
                *([] if args.skip_random_controls else [f"random_neon_lambda_{value:g}" for value in args.lambda_values]),
            ]
        )

    aggregate_outputs(output_dir)
    print(f"Wrote REINVENT {objective_name(config)} replicate results to {output_dir}")


if __name__ == "__main__":
    main()

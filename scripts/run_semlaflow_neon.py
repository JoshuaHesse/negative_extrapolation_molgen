from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import posebusters as pb
import semlaflow.scriptutil as semla_util
import torch
from rdkit import Chem
from semlaflow import predict as semla_predict
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.util.molrepr import GeometricMol, GeometricMolBatch

from neon_molgen.scoring import score_smiles, summarize_scores
from scripts.run_semlaflow_probe import rdkit_row, write_sdf

SCOPE_PREFIXES = {
    "full_model": ("gen.", "ema_gen.module."),
    "edge_out_only": (
        "gen.edge_out_proj.",
        "ema_gen.module.edge_out_proj.",
    ),
    "edge_refine_output": (
        "gen.dynamics.refine_layer.",
        "gen.edge_out_proj.",
        "ema_gen.module.dynamics.refine_layer.",
        "ema_gen.module.edge_out_proj.",
    ),
    "chemistry_tail": (
        "gen.dynamics.feat_norm.",
        "gen.dynamics.bond_norm.",
        "gen.dynamics.refine_layer.",
        "gen.edge_out_proj.",
        "gen.atom_classifier_head.",
        "gen.charge_classifier_head.",
        "ema_gen.module.dynamics.feat_norm.",
        "ema_gen.module.dynamics.bond_norm.",
        "ema_gen.module.dynamics.refine_layer.",
        "ema_gen.module.edge_out_proj.",
        "ema_gen.module.atom_classifier_head.",
        "ema_gen.module.charge_classifier_head.",
    ),
    "last_block_chemistry_tail": (
        "gen.dynamics.layers.11.node_ff.node_norm.",
        "gen.dynamics.layers.11.node_ff.invariant_mlp.node_ff.",
        "gen.dynamics.layers.11.message_ff.node_norm.",
        "gen.dynamics.layers.11.message_ff.edge_norm.",
        "gen.dynamics.layers.11.message_ff.node_proj.",
        "gen.dynamics.layers.11.message_ff.message_mlp.",
        "gen.dynamics.layers.11.node_attn.",
        "gen.dynamics.final_ff_block.node_norm.",
        "gen.dynamics.final_ff_block.invariant_mlp.node_ff.",
        "gen.dynamics.feat_norm.",
        "gen.dynamics.bond_norm.",
        "gen.dynamics.refine_layer.",
        "gen.edge_out_proj.",
        "gen.atom_classifier_head.",
        "gen.charge_classifier_head.",
        "ema_gen.module.dynamics.layers.11.node_ff.node_norm.",
        "ema_gen.module.dynamics.layers.11.node_ff.invariant_mlp.node_ff.",
        "ema_gen.module.dynamics.layers.11.message_ff.node_norm.",
        "ema_gen.module.dynamics.layers.11.message_ff.edge_norm.",
        "ema_gen.module.dynamics.layers.11.message_ff.node_proj.",
        "ema_gen.module.dynamics.layers.11.message_ff.message_mlp.",
        "ema_gen.module.dynamics.layers.11.node_attn.",
        "ema_gen.module.dynamics.final_ff_block.node_norm.",
        "ema_gen.module.dynamics.final_ff_block.invariant_mlp.node_ff.",
        "ema_gen.module.dynamics.feat_norm.",
        "ema_gen.module.dynamics.bond_norm.",
        "ema_gen.module.dynamics.refine_layer.",
        "ema_gen.module.edge_out_proj.",
        "ema_gen.module.atom_classifier_head.",
        "ema_gen.module.charge_classifier_head.",
    ),
    "categorical_heads": (
        "gen.edge_out_proj.",
        "gen.atom_classifier_head.",
        "gen.charge_classifier_head.",
        "ema_gen.module.edge_out_proj.",
        "ema_gen.module.atom_classifier_head.",
        "ema_gen.module.charge_classifier_head.",
    ),
    "output_heads": (
        "gen.edge_out_proj.",
        "gen.atom_classifier_head.",
        "gen.charge_classifier_head.",
        "ema_gen.module.edge_out_proj.",
        "ema_gen.module.atom_classifier_head.",
        "ema_gen.module.charge_classifier_head.",
    ),
    "last_block_output": (
        "gen.dynamics.layers.11.",
        "gen.dynamics.final_ff_block.",
        "gen.dynamics.coord_norm.",
        "gen.dynamics.feat_norm.",
        "gen.dynamics.coord_proj.",
        "gen.dynamics.coord_head.",
        "gen.dynamics.bond_norm.",
        "gen.dynamics.refine_layer.",
        "gen.edge_out_proj.",
        "gen.atom_classifier_head.",
        "gen.charge_classifier_head.",
        "ema_gen.module.dynamics.layers.11.",
        "ema_gen.module.dynamics.final_ff_block.",
        "ema_gen.module.dynamics.coord_norm.",
        "ema_gen.module.dynamics.feat_norm.",
        "ema_gen.module.dynamics.coord_proj.",
        "ema_gen.module.dynamics.coord_head.",
        "ema_gen.module.dynamics.bond_norm.",
        "ema_gen.module.dynamics.refine_layer.",
        "ema_gen.module.edge_out_proj.",
        "ema_gen.module.atom_classifier_head.",
        "ema_gen.module.charge_classifier_head.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-seed SemlaFlow NEON experiment from generated 3D molecules."
    )
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument(
        "--baseline-scores-cache",
        help=(
            "Optional shared CSV cache for baseline objective annotations. "
            "The first run writes it and later matched selections reuse it."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bad-checkpoint",
        help="Reuse an existing bad-tuned checkpoint instead of preparing data and retraining.",
    )
    parser.add_argument(
        "--reference-summary",
        help="Reuse base and bad_tuned rows from an existing summary CSV instead of resampling them.",
    )
    parser.add_argument(
        "--reference-models",
        nargs="+",
        default=["base", "bad_tuned"],
        help="Rows to retain from --reference-summary.",
    )
    parser.add_argument(
        "--include-trained-model",
        action="store_true",
        help="Also sample the directly trained model when --reference-summary is used.",
    )
    parser.add_argument("--dataset", default="geom-drugs", choices=["geom-drugs", "qm9"])
    parser.add_argument("--dataset-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--objective-column", default="brenk_top5_oxygen_nitrogen_single_bond_hit")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument(
        "--selection-mode",
        choices=["objective_hits", "objective_nonhits", "balanced_objective_hits", "random"],
        default="objective_hits",
        help=(
            "Select objective hits for bad fine-tuning, objective non-hits for "
            "positive fine-tuning, or an unconditional random reference set."
        ),
    )
    parser.add_argument(
        "--balanced-objective-columns",
        nargs="+",
        default=[],
        help="Boolean score columns used by balanced_objective_hits selection.",
    )
    parser.add_argument("--per-objective-train-size", type=int)
    parser.add_argument("--per-objective-val-size", type=int)
    parser.add_argument(
        "--trained-model-name",
        default="bad_tuned",
        help="Checkpoint and summary name for the directly fine-tuned model.",
    )
    parser.add_argument("--bad-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--type-loss-weight", type=float, default=1.0)
    parser.add_argument("--bond-loss-weight", type=float, default=1.0)
    parser.add_argument("--charge-loss-weight", type=float, default=1.0)
    parser.add_argument("--batch-cost", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--eval-batch-cost", type=int, default=4096)
    parser.add_argument("--integration-steps", type=int, default=100)
    parser.add_argument("--cat-sampling-noise-level", type=int, default=1)
    parser.add_argument("--ode-sampling-strategy", default="log")
    parser.add_argument("--bucket-cost-scale", default="linear")
    parser.add_argument("--lambda-values", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument(
        "--random-correction-checkpoint",
        help=(
            "Optional checkpoint fine-tuned on a matched random baseline sample. "
            "When supplied, also write NE checkpoints using bad_delta - scale * random_delta."
        ),
    )
    parser.add_argument("--random-correction-scale", type=float, default=1.0)
    parser.add_argument(
        "--additional-correction-checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Additional task-arithmetic correction checkpoint. May be repeated. "
            "For example, positive=positive_tuned.ckpt produces checkpoints from "
            "base - lambda * (bad_delta - positive_delta)."
        ),
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=sorted(SCOPE_PREFIXES),
        default=["full_model", "output_heads", "last_block_output"],
    )
    parser.add_argument("--posebusters-config", default="mol")
    parser.add_argument("--posebusters-workers", type=int, default=4)
    parser.add_argument("--skip-posebusters", action="store_true")
    parser.add_argument(
        "--delete-neon-checkpoints",
        action="store_true",
        help="Delete derived NE checkpoints after sampling to reduce disk usage.",
    )
    parser.add_argument(
        "--sample-raw-gen",
        action="store_true",
        help="Sample from the raw gen.* weights instead of the EMA copy. Fine-tuning already updates gen.*.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed training checkpoints and per-model sample summaries in the output directory.",
    )
    parser.add_argument(
        "--bad-model-only",
        action="store_true",
        help=(
            "Train and sample only the bad-tuned model. This is intended for "
            "diagnosing bad-direction strength before running NE extrapolation."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)


def format_lambda(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_sdf(path: Path) -> list[Chem.Mol | None]:
    return list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False))


def find_baseline_sdf(baseline_dir: Path) -> Path:
    sdfs = sorted(baseline_dir.glob("*.sdf"))
    if len(sdfs) != 1:
        raise ValueError(f"Expected exactly one SDF in {baseline_dir}, found {len(sdfs)}.")
    return sdfs[0]


def strip_explicit_h_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    mol = Chem.RemoveHs(mol, sanitize=False)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def add_posebusters_pass_column(
    scores: pd.DataFrame,
    molecules: list[Chem.Mol | None],
    posebusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map PoseBusters pass/fail rows back to the original sampled molecule order."""
    frame = scores.copy()
    frame["posebusters_all_checks"] = False
    if posebusters_df.empty:
        return frame

    pass_values = posebusters_df.fillna(False).astype(bool).all(axis=1).tolist()
    present_index = 0
    for sample_index, mol in enumerate(molecules):
        if mol is None:
            continue
        if present_index >= len(pass_values):
            break
        frame.at[sample_index, "posebusters_all_checks"] = bool(pass_values[present_index])
        present_index += 1
    return frame


def update_3d_usable_yield(
    summary: dict,
    scores: pd.DataFrame,
    *,
    liability_column: str,
    n_sampled: int,
) -> None:
    """Fixed-budget yield of valid, unique, liability-free molecules that pass PoseBusters."""
    if "posebusters_all_checks" not in scores or liability_column not in scores:
        summary["n_valid_unique_liability_free_posebusters_pass"] = np.nan
        summary["usable_3d_yield"] = np.nan
        return

    valid_unique = scores[scores["valid"]].drop_duplicates("canonical_smiles").copy()
    usable = valid_unique[
        valid_unique[liability_column].eq(0)
        & valid_unique["posebusters_all_checks"].fillna(False).astype(bool)
    ]
    n_usable = int(len(usable))
    summary["n_valid_unique_liability_free_posebusters_pass"] = float(n_usable)
    summary["usable_3d_yield"] = float(n_usable / n_sampled) if n_sampled else np.nan


def load_baseline_with_scores(
    baseline_dir: Path,
    objective_column: str,
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, list[Chem.Mol | None]]:
    rdkit_profile = pd.read_csv(baseline_dir / "rdkit_profile.csv")
    sdf_mols = load_sdf(find_baseline_sdf(baseline_dir))
    present = rdkit_profile[rdkit_profile["rdkit_mol_present"].astype(bool)].copy().reset_index(drop=True)
    if len(sdf_mols) != len(present):
        raise ValueError(
            f"SDF/profile mismatch: SDF has {len(sdf_mols)} molecules, profile has {len(present)} present molecules."
        )

    if cache_path is not None and cache_path.is_file():
        frame = pd.read_csv(cache_path)
        if len(frame) != len(present):
            raise ValueError(
                f"Baseline score cache mismatch: cache has {len(frame)} rows, "
                f"but baseline has {len(present)} present molecules."
            )
        required = {"sdf_position", "objective_hit", f"score_{objective_column}"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"Baseline score cache is missing columns: {sorted(missing)}")
        print(f"Reusing baseline score cache {cache_path}")
        return frame, sdf_mols

    smiles = present["canonical_smiles"].fillna("").astype(str).map(strip_explicit_h_smiles).tolist()
    scores = score_smiles(smiles).reset_index(drop=True)
    if objective_column not in scores:
        raise ValueError(f"Missing objective column {objective_column!r}.")

    frame = pd.concat(
        [
            present[["index", "canonical_smiles", "scaffold_smiles"]].reset_index(drop=True),
            scores.add_prefix("score_"),
        ],
        axis=1,
    )
    frame["sdf_position"] = np.arange(len(frame))
    frame["objective_hit"] = scores[objective_column].astype(float)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
        print(f"Wrote baseline score cache {cache_path}")
    return frame, sdf_mols


def select_objective_molecules(
    frame: pd.DataFrame,
    sdf_mols: list[Chem.Mol | None],
    *,
    objective_value: int,
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[Chem.Mol], list[Chem.Mol], dict]:
    selected_frame = frame[frame["objective_hit"].eq(objective_value)].copy()
    selection_label = "hits" if objective_value == 1 else "nonhits"
    if len(selected_frame) == 0:
        raise ValueError(f"Objective selection found no {selection_label}.")

    rng = np.random.default_rng(seed)
    shuffled_positions = rng.permutation(selected_frame["sdf_position"].to_numpy())
    selected_mols = []
    skipped = 0
    for position in shuffled_positions:
        mol = sdf_mols[int(position)]
        if mol is None:
            skipped += 1
            continue
        try:
            GeometricMol.from_rdkit(mol)
        except Exception:
            skipped += 1
            continue
        selected_mols.append(mol)

    if len(selected_mols) < 10:
        raise ValueError(
            f"Only {len(selected_mols)} selected molecules were convertible to SemlaFlow format."
        )

    n_train = min(train_size, len(selected_mols))
    train_mols = selected_mols[:n_train]
    remaining = selected_mols[n_train:]
    if remaining:
        val_mols = remaining[: min(val_size, len(remaining))]
    else:
        val_mols = train_mols[: min(val_size, len(train_mols))]

    metadata = {
        "n_profile_molecules": int(len(frame)),
        f"n_objective_{selection_label}": int(len(selected_frame)),
        f"n_convertible_objective_{selection_label}": int(len(selected_mols)),
        "n_skipped_selected_molecules": int(skipped),
        "n_train_molecules": int(len(train_mols)),
        "n_val_molecules": int(len(val_mols)),
    }
    return train_mols, val_mols, metadata


def select_random_molecules(
    frame: pd.DataFrame,
    sdf_mols: list[Chem.Mol | None],
    *,
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[Chem.Mol], list[Chem.Mol], dict]:
    """Select a matched unconditional sample for estimating generic fine-tuning drift."""
    rng = np.random.default_rng(seed)
    shuffled_positions = rng.permutation(frame["sdf_position"].to_numpy())
    random_mols = []
    skipped = 0
    for position in shuffled_positions:
        mol = sdf_mols[int(position)]
        if mol is None:
            skipped += 1
            continue
        try:
            GeometricMol.from_rdkit(mol)
        except Exception:
            skipped += 1
            continue
        random_mols.append(mol)
        if len(random_mols) >= train_size + val_size:
            break

    if len(random_mols) < 10:
        raise ValueError(f"Only {len(random_mols)} random molecules were convertible to SemlaFlow format.")

    n_train = min(train_size, len(random_mols))
    train_mols = random_mols[:n_train]
    remaining = random_mols[n_train:]
    val_mols = remaining[: min(val_size, len(remaining))]
    if not val_mols:
        val_mols = train_mols[: min(val_size, len(train_mols))]

    metadata = {
        "n_profile_molecules": int(len(frame)),
        "n_convertible_random_molecules": int(len(random_mols)),
        "n_skipped_selected_molecules": int(skipped),
        "n_train_molecules": int(len(train_mols)),
        "n_val_molecules": int(len(val_mols)),
    }
    return train_mols, val_mols, metadata


def select_balanced_objective_molecules(
    frame: pd.DataFrame,
    sdf_mols: list[Chem.Mol | None],
    *,
    objective_columns: list[str],
    train_size_per_objective: int,
    val_size_per_objective: int,
    seed: int,
) -> tuple[list[Chem.Mol], list[Chem.Mol], dict, pd.DataFrame]:
    """Select unique molecules with equal assignment quotas across objective families."""
    if not objective_columns:
        raise ValueError("balanced_objective_hits requires --balanced-objective-columns.")
    if train_size_per_objective <= 0 or val_size_per_objective <= 0:
        raise ValueError("Balanced per-objective train and validation sizes must be positive.")

    score_columns = [f"score_{column}" for column in objective_columns]
    missing = [column for column in score_columns if column not in frame]
    if missing:
        raise ValueError(f"Missing balanced objective score columns: {missing}")

    hits = frame[score_columns].fillna(0).astype(bool)
    hit_multiplicity = hits.sum(axis=1)
    exclusive_counts = {
        objective: int((hits[f"score_{objective}"] & hit_multiplicity.eq(1)).sum())
        for objective in objective_columns
    }
    # Allocate scarce families first. Within each family, prefer exclusive
    # examples so overlapping molecules remain available to fill later quotas.
    allocation_order = sorted(objective_columns, key=lambda item: exclusive_counts[item])
    quota = train_size_per_objective + val_size_per_objective
    rng = np.random.default_rng(seed)
    available = set(frame.index.tolist())
    assignments: dict[str, list[tuple[int, Chem.Mol]]] = {}
    skipped = 0

    for objective in allocation_order:
        score_column = f"score_{objective}"
        candidates = [index for index in available if bool(hits.at[index, score_column])]
        rng.shuffle(candidates)
        candidates.sort(key=lambda index: int(hit_multiplicity.at[index]))
        selected: list[tuple[int, Chem.Mol]] = []
        for index in candidates:
            position = int(frame.at[index, "sdf_position"])
            mol = sdf_mols[position]
            if mol is None:
                skipped += 1
                available.discard(index)
                continue
            try:
                GeometricMol.from_rdkit(mol)
            except Exception:
                skipped += 1
                available.discard(index)
                continue
            selected.append((index, mol))
            available.remove(index)
            if len(selected) == quota:
                break
        if len(selected) != quota:
            raise ValueError(
                f"Balanced selection for {objective!r} found {len(selected)} unique convertible "
                f"molecules, but {quota} are required."
            )
        assignments[objective] = selected

    train_mols: list[Chem.Mol] = []
    val_mols: list[Chem.Mol] = []
    assignment_rows = []
    for objective in objective_columns:
        selected = assignments[objective]
        for split, subset in (
            ("train", selected[:train_size_per_objective]),
            ("val", selected[train_size_per_objective:]),
        ):
            destination = train_mols if split == "train" else val_mols
            for index, mol in subset:
                destination.append(mol)
                row = {
                    "assigned_objective": objective,
                    "split": split,
                    "sdf_position": int(frame.at[index, "sdf_position"]),
                    "canonical_smiles": frame.at[index, "canonical_smiles"],
                    "family_hit_multiplicity": int(hit_multiplicity.at[index]),
                }
                row.update(
                    {
                        column: int(bool(hits.at[index, f"score_{column}"]))
                        for column in objective_columns
                    }
                )
                assignment_rows.append(row)

    assignment_frame = pd.DataFrame(assignment_rows)
    metadata = {
        "n_profile_molecules": int(len(frame)),
        "balanced_objective_columns": objective_columns,
        "balanced_allocation_order": allocation_order,
        "exclusive_pool_counts": exclusive_counts,
        "n_unique_union_molecules": int(hits.any(axis=1).sum()),
        "n_skipped_selected_molecules": int(skipped),
        "n_train_molecules": int(len(train_mols)),
        "n_val_molecules": int(len(val_mols)),
        "n_train_per_objective": int(train_size_per_objective),
        "n_val_per_objective": int(val_size_per_objective),
    }
    return train_mols, val_mols, metadata, assignment_frame


def write_smol_dataset(mols: list[Chem.Mol], path: Path) -> None:
    smol_mols = [GeometricMol.from_rdkit(mol) for mol in mols]
    batch = GeometricMolBatch.from_list(smol_mols)
    path.write_bytes(batch.to_bytes())


def prepare_training_data(args: argparse.Namespace, output_dir: Path) -> tuple[Path, dict]:
    baseline_dir = Path(args.baseline_dir)
    cache_path = Path(args.baseline_scores_cache) if args.baseline_scores_cache else None
    frame, sdf_mols = load_baseline_with_scores(
        baseline_dir,
        args.objective_column,
        cache_path=cache_path,
    )
    assignment_frame = None
    if args.selection_mode == "objective_hits":
        train_mols, val_mols, metadata = select_objective_molecules(
            frame,
            sdf_mols,
            objective_value=1,
            train_size=args.train_size,
            val_size=args.val_size,
            seed=args.seed,
        )
    elif args.selection_mode == "objective_nonhits":
        train_mols, val_mols, metadata = select_objective_molecules(
            frame,
            sdf_mols,
            objective_value=0,
            train_size=args.train_size,
            val_size=args.val_size,
            seed=args.seed,
        )
    elif args.selection_mode == "balanced_objective_hits":
        if args.per_objective_train_size is None or args.per_objective_val_size is None:
            raise ValueError(
                "balanced_objective_hits requires --per-objective-train-size and "
                "--per-objective-val-size."
            )
        train_mols, val_mols, metadata, assignment_frame = select_balanced_objective_molecules(
            frame,
            sdf_mols,
            objective_columns=args.balanced_objective_columns,
            train_size_per_objective=args.per_objective_train_size,
            val_size_per_objective=args.per_objective_val_size,
            seed=args.seed,
        )
    else:
        train_mols, val_mols, metadata = select_random_molecules(
            frame,
            sdf_mols,
            train_size=args.train_size,
            val_size=args.val_size,
            seed=args.seed,
        )

    selection_dir = output_dir / "selection"
    data_dir = selection_dir / f"{args.selection_mode}_smol"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_smol_dataset(train_mols, data_dir / "train.smol")
    write_smol_dataset(val_mols, data_dir / "val.smol")

    if cache_path is None:
        frame.to_csv(selection_dir / "baseline_objective_scores.csv", index=False)
    if assignment_frame is not None:
        assignment_frame.to_csv(selection_dir / "balanced_assignments.csv", index=False)
    selection_metadata = {
        "objective_column": args.objective_column,
        "selection_mode": args.selection_mode,
        "baseline_dir": str(baseline_dir),
        "baseline_scores_cache": str(cache_path) if cache_path is not None else None,
        **metadata,
    }
    (selection_dir / "selection_summary.json").write_text(json.dumps(selection_metadata, indent=2, sort_keys=True))
    return data_dir, selection_metadata


def build_training_dm(args: argparse.Namespace, data_dir: Path, vocab):
    if args.dataset == "geom-drugs":
        coord_std = semla_util.GEOM_COORDS_STD_DEV
        bucket_limits = semla_util.GEOM_DRUGS_BUCKET_LIMITS
    elif args.dataset == "qm9":
        coord_std = semla_util.QM9_COORDS_STD_DEV
        bucket_limits = semla_util.QM9_BUCKET_LIMITS
    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")

    n_bond_types = semla_util.get_n_bond_types("uniform-sample")
    transform = partial(semla_util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)
    train_dataset = GeometricDataset.load(data_dir / "train.smol", transform=transform)
    val_dataset = GeometricDataset.load(data_dir / "val.smol", transform=transform)

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise="uniform-sample",
        bond_noise="uniform-sample",
        scale_ot=False,
        zero_com=True,
    )
    train_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation="unmask",
        bond_interpolation="unmask",
        coord_noise_std=0.2,
        type_dist_temp=1.0,
        equivariant_ot=True,
        batch_ot=False,
        time_alpha=2.0,
        time_beta=1.0,
    )
    val_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation="unmask",
        bond_interpolation="unmask",
        equivariant_ot=False,
        batch_ot=False,
        fixed_time=0.9,
    )
    return GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        None,
        args.batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=val_interpolant,
        test_interpolant=None,
        bucket_limits=bucket_limits,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
    )


def load_semlaflow_model(args: argparse.Namespace, ckpt_path: Path, vocab):
    model_args = argparse.Namespace(
        ckpt_path=str(ckpt_path),
        integration_steps=args.integration_steps,
        ode_sampling_strategy=args.ode_sampling_strategy,
        cat_sampling_noise_level=args.cat_sampling_noise_level,
    )
    model = semla_predict.load_model(model_args, vocab)
    model.lr = args.learning_rate
    model.warm_up_steps = 0
    return model


def train_bad_model(args: argparse.Namespace, data_dir: Path, output_dir: Path, vocab) -> Path:
    dm = build_training_dm(args, data_dir, vocab)
    model = load_semlaflow_model(args, Path(args.ckpt_path), vocab)
    loss_weights = {
        "coord-loss": args.coord_loss_weight,
        "type-loss": args.type_loss_weight,
        "bond-loss": args.bond_loss_weight,
        "charge-loss": args.charge_loss_weight,
    }
    if any(weight < 0 for weight in loss_weights.values()):
        raise ValueError(f"Loss weights must be non-negative: {loss_weights}")

    # SemlaFlow exposes weights for categorical losses but not coordinates. Apply
    # all four weights after its native per-component normalization so that the
    # fine-tuning direction can be aligned with chemistry rather than geometry.
    native_loss = model._loss

    def weighted_loss(data, interpolated, predicted):
        losses = native_loss(data, interpolated, predicted)
        return {name: value * loss_weights[name] for name, value in losses.items()}

    model._loss = weighted_loss
    train_batches = len(dm.train_dataloader())
    if train_batches == 0:
        raise ValueError(
            "SemlaFlow produced zero train batches. Increase --train-size or reduce --batch-cost."
        )
    model.total_steps = max(1, train_batches * args.bad_epochs)

    accelerator = "gpu" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        min_epochs=args.bad_epochs,
        max_epochs=args.bad_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=1.0,
        precision="32",
    )
    trainer.fit(model, datamodule=dm)

    base_checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    base_checkpoint["state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    base_checkpoint["epoch"] = int(base_checkpoint.get("epoch", 0)) + args.bad_epochs
    base_checkpoint["global_step"] = int(base_checkpoint.get("global_step", 0)) + int(model.total_steps)
    base_checkpoint["neon_bad_finetune_metadata"] = {
        "objective_column": args.objective_column,
        "selection_mode": args.selection_mode,
        "loss_weights": loss_weights,
        "learning_rate": args.learning_rate,
        "epochs": args.bad_epochs,
    }
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    bad_path = model_dir / f"{args.trained_model_name}.ckpt"
    torch.save(base_checkpoint, bad_path)
    return bad_path


def tensor_in_scope(key: str, scope: str) -> bool:
    prefixes = SCOPE_PREFIXES[scope]
    return any(key.startswith(prefix) for prefix in prefixes)


def state_delta_norm(
    deltas: dict[str, torch.Tensor],
    scoped_keys: list[str],
) -> float:
    """Global L2 norm over the floating-point tensors in an NE scope."""
    squared_norm = 0.0
    for key in scoped_keys:
        delta = deltas[key].detach().to(dtype=torch.float32, device="cpu")
        squared_norm += float(torch.sum(delta * delta))
    return squared_norm**0.5


def checkpoint_delta(
    state: dict[str, torch.Tensor],
    base_state: dict[str, torch.Tensor],
    scoped_keys: list[str],
) -> dict[str, torch.Tensor]:
    return {key: state[key] - base_state[key] for key in scoped_keys}


def subtract_deltas(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    scoped_keys: list[str],
    *,
    right_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    return {key: left[key] - right_scale * right[key] for key in scoped_keys}


def write_neon_checkpoints(
    args: argparse.Namespace,
    bad_ckpt_path: Path,
    output_dir: Path,
) -> Iterator[tuple[str, Path]]:
    base = torch.load(args.ckpt_path, map_location="cpu")
    bad = torch.load(bad_ckpt_path, map_location="cpu")
    base_state = base["state_dict"]
    bad_state = bad["state_dict"]
    if set(base_state) != set(bad_state):
        raise ValueError("Base and bad SemlaFlow checkpoints have different state_dict keys.")

    if args.random_correction_scale < 0:
        raise ValueError("--random-correction-scale must be non-negative.")
    correction_states: list[tuple[str, str, dict[str, torch.Tensor], float]] = []
    if args.random_correction_checkpoint:
        random_checkpoint = torch.load(args.random_correction_checkpoint, map_location="cpu")
        random_state = random_checkpoint["state_dict"]
        if set(base_state) != set(random_state):
            raise ValueError("Base and random-reference SemlaFlow checkpoints have different state_dict keys.")
        correction_states.append(
            (
                "random",
                str(args.random_correction_checkpoint),
                random_state,
                args.random_correction_scale,
            )
        )

    for spec in args.additional_correction_checkpoint:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --additional-correction-checkpoint {spec!r}; expected NAME=PATH."
            )
        name, checkpoint_path = spec.split("=", 1)
        if not name or not checkpoint_path:
            raise ValueError(
                f"Invalid --additional-correction-checkpoint {spec!r}; expected NAME=PATH."
            )
        if not name.replace("_", "").isalnum():
            raise ValueError(
                f"Correction name {name!r} may contain only letters, digits, and underscores."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        correction_state = checkpoint["state_dict"]
        if set(base_state) != set(correction_state):
            raise ValueError(
                f"Base and {name!r} correction checkpoints have different state_dict keys."
            )
        correction_states.append((name, checkpoint_path, correction_state, 1.0))

    scope_summaries = {}
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for scope in args.scopes:
        scoped_keys = [
            key
            for key, value in base_state.items()
            if tensor_in_scope(key, scope) and torch.is_tensor(value) and value.is_floating_point()
        ]
        if not scoped_keys:
            raise ValueError(f"Scope {scope!r} selected no floating-point tensors.")
        scope_summaries[scope] = {
            "prefixes": SCOPE_PREFIXES[scope],
            "n_scoped_tensors": len(scoped_keys),
            "scoped_keys": scoped_keys,
        }

    (model_dir / "neon_scope_summary.json").write_text(
        json.dumps(scope_summaries, indent=2, sort_keys=True)
    )

    # Yield one checkpoint at a time. The caller samples and deletes it before
    # requesting the next one, avoiding a large transient checkpoint footprint.
    for scope in args.scopes:
        scoped_keys = scope_summaries[scope]["scoped_keys"]
        bad_delta = checkpoint_delta(bad_state, base_state, scoped_keys)
        bad_norm = state_delta_norm(bad_delta, scoped_keys)
        correction_deltas = {
            name: checkpoint_delta(state, base_state, scoped_keys)
            for name, _, state, _ in correction_states
        }
        positive_corrected_delta = None
        if "positive" in correction_deltas:
            positive_corrected_delta = subtract_deltas(
                bad_delta,
                correction_deltas["positive"],
                scoped_keys,
            )
        scope_summaries[scope]["bad_direction_norm"] = bad_norm
        if "random" in correction_deltas:
            random_norm = state_delta_norm(correction_deltas["random"], scoped_keys)
            if random_norm == 0.0:
                raise ValueError("Random fine-tuning produced a zero parameter update.")
            scope_summaries[scope]["random_ne_control"] = {
                "definition": "trained_random_global_scope_l2_v1",
                "raw_direction_norm": random_norm,
                "reference_direction_norm": bad_norm,
                "direction_scale": bad_norm / random_norm,
            }
        if positive_corrected_delta is not None:
            positive_corrected_norm = state_delta_norm(
                positive_corrected_delta,
                scoped_keys,
            )
            scope_summaries[scope]["positive_corrected_direction_norm"] = (
                positive_corrected_norm
            )
            if "random" in correction_deltas:
                random_corrected_delta = subtract_deltas(
                    bad_delta,
                    correction_deltas["random"],
                    scoped_keys,
                )
                random_corrected_norm = state_delta_norm(
                    random_corrected_delta,
                    scoped_keys,
                )
                if random_corrected_norm == 0.0:
                    raise ValueError("Random-corrected direction has zero parameter norm.")
                scope_summaries[scope]["random_corrected_control"] = {
                    "definition": "trained_random_corrected_global_scope_l2_v1",
                    "raw_direction_norm": random_corrected_norm,
                    "reference_direction": "positive_corrected_direction",
                    "reference_direction_norm": positive_corrected_norm,
                    "direction_scale": positive_corrected_norm / random_corrected_norm,
                }

        for lambda_value in args.lambda_values:
            neon = torch.load(args.ckpt_path, map_location="cpu")
            neon_state = neon["state_dict"]
            for key in scoped_keys:
                neon_state[key] = base_state[key] - lambda_value * (bad_state[key] - base_state[key])
            neon["neon_metadata"] = {
                "method": "negative_extrapolation",
                "bad_checkpoint": str(bad_ckpt_path),
                "lambda": lambda_value,
                "scope": scope,
                "objective_column": args.objective_column,
            }
            path = model_dir / f"{scope}_neon_lambda_{format_lambda(lambda_value)}.ckpt"
            torch.save(neon, path)
            model_name = f"{scope}_neon_lambda_{format_lambda(lambda_value)}"
            yield model_name, path

            if "random" in correction_deltas:
                random_delta = correction_deltas["random"]
                random_norm = scope_summaries[scope]["random_ne_control"][
                    "raw_direction_norm"
                ]
                random_scale = bad_norm / random_norm
                random_neon = torch.load(args.ckpt_path, map_location="cpu")
                random_neon_state = random_neon["state_dict"]
                for key in scoped_keys:
                    random_neon_state[key] = (
                        base_state[key] - lambda_value * random_scale * random_delta[key]
                    )
                random_neon["neon_metadata"] = {
                    "method": "trained_random_negative_extrapolation",
                    "random_checkpoint": next(
                        path for name, path, _, _ in correction_states if name == "random"
                    ),
                    "norm_definition": "global_scope_l2_v1",
                    "reference_direction": "bad_finetune",
                    "bad_direction_norm": bad_norm,
                    "random_direction_raw_norm": random_norm,
                    "random_direction_scale": random_scale,
                    "lambda": lambda_value,
                    "scope": scope,
                    "objective_column": args.objective_column,
                }
                random_name = f"{scope}_random_neon_lambda_{format_lambda(lambda_value)}"
                random_path = model_dir / f"{random_name}.ckpt"
                torch.save(random_neon, random_path)
                yield random_name, random_path

            for correction_name, correction_path, _correction_state, correction_scale in correction_states:
                corrected = torch.load(args.ckpt_path, map_location="cpu")
                corrected_state = corrected["state_dict"]
                corrected_delta = subtract_deltas(
                    bad_delta,
                    correction_deltas[correction_name],
                    scoped_keys,
                    right_scale=correction_scale,
                )
                corrected_scale = 1.0
                norm_reference = None
                if correction_name == "random" and positive_corrected_delta is not None:
                    corrected_norm = state_delta_norm(corrected_delta, scoped_keys)
                    positive_corrected_norm = state_delta_norm(positive_corrected_delta, scoped_keys)
                    if corrected_norm == 0.0:
                        raise ValueError("Random-corrected direction has zero parameter norm.")
                    corrected_scale = positive_corrected_norm / corrected_norm
                    norm_reference = "positive_corrected_direction"
                for key in scoped_keys:
                    corrected_state[key] = (
                        base_state[key] - lambda_value * corrected_scale * corrected_delta[key]
                    )
                corrected["neon_metadata"] = {
                    "method": f"{correction_name}_corrected_negative_extrapolation",
                    "bad_checkpoint": str(bad_ckpt_path),
                    "correction_name": correction_name,
                    "correction_checkpoint": correction_path,
                    "correction_scale": correction_scale,
                    "direction_norm_scale": corrected_scale,
                    "norm_definition": "global_scope_l2_v1" if norm_reference else None,
                    "norm_reference": norm_reference,
                    "lambda": lambda_value,
                    "scope": scope,
                    "objective_column": args.objective_column,
                }
                correction_label = correction_name
                if norm_reference:
                    correction_label = f"norm_matched_{correction_name}"
                corrected_name = f"{scope}_{correction_label}_corrected_neon_lambda_{format_lambda(lambda_value)}"
                corrected_path = model_dir / f"{corrected_name}.ckpt"
                torch.save(corrected, corrected_path)
                yield corrected_name, corrected_path

        (model_dir / "neon_scope_summary.json").write_text(
            json.dumps(scope_summaries, indent=2, sort_keys=True)
        )


def sample_checkpoint(args: argparse.Namespace, ckpt_path: Path, output_dir: Path, model_name: str, vocab) -> pd.DataFrame:
    summary_path = output_dir / "samples" / model_name / "summary.json"
    if args.resume and summary_path.is_file():
        print(f"Resume: reusing {summary_path}")
        return pd.DataFrame([json.loads(summary_path.read_text())])

    model_args = argparse.Namespace(
        ckpt_path=str(ckpt_path),
        data_path=args.data_path,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        n_molecules=args.eval_samples,
        batch_cost=args.eval_batch_cost,
        integration_steps=args.integration_steps,
        cat_sampling_noise_level=args.cat_sampling_noise_level,
        ode_sampling_strategy=args.ode_sampling_strategy,
        bucket_cost_scale=args.bucket_cost_scale,
    )
    dm = semla_predict.dm_from_ckpt(model_args, vocab)
    model = semla_predict.load_model(model_args, vocab)
    if args.sample_raw_gen:
        model.ema_gen = None
    molecules, _ = semla_util.generate_molecules(
        model,
        dm,
        args.integration_steps,
        args.ode_sampling_strategy,
    )

    model_dir = output_dir / "samples" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = model_dir / f"{model_name}.sdf"
    write_sdf(molecules, sdf_path)
    rdkit_df = pd.DataFrame([rdkit_row(index, mol) for index, mol in enumerate(molecules)])
    rdkit_df.to_csv(model_dir / "rdkit_profile.csv", index=False)

    smiles = rdkit_df["canonical_smiles"].fillna("").astype(str).map(strip_explicit_h_smiles).tolist()
    scores = score_smiles(smiles)

    if not args.skip_posebusters:
        present = [mol for mol in molecules if mol is not None]
        buster = pb.PoseBusters(config=args.posebusters_config, max_workers=args.posebusters_workers)
        posebusters_df = buster.bust(present, None, None).reset_index(drop=True) if present else pd.DataFrame()
        posebusters_df.to_csv(model_dir / "posebusters_profile.csv", index=False)
        scores = add_posebusters_pass_column(scores, molecules, posebusters_df)
    else:
        posebusters_df = pd.DataFrame()

    scores.to_csv(model_dir / "scores.csv", index=False)

    summary = summarize_scores(
        scores,
        liability_free_column=args.objective_column,
        n_sampled=args.eval_samples,
    )
    summary.update(
        {
            "model": model_name,
            "checkpoint": str(ckpt_path),
            "objective_column": args.objective_column,
            "objective_hit_fraction": summary.get(f"{args.objective_column}_fraction"),
            "coord_loss_weight": args.coord_loss_weight,
            "type_loss_weight": args.type_loss_weight,
            "bond_loss_weight": args.bond_loss_weight,
            "charge_loss_weight": args.charge_loss_weight,
        }
    )

    if not args.skip_posebusters:
        if not posebusters_df.empty:
            summary["posebusters_all_checks_fraction"] = float(
                posebusters_df.fillna(False).astype(bool).all(axis=1).mean()
            )
            summary["posebusters_all_checks_yield"] = float(
                scores["posebusters_all_checks"].fillna(False).astype(bool).sum() / args.eval_samples
            )
        update_3d_usable_yield(
            summary,
            scores,
            liability_column=args.objective_column,
            n_sampled=args.eval_samples,
        )

    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return pd.DataFrame([summary])


def main() -> None:
    args = parse_args()
    if args.selection_mode == "random" and not args.bad_model_only:
        raise ValueError("Random selection is only supported with --bad-model-only.")
    set_seed(args.seed)
    semla_util.disable_lib_stdout()
    semla_util.configure_fs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    vocab = semla_util.build_vocab()
    resumable_trained_checkpoint = output_dir / "models" / f"{args.trained_model_name}.ckpt"
    if args.bad_checkpoint:
        bad_ckpt_path = Path(args.bad_checkpoint)
        if not bad_ckpt_path.is_file():
            raise FileNotFoundError(bad_ckpt_path)
        if args.prepare_only:
            raise ValueError("--prepare-only cannot be combined with --bad-checkpoint.")
    elif args.resume and resumable_trained_checkpoint.is_file():
        bad_ckpt_path = resumable_trained_checkpoint
        print(f"Resume: reusing trained checkpoint {bad_ckpt_path}")
    else:
        data_dir, selection_metadata = prepare_training_data(args, output_dir)
        print(json.dumps(selection_metadata, indent=2, sort_keys=True))
        if args.prepare_only:
            return
        bad_ckpt_path = train_bad_model(args, data_dir, output_dir, vocab)

    neon_checkpoints = [] if args.bad_model_only else write_neon_checkpoints(
        args, bad_ckpt_path, output_dir
    )

    if args.bad_model_only:
        if args.reference_summary:
            reference = pd.read_csv(args.reference_summary)
            reference = reference[reference["model"].isin(args.reference_models)].copy()
            missing_models = set(args.reference_models) - set(reference["model"])
            if missing_models:
                raise ValueError(
                    f"--reference-summary is missing requested models: {sorted(missing_models)}"
                )
            summaries = [reference]
        else:
            summaries = [
                sample_checkpoint(args, Path(args.ckpt_path), output_dir, "base", vocab)
            ]
        summaries.append(sample_checkpoint(
            args, bad_ckpt_path, output_dir, args.trained_model_name, vocab
        ))
    elif args.reference_summary:
        reference = pd.read_csv(args.reference_summary)
        reference = reference[reference["model"].isin(args.reference_models)].copy()
        missing_models = set(args.reference_models) - set(reference["model"])
        if missing_models:
            raise ValueError(f"--reference-summary is missing requested models: {sorted(missing_models)}")
        summaries = [reference]
        if args.include_trained_model:
            summaries.append(sample_checkpoint(
                args, bad_ckpt_path, output_dir, args.trained_model_name, vocab
            ))
    else:
        summaries = [
            sample_checkpoint(args, Path(args.ckpt_path), output_dir, "base", vocab),
            sample_checkpoint(args, bad_ckpt_path, output_dir, "bad_tuned", vocab),
        ]
    for model_name, ckpt_path in neon_checkpoints:
        summaries.append(sample_checkpoint(args, ckpt_path, output_dir, model_name, vocab))
        if args.delete_neon_checkpoints:
            ckpt_path.unlink()

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"Wrote {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

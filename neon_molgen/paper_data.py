"""Validated paper inputs shared by notebooks and statistical analyses."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_SEEDS = (13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
REINVENT_BASE = Path("results/external/reinvent4/base_evaluation_10000")
CALIBRATION = Path("results/analysis/fcd_calibration_size_check/comparison_metrics.csv")
MW_BANDS = (
    "base_resample", "mw_bottom_10pct", "mw_bottom_20pct", "mw_20_40pct",
    "mw_40_60pct", "mw_60_80pct", "mw_top_20pct", "mw_top_10pct",
)
CALIBRATION_MODES = ("original", "pair_matched", "common_size")


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
    """Reconstruct the fixed-budget endpoint from saved molecule-level flags."""
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


def _seed_rows(frame: pd.DataFrame, seeds, context: str) -> pd.DataFrame:
    selected = frame[frame.seed.isin(seeds)].copy()
    if selected.seed.duplicated().any() or set(selected.seed) != set(seeds):
        raise ValueError(f"{context}: expected exactly one row for each seed {list(seeds)}")
    return selected


def load_reinvent_base(root: Path, *, objective: str, seeds=PAPER_SEEDS) -> pd.DataFrame:
    folder = root / REINVENT_BASE
    marker = folder / "evaluation_complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Missing {marker}. Run make paper-reinvent-base-budget first.")
    manifest = json.loads(marker.read_text())
    if (manifest.get("definition") != "reinvent_base_fixed_budget_v1"
            or manifest.get("n_sampled") != 10000
            or not set(seeds).issubset(manifest.get("seeds", []))
            or objective not in manifest.get("objectives", {})):
        raise ValueError(f"Incomplete or incompatible REINVENT base evaluation: {marker}")
    frame = pd.read_csv(folder / "objective_metrics.csv")
    frame = _seed_rows(frame[frame.objective.eq(objective)], seeds, f"REINVENT base/{objective}")
    required = ["n_sampled", "n_valid", "usable_yield", f"{objective}_hit_fraction"]
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite fixed-budget base endpoints")
    if not frame.model.eq("base").all() or not frame.n_sampled.eq(10000).all():
        raise ValueError("REINVENT baseline must contain only 10,000-attempt base evaluations")
    if not frame.n_valid.between(0, 10000).all() or not frame.usable_yield.between(0, 1).all():
        raise ValueError("Invalid fixed-budget base counts/yields")
    return frame


def replace_reinvent_base(frame: pd.DataFrame, root: Path, *, objective: str,
                          seeds=PAPER_SEEDS) -> pd.DataFrame:
    """Replace base rows only; never overwrite historical experiment tables."""
    base = load_reinvent_base(root, objective=objective, seeds=seeds)
    edited = frame[frame.seed.isin(seeds) & frame.model.ne("base")].copy()
    if not edited.n_sampled.eq(10000).all():
        raise ValueError("Expected 10,000 requested samples for REINVENT edited evaluations")
    return pd.concat([edited, base], ignore_index=True, sort=False)


def replace_reinvent_base_diversity(frame: pd.DataFrame, root: Path,
                                    *, seeds=PAPER_SEEDS) -> pd.DataFrame:
    endpoints = load_reinvent_base(root, objective="chelator", seeds=seeds)
    path = root / REINVENT_BASE / "analysis/diversity/diversity_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run make paper-reinvent-base-budget first.")
    base = _seed_rows(pd.read_csv(path), seeds, "REINVENT base diversity")
    if not base.model.eq("base").all():
        raise ValueError("Expected base-only diversity table")
    counts = base.set_index("seed").n_valid.sort_index()
    expected = endpoints.set_index("seed").n_valid.sort_index()
    if not np.array_equal(counts.to_numpy(), expected.to_numpy()):
        raise ValueError("REINVENT base diversity does not match the 10k evaluation")
    # The standalone base analysis fits different fingerprint clusters. Only
    # intrinsic diversity statistics can be mixed with the existing model tables.
    cluster_columns = [c for c in base if "cluster" in c]
    base[cluster_columns] = np.nan
    edited = frame[frame.seed.isin(seeds) & frame.model.ne("base")].copy()
    return pd.concat([edited, base], ignore_index=True, sort=False)


def load_mw_calibration_comparisons(root: Path, *, seeds=PAPER_SEEDS) -> pd.DataFrame:
    path = root / CALIBRATION
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run make paper-fcd-mw-calibration-size-check first.")
    frame = pd.read_csv(path)
    frame = frame[frame.seed.isin(seeds)].copy()
    keys = ["generator", "seed", "band", "mode"]
    expected = pd.MultiIndex.from_product(
        [("rnn", "transformer", "reinvent"), seeds, MW_BANDS, CALIBRATION_MODES], names=keys,
    )
    if frame.duplicated(keys).any() or set(frame.set_index(keys).index) != set(expected):
        raise ValueError("Incomplete three-generator, three-mode MW calibration")
    if not np.isfinite(frame[["fcd", "fdd", "n_query", "n_reference", "n_overlap"]]).all().all():
        raise ValueError("Non-finite MW calibration values")
    if (frame[["n_query", "n_reference"]] < 2).any().any():
        raise ValueError("Insufficient calibration sample counts")
    matched = frame[frame["mode"].ne("original")]
    if not matched.n_query.eq(matched.n_reference).all():
        raise ValueError("Matched calibration has unequal query/reference counts")
    common = frame[frame["mode"].eq("common_size")]
    if (not common.n_query.eq(common.common_size).all()
            or not common.groupby(["generator", "seed"]).n_query.nunique().eq(1).all()):
        raise ValueError("Common-size calibration varies sample count across MW bands")
    if not frame.n_overlap.between(0, frame[["n_query", "n_reference"]].min(axis=1)).all():
        raise ValueError("Impossible query/reference overlap count")
    return frame

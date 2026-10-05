"""Compare original MW calibration with matched sample sizes, in separate outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_distribution_distance import (
    descriptor_matrix,
    fdd_descriptor_vector,
    frechet_distance,
    read_sample_smiles,
    sample_smiles,
    scale_fdd_descriptors,
)
from scripts.analyze_fcd_mw_calibration import MW_BANDS, molecular_weight_frame, select_band

GENERATORS = {
    "rnn": "results/objectives/guacamol_rnn_chelator_removal",
    "transformer": "results/objectives/guacamol_transformer_chelator_lastblock_final",
    "reinvent": "results/external/reinvent4/chelator_replicates",
}
SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def make_comparisons(frame: pd.DataFrame, seed: int, cap: int, minimum: int) -> list[dict]:
    reference = sample_smiles(frame.smiles.tolist(), cap, key=f"mw_calibration::{seed}::reference")
    selections = {}
    for band, lo, hi, label in MW_BANDS:
        selected, cut_lo, cut_hi = select_band(frame, lo, hi)
        if len(selected) < minimum:
            raise ValueError(f"Seed {seed}, {band}: only {len(selected)} molecules; minimum is {minimum}.")
        query = sample_smiles(selected, cap, key=f"mw_calibration::{seed}::{band}")
        selections[band] = (query, label, len(selected), cut_lo, cut_hi)
    common_n = min(len(reference), *(len(item[0]) for item in selections.values()))
    rows = []
    for band, (query, label, available, lo, hi) in selections.items():
        common_query = sample_smiles(query, common_n, key=f"matched_common::{seed}::{band}")
        # Nested reference samples retain the same random draw across modes.
        for mode, left, right in [
            ("original", query, reference),
            ("pair_matched", query, reference[:len(query)]),
            ("common_size", common_query, reference[:common_n]),
        ]:
            rows.append({
                "seed": seed, "band": band, "band_label": label, "mode": mode,
                "n_base_available": len(frame), "n_band_available": available,
                "mw_cut_low": lo, "mw_cut_high": hi, "common_size": common_n,
                "query": left, "reference": right,
            })
    return rows


def validate_original(plan: list[dict], old: pd.DataFrame) -> None:
    original = old.set_index("band", verify_integrity=True)
    if set(original.index) != {row["band"] for row in plan}:
        raise ValueError("The original calibration does not contain the expected eight bands.")
    for row in plan:
        if row["mode"] != "original":
            continue
        expected = original.loc[row["band"]]
        actual = {
            "n_base_available": row["n_base_available"],
            "n_band_available": row["n_band_available"],
            "n_compared": len(row["query"]), "n_reference": len(row["reference"]),
        }
        if any(int(expected[key]) != value for key, value in actual.items()):
            raise ValueError(f"Original sampling counts do not match current inputs: {row['band']}")
        for key in ("mw_cut_low", "mw_cut_high"):
            if not np.isclose(row[key], expected[key]):
                raise ValueError(f"Original MW quantiles do not match current inputs: {row['band']}")


def feature_cache(model, smiles: list[str], path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if stored["smiles"].tolist() != smiles:
                raise ValueError(f"Molecular order changed for feature cache: {path}")
            features = stored["chemnet"]
            descriptors = stored["descriptors"]
    else:
        print(f"  ChemNet inference for {len(smiles)} unique molecules", flush=True)
        features = model.get_predictions(smiles)
        descriptors = scale_fdd_descriptors(descriptor_matrix(smiles, fdd_descriptor_vector))
        if len(features) != len(smiles) or len(descriptors) != len(smiles):
            raise ValueError("An input molecule was lost during feature calculation.")
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, smiles=np.asarray(smiles), chemnet=features, descriptors=descriptors)
        temporary.replace(path)
    if not np.isfinite(features).all() or not np.isfinite(descriptors).all():
        raise ValueError(f"Non-finite molecular features in {path}")
    return features, descriptors


def evaluate_plan(model, plan: list[dict], features: np.ndarray, descriptors: np.ndarray,
                  smiles: list[str], old: pd.DataFrame) -> pd.DataFrame:
    index = {smi: i for i, smi in enumerate(smiles)}
    old = old.set_index("band", verify_integrity=True)
    statistics = {}
    distances = {}
    results = []

    def stats(indices):
        if indices not in statistics:
            matrix = features[list(indices)]
            statistics[indices] = {"mu": matrix.mean(0), "sigma": np.cov(matrix.T)}
        return statistics[indices]

    for row in plan:
        a = tuple(index[smi] for smi in row["query"])
        b = tuple(index[smi] for smi in row["reference"])
        if (a, b) not in distances:
            distances[a, b] = (
                float(model.metric(stats(a), stats(b))),
                frechet_distance(descriptors[list(a)], descriptors[list(b)]),
            )
        fcd, fdd = distances[a, b]
        if not np.isfinite([fcd, fdd]).all():
            raise ValueError(f"Distance calculation failed for {row['seed']} / {row['band']} / {row['mode']}")
        original = old.loc[row["band"]]
        if row["mode"] == "original":
            if not np.isclose(fcd, original.fcd, rtol=0.002, atol=0.002):
                raise ValueError(f"Could not reproduce original FCD for {row['band']}: {fcd} vs {original.fcd}")
            if not np.isclose(fdd, original.fdd, rtol=0.002, atol=0.00001):
                raise ValueError(f"Could not reproduce original FDD for {row['band']}: {fdd} vs {original.fdd}")
        results.append({
            **{key: value for key, value in row.items() if key not in {"query", "reference"}},
            "n_query": len(a), "n_reference": len(b), "n_overlap": len(set(a) & set(b)),
            "fcd": fcd, "fdd": fdd,
            "original_saved_fcd": original.fcd, "original_saved_fdd": original.fdd,
            "fcd_change_from_original": fcd - original.fcd,
            "fdd_change_from_original": fdd - original.fdd,
        })
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generators", nargs="+", choices=GENERATORS, default=list(GENERATORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/fcd_calibration_size_check"))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--min-size", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.sample_size < args.min_size or args.min_size < 2 or args.threads < 1:
        parser.error("Require sample-size >= min-size >= 2 and positive threads.")
    if len(args.seeds) != len(set(args.seeds)) or len(args.generators) != len(set(args.generators)):
        parser.error("Do not repeat seeds or generators.")
    import torch
    from fcd_torch import FCD
    from fcd_torch import fcd as fcd_module

    torch.set_num_threads(args.threads)
    model_path = Path(fcd_module.__file__).parent / "ChemNet_v0.13_pretrained.pt"
    model = FCD(device=args.device, n_jobs=args.jobs, model_path=str(model_path))
    model_hash = digest(model_path)
    runtime = {name: version(name) for name in ["fcd-torch", "numpy", "scipy", "rdkit", "torch"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for generator in args.generators:
        source = Path(GENERATORS[generator])
        old_path = source / "analysis/fcd_mw_calibration/mw_fcd_calibration_metrics.csv"
        old = pd.read_csv(old_path)
        for seed in args.seeds:
            input_path = source / f"seed_{seed}/base_samples.csv"
            folder = args.output_dir / generator / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            config = {
                "definition": "mw_calibration_size_check_v1", "generator": generator, "seed": seed,
                "source_sha256": digest(input_path), "original_metrics_sha256": digest(old_path),
                "model_sha256": model_hash, "sample_size": args.sample_size,
                "min_size": args.min_size, "runtime": runtime, "device": args.device,
            }
            manifest = folder / "manifest.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != config:
                    raise ValueError(f"Inputs or configuration changed: {folder}; use a new output directory.")
            elif any(folder.iterdir()):
                raise ValueError(f"Non-empty output directory has no manifest: {folder}")
            else:
                write_json(manifest, config)
            completed = folder / "metrics.csv"
            if completed.exists():
                frame = pd.read_csv(completed)
                print(f"Reusing {generator} seed {seed}", flush=True)
            else:
                print(f"Analyzing {generator} seed {seed}", flush=True)
                base_frame = molecular_weight_frame(read_sample_smiles(input_path))
                plan = make_comparisons(base_frame, seed, args.sample_size, args.min_size)
                original = old[old.seed.eq(seed)]
                validate_original(plan, original)
                smiles = sorted({smi for row in plan for field in ("query", "reference") for smi in row[field]})
                features, descriptors = feature_cache(model, smiles, folder / "molecular_features.npz")
                frame = evaluate_plan(model, plan, features, descriptors, smiles, original)
                frame.insert(0, "generator", generator)
                write_csv(completed, frame)
                print(f"Completed {generator} seed {seed}; common n={frame.common_size.iloc[0]}", flush=True)
            all_frames.append(frame)
            write_csv(args.output_dir / "comparison_metrics.partial.csv", pd.concat(all_frames, ignore_index=True))
    metrics = pd.concat(all_frames, ignore_index=True)
    write_csv(args.output_dir / "comparison_metrics.csv", metrics)
    summary = metrics.groupby(["generator", "band", "band_label", "mode"])[
        ["fcd", "fdd", "fcd_change_from_original", "fdd_change_from_original", "n_query", "n_reference", "n_overlap"]
    ].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    write_csv(args.output_dir / "comparison_summary.csv", summary.reset_index())
    write_json(args.output_dir / "comparison_complete.json", {
        "generators": args.generators, "seeds": args.seeds, "modes": ["original", "pair_matched", "common_size"],
        "note": "Exploratory comparison only. Query and reference can overlap within the same source pool. Original paper outputs are unchanged.",
    })
    print(f"Wrote isolated calibration comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

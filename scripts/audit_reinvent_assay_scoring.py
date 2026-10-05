"""Audit saved REINVENT assay outputs against current SMARTS without replacing them."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from rdkit import Chem

from neon_molgen.scoring import LIABILITY_SMARTS, make_liability_patterns

SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
MODELS = ["base", "positive", "neon_lambda_1", "random_neon_lambda_1"]


def score_records(smiles, patterns, cache):
    records = []
    for smi in smiles:
        if smi not in cache:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("A previously valid molecule cannot be parsed by the current RDKit.")
            hits = {name: mol.HasSubstructMatch(query) for name, query in patterns.items()}
            cache[smi] = {
                "canonical_smiles": Chem.MolToSmiles(mol),
                "nitro_hit": hits["nitro"],
                "without_nitro_hit": any(value for name, value in hits.items() if name != "nitro"),
                "current_hit": any(hits.values()),
            }
        records.append(cache[smi])
    return pd.DataFrame(records)


def audit_seed(task):
    root, seed = task
    patterns = make_liability_patterns()["assay_interference"]
    metrics = pd.read_csv(root / "objective_metrics.csv")
    metrics = metrics[metrics.seed.eq(seed)].set_index("model", verify_integrity=True)
    folder = root / f"seed_{seed}"
    cache, rows, selections, sources = {}, [], [], {}
    for model in MODELS:
        path = folder / f"{model}_samples.csv"
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path)
        frame = frame[frame.valid.eq(True)]
        column = "canonical_smiles" if "canonical_smiles" in frame else "smiles"
        scored = score_records(frame[column].tolist(), patterns, cache)
        old = metrics.loc[model]
        if len(scored) != int(old.n_valid):
            raise ValueError(f"Saved valid count differs from molecule file: {path}")
        unique = scored.drop_duplicates("canonical_smiles")
        rows.append({
            "seed": seed, "model": model, "n_sampled": old.n_sampled,
            "n_valid": len(scored), "saved_hit_fraction": old.assay_interference_hit_fraction,
            "current_hit_fraction": scored.current_hit.mean(),
            "nitro_hit_fraction": scored.nitro_hit.mean(),
            "without_nitro_hit_fraction": scored.without_nitro_hit.mean(),
            "saved_usable_yield": old.usable_yield,
            "current_usable_yield": (~unique.current_hit).sum() / old.n_sampled,
        })
    for name in ["bad", "good"]:
        path = folder / "selection" / f"{name}_smiles.smi"
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        scored = score_records(path.read_text().splitlines(), patterns, cache)
        selections.append({
            "seed": seed, "selection": name, "n": len(scored),
            "current_hit_count": int(scored.current_hit.sum()),
            "nitro_hit_count": int(scored.nitro_hit.sum()),
            "nitro_only_count": int((scored.nitro_hit & ~scored.without_nitro_hit).sum()),
        })
    print(f"Audited assay-interference seed {seed}", flush=True)
    return rows, selections, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/external/reinvent4/assay_interference_replicates"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/reinvent_base_budget_check"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive jobs and distinct seeds.")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(audit_seed, [(args.results_dir, seed) for seed in args.seeds]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row for rows, _, _ in results for row in rows]).to_csv(args.output_dir / "assay_scoring_audit.csv", index=False)
    pd.DataFrame([row for _, rows, _ in results for row in rows]).to_csv(args.output_dir / "assay_selection_audit.csv", index=False)
    sources = {path: digest for _, _, files in results for path, digest in files.items()}
    path = args.results_dir / "objective_metrics.csv"
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / "assay_audit_manifest.json").write_text(json.dumps({
        "seeds": args.seeds, "models": MODELS, "source_sha256": sources,
        "smarts": LIABILITY_SMARTS["assay_interference"],
        "note": "Diagnostic rescoring only. Original outputs, training selections, and paper files are unchanged.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

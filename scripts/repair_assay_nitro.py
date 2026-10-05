"""Invalidate audited legacy nitro results and repair only affected assay runs.

Preparation is a dry run unless --apply is supplied. No GPU work is performed
by preparation. Existing base pools and verified evaluations are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = (13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
STALE_SEEDS = SEEDS[:-1]
STALE_TRANSFORMER_NE = (29, 31, 37, 41, 43)
ROOTS = {
    "transformer": Path("results/objectives/guacamol_transformer_assay_interference_lastblock_final"),
    "reinvent": Path("results/external/reinvent4/assay_interference_replicates"),
}
CONFIGS = {
    "transformer": Path("configs/objectives/guacamol_transformer_assay_interference_lastblock_final_from_base.json"),
    "reinvent": Path("configs/reinvent/reinvent4_assay_interference_removal.json"),
}
MANIFEST_DIR = Path("results/cleanup_manifests/assay_nitro_v2")
MARKER = "assay_nitro_repair.json"
DEFINITION = "assay_nitro_dependency_repair_v1"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def snapshot(paths) -> dict[str, str]:
    return {str(path): file_hash(path) for path in sorted(set(paths))}


def verify_hashes(files: dict[str, str]) -> None:
    for name, digest in files.items():
        path = Path(name)
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f"Missing or changed provenance input/output: {path}")


def regular_files(folder: Path) -> list[Path]:
    if folder.is_symlink():
        raise ValueError(f"Refusing to traverse a symlink: {folder}")
    files = []
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing to clean a result tree containing a symlink: {path}")
        if path.is_file():
            files.append(path)
    return files


def deletion_paths(kind: str, seed: int, folder: Path) -> list[Path]:
    """Explicit policy from the row-level audit, not a timestamp heuristic."""
    files = regular_files(folder)
    if kind == "reinvent":
        keep = {
            "baseline/prior_samples.csv", "baseline/sample_base.log",
            "baseline/sample_base.json", "configs/sample_base.toml", MARKER,
        }
        return [p for p in files if p.relative_to(folder).as_posix() not in keep]
    if seed == 11:
        return [p for p in files if p.name != MARKER]
    stale = {"base_samples.csv", "bad_samples.csv", "positive_samples.csv",
             "positive.pt", "positive_training_history.csv", "summary.csv",
             "random_control_refresh_complete.json"}
    if seed in STALE_TRANSFORMER_NE:
        stale.update({"bad.pt", "bad_training_history.csv"})
    return [p for p in files if p.name in stale or (
        seed in STALE_TRANSFORMER_NE
        and p.name.startswith(("neon_", "random_neon_"))
    )]


def input_paths(kind: str, seed: int) -> list[Path]:
    common = [CONFIGS[kind], Path("neon_molgen/scoring.py"), Path("scripts/repair_assay_nitro.py")]
    if kind == "reinvent":
        return common + [
            Path("external/REINVENT4/priors/reinvent.prior"),
            ROOTS[kind] / f"seed_{seed}/baseline/prior_samples.csv",
            Path("scripts/run_reinvent_reactive_replicates.py"),
            Path("scripts/reinvent_negative_extrapolate.py"),
            ROOTS[kind] / "run_config.json",
        ]
    source = Path("results/guacamol_transformer_base")
    folder = ROOTS[kind] / f"seed_{seed}"
    return common + [
        source / f"seed_{seed}/base.pt", source / f"seed_{seed}/base_samples.csv",
        source / "reference_canonical_smiles.smi",
        *[folder / f"{label}_smiles.smi" for label in ("bad", "good", "random")],
        folder / "selection_summary.json",
        Path("scripts/run_objective_from_base.py"),
        Path("neon_molgen/train.py"), Path("neon_molgen/model.py"),
    ]


def downstream_files() -> list[Path]:
    paths = []
    for root in ROOTS.values():
        paths.extend(root.glob("*.csv"))
        paths.extend(regular_files(root / "analysis"))
    for name in ("confirmatory_endpoint_values.csv", "confirmatory_paired_comparisons.csv",
                 "liability_paired_comparisons.csv", "confirmatory_statistics_report.md"):
        paths.append(Path("results/paper_statistics") / name)
    paths.append(Path("results/publication/tables/si_confirmatory_liability_primary.tex"))
    return sorted({p for p in paths if p.is_file()})


def prepare(apply: bool) -> None:
    plans = []
    for kind, root in ROOTS.items():
        for seed in (11, *STALE_SEEDS):
            folder = root / f"seed_{seed}"
            marker = folder / MARKER
            if marker.exists():
                previous = json.loads(marker.read_text())
                if previous.get("definition") != DEFINITION:
                    raise ValueError(f"Unknown cleanup manifest: {marker}")
                if previous["status"] == "planned":
                    plans.append((marker, previous))
                    continue
                print(f"Already prepared: {kind} seed {seed}")
                continue
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            print(f"Checking {kind} seed {seed} inputs and deletion inventory...", flush=True)
            deleted = deletion_paths(kind, seed, folder)
            # Retain hashes rather than copies of scientifically invalid results.
            retained = [p for p in regular_files(folder) if p not in deleted]
            inputs = input_paths(kind, seed) if seed != 11 else []
            plan = {
                "definition": DEFINITION, "model": kind, "seed": seed,
                "status": "planned", "retired": seed == 11,
                "reason": "Legacy nitro scoring/selection or dependent task-vector normalization",
                "inputs": snapshot(inputs), "retained": snapshot(retained),
                "deleted": {str(p): {"sha256": file_hash(p), "bytes": p.stat().st_size}
                            for p in deleted},
            }
            plans.append((marker, plan))
    global_path = MANIFEST_DIR / "downstream.json"
    if global_path.exists():
        global_plan = json.loads(global_path.read_text())
    else:
        global_plan = {"status": "planned", "deleted": snapshot(downstream_files())}
    downstream = ([Path(p) for p in global_plan["deleted"] if Path(p).exists()]
                  if global_plan["status"] == "planned" else [])
    total = sum(len(plan["deleted"]) for _, plan in plans) + len(downstream)
    size = sum(v["bytes"] for _, plan in plans for v in plan["deleted"].values())
    size += sum(p.stat().st_size for p in downstream)
    print(f"{'DELETE' if apply else 'DRY RUN'}: {total} files, {size / 2**30:.3f} GiB")
    for _, plan in plans:
        print(f"  {plan['model']} seed {plan['seed']}: {len(plan['deleted'])} files")
    if not apply:
        return
    # Complete preflight before deleting anything; persist the inventory first.
    for _, plan in plans:
        verify_hashes(plan["inputs"])
        verify_hashes(plan["retained"])
        verify_hashes({p: info["sha256"] for p, info in plan["deleted"].items() if Path(p).exists()})
    verify_hashes({str(p): global_plan["deleted"][str(p)] for p in downstream})
    for marker, plan in plans:
        write_json(MANIFEST_DIR / f"{plan['model']}_seed_{plan['seed']}.json", plan)
        write_json(marker, plan)
    if not global_path.exists():
        write_json(global_path, global_plan)
    for marker, plan in plans:
        for name, info in plan["deleted"].items():
            path = Path(name)
            if not path.exists():
                continue
            if file_hash(path) != info["sha256"]:
                raise ValueError(f"File changed during cleanup: {path}")
            path.unlink()
        plan["status"] = "retired" if plan["retired"] else "cleaned"
        write_json(marker, plan)
        print(f"Cleaned {plan['model']} seed {plan['seed']}", flush=True)
    for path in downstream:
        path.unlink()
    global_plan["status"] = "complete"
    write_json(global_path, global_plan)


def validate_selections(folder: Path, *, reinvent: bool) -> None:
    from rdkit import Chem

    from neon_molgen.scoring import make_liability_patterns

    queries = list(make_liability_patterns()["assay_interference"].values())
    selection = folder / "selection" if reinvent else folder
    counts = []
    for label, expected in (("bad", True), ("good", False), ("random", None)):
        values = (selection / f"{label}_smiles.smi").read_text().splitlines()
        counts.append(len(values))
        for smiles in values:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid training molecule in {selection}/{label}_smiles.smi")
            if expected is not None and any(mol.HasSubstructMatch(q) for q in queries) != expected:
                raise ValueError(f"Selection violates current SMARTS: {selection}/{label}_smiles.smi")
    if not counts[0] or len(set(counts)) != 1:
        raise ValueError(f"Expected nonempty, size-matched training sets, found {counts}")


def reset_unfinished(folder: Path, plan: dict) -> None:
    """Only completed seeds are resumable; never accept half-written outputs."""
    verify_hashes(plan["inputs"])
    verify_hashes(plan["retained"])
    for path in regular_files(folder):
        if path.name != MARKER and str(path) not in plan["retained"]:
            path.unlink()


def repair_transformer(seed: int, folder: Path, device: str) -> None:
    import pandas as pd

    from neon_molgen.checkpoint import load_checkpoint
    from neon_molgen.model import clone_model, negative_extrapolate, parameter_update_norm
    from neon_molgen.scoring import summarize_scores
    from neon_molgen.train import train_model_with_history
    from scripts import run_objective_from_base as runner

    config = json.loads(CONFIGS["transformer"].read_text())
    validate_selections(folder, reinvent=False)
    base_dir = Path(config["base_results_dir"]) / f"seed_{seed}"
    base, tokenizer, base_config = load_checkpoint(base_dir / "base.pt", device=device)
    reference = runner.load_reference_canonical_smiles(base_dir, base_config)
    sampling_config = {**base_config, "sampling": {**base_config["sampling"], **config["sampling"]}}
    run_config = {**config, "seed": seed, "output_dir": str(folder)}
    training = config["training"]
    trained = {}

    def train(stage):
        if stage in trained:
            return trained[stage]
        label, offset = {"bad": ("bad", 1), "positive": ("good", 2), "random": ("random", 3)}[stage]
        runner.seed_everything(seed + offset)
        molecules = (folder / f"{label}_smiles.smi").read_text().splitlines()
        model, history = train_model_with_history(
            clone_model(base), molecules, tokenizer,
            epochs=training["positive_epochs" if stage == "positive" else "bad_epochs"],
            batch_size=training["batch_size"], learning_rate=training["learning_rate"],
            device=device, desc=f"assay nitro repair {stage} seed={seed}", seed=seed + offset,
        )
        # Preserve verified historical records; retain new ones only when stale ones were removed.
        if not (folder / f"{stage}_training_history.csv").exists():
            runner.save_and_log_training_history(run_config, history, stage=stage, seed=seed, output_dir=folder)
        trained[stage] = model
        return model

    def evaluate(name, model):
        return runner.evaluate_model(
            name, model, tokenizer, sampling_config, device=device, output_dir=folder,
            reference_canonical_smiles=reference, reference_scaffold_counts={},
            run_config=run_config, sampling_seed=runner.derive_seed(seed, f"sample:{name}"),
        )

    if not (folder / "base_samples.csv").exists():
        runner.score_cached_samples(pd.read_csv(base_dir / "base_samples.csv")).to_csv(
            folder / "base_samples.csv", index=False)
    for stage, name in (("positive", "positive"), ("bad", "bad"), ("random", "random_finetune")):
        if not (folder / f"{name}_samples.csv").exists():
            evaluate(name, train(stage))
    names = ["base", "positive", "bad", "random_finetune"]
    for scope in runner.neon_scopes(config):
        tag = f"_{scope['name']}" if scope["name"] else ""
        norm_path = folder / f"random_neon_norm_matching{tag}.json"
        for scale in config["neon"]["lambda_values"]:
            for label, stage in (("neon", "bad"), ("random_neon", "random")):
                name = runner.scoped_model_name(label, scope, scale)
                names.append(name)
                if (folder / f"{name}_samples.csv").exists():
                    continue
                bad = train("bad")
                model = negative_extrapolate(
                    base, train(stage), float(scale), include_patterns=scope["include"],
                    exclude_patterns=scope["exclude"],
                    **({"norm_match_to": bad} if stage == "random" else {}),
                )
                evaluate(name, model)
                del model
                if stage == "random":
                    bad_norm = parameter_update_norm(base, bad, include_patterns=scope["include"], exclude_patterns=scope["exclude"])
                    rand_norm = parameter_update_norm(base, train("random"), include_patterns=scope["include"], exclude_patterns=scope["exclude"])
                    if rand_norm == 0:
                        raise ValueError("Random training produced a zero update")
                    write_json(norm_path, {
                        "definition": "global_scope_l2_v1", "scope": scope["name"],
                        "bad_direction_source": "nitro_corrected_selection_retraining",
                        "standard_neon_refreshed": True, "bad_update_norm": bad_norm,
                        "random_update_norm": rand_norm, "random_direction_scale": bad_norm / rand_norm,
                    })
    rows = []
    for name in names:
        scores = pd.read_csv(folder / f"{name}_samples.csv")
        rows.append({"seed": seed, "model": name, **summarize_scores(
            scores, reference_canonical_smiles=reference,
            liability_free_column="assay_interference_hit",
            n_sampled=len(scores) if name == "base" else config["sampling"]["eval_samples"],
        )})
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    write_json(folder / "random_control_refresh_complete.json", {
        "definition": runner.RANDOM_CONTROL_REFRESH_DEFINITION, "seed": seed,
        "objective": config["objective"]["name"], "standard_neon_refreshed": True,
        "deleted_transient_checkpoints": True,
    })


def repair_reinvent(seed: int, folder: Path, device: str) -> None:
    import pandas as pd

    from neon_molgen.scoring import score_smiles, summarize_scores
    from scripts import run_reinvent_reactive_replicates as runner

    config = json.loads(CONFIGS["reinvent"].read_text())
    baseline = folder / "baseline"
    scores = score_smiles(runner.read_smiles_table(baseline / "prior_samples.csv", smiles_column="SMILES"))
    scores.to_csv(baseline / "prior_samples_scored.csv", index=False)
    pd.DataFrame([summarize_scores(scores, liability_free_column="assay_interference_hit", n_sampled=50000)]).to_csv(
        baseline / "prior_samples_summary.csv", index=False)
    # Use the original runner and settings. The cleanup has removed every stale
    # downstream file, while the freshly rescored raw 50k pool is reused.
    args = argparse.Namespace(
        config=str(CONFIGS["reinvent"]), prior="external/REINVENT4/priors/reinvent.prior",
        output_dir=str(ROOTS["reinvent"]), baseline_samples=50000, eval_samples=10000,
        lambda_values=[0.25, 0.5, 0.75, 1.0], device=device, tl_epochs=2,
        batch_size=256, learning_rate=1e-4, skip_existing=True, skip_analysis=True,
        skip_random_controls=False, delete_random_control_models=True,
        neon_scope_name="full_model", neon_include_prefixes=None,
        assay_nitro_repair=True,
    )
    runner.run_seed(args, config, seed)
    validate_selections(folder, reinvent=True)
    # Model arithmetic is fully recoverable from input hashes, settings and seeds.
    # Do not retain dozens of large, disposable edited checkpoints.
    for path in (folder / "models").glob("*.model*"):
        path.unlink()


def validate_outputs(kind: str, folder: Path) -> None:
    import numpy as np
    import pandas as pd
    from rdkit import Chem

    from neon_molgen.scoring import make_liability_patterns

    patterns = make_liability_patterns()["assay_interference"]
    files = folder.glob("*_samples.csv") if kind == "transformer" else (folder / "scores").glob("scores_*.csv")
    cache = {}
    count = 0
    for path in files:
        frame = pd.read_csv(path)
        valid = frame[frame.valid.eq(True)]
        column = "smiles" if "smiles" in valid else "canonical_smiles"
        expected = []
        for smiles in valid[column]:
            if smiles not in cache:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Previously valid molecule cannot be parsed: {path}")
                hits = {name: mol.HasSubstructMatch(query) for name, query in patterns.items()}
                cache[smiles] = (hits["nitro"], any(hits.values()))
            expected.append(cache[smiles])
        actual = valid[["assay_interference_nitro_hit", "assay_interference_hit"]].to_numpy(dtype=bool)
        if not np.array_equal(np.asarray(expected, dtype=bool).reshape(-1, 2), actual):
            raise ValueError(f"Saved scores disagree with current nitro/assay definition: {path}")
        count += 1
    expected_count = 11 if kind == "transformer" else 12
    # Transformer initial_samples.csv is also checked; ten evaluation arms + pool.
    if count != expected_count:
        raise ValueError(f"Incomplete outputs in {folder}: {count}, expected {expected_count}")


def run(kind: str, device: str) -> None:
    import pandas as pd

    root = ROOTS[kind]
    for seed in STALE_SEEDS:
        folder = root / f"seed_{seed}"
        marker = folder / MARKER
        if not marker.exists():
            raise RuntimeError("Run make paper-assay-nitro-clean first; refusing unprepared legacy results")
        plan = json.loads(marker.read_text())
        if plan["definition"] != DEFINITION or plan["retired"]:
            raise ValueError(f"Unexpected repair plan: {marker}")
        verify_hashes(plan["inputs"])
        verify_hashes(plan["retained"])
        if plan["status"] == "complete":
            verify_hashes(plan["outputs"])
            print(f"[{kind} {seed}] verified completed repair; skipping", flush=True)
            continue
        reset_unfinished(folder, plan)
        plan["status"] = "running"
        write_json(marker, plan)
        print(f"[{kind} {seed}] rebuilding invalidated assay results", flush=True)
        if kind == "transformer":
            repair_transformer(seed, folder, device)
        else:
            repair_reinvent(seed, folder, device)
        validate_outputs(kind, folder)
        verify_hashes(plan["retained"])
        plan["outputs"] = snapshot(p for p in regular_files(folder) if p.name != MARKER)
        plan["status"] = "complete"
        write_json(marker, plan)
    # Aggregate only the ten confirmatory seeds, including unchanged seed 47.
    if kind == "transformer":
        from scripts.run_objective_from_base import summarize_objective
        summaries = [pd.read_csv(root / f"seed_{seed}/summary.csv") for seed in SEEDS]
        summarize_objective(pd.concat(summaries, ignore_index=True), root)
    else:
        from scripts.run_reinvent_reactive_replicates import aggregate_outputs
        aggregate_outputs(root, seeds=set(SEEDS))


def refuse_pending_repair(folder: Path) -> None:
    marker = folder / MARKER
    if marker.exists():
        plan = json.loads(marker.read_text())
        if plan["status"] != "complete":
            raise RuntimeError(f"Nitro repair pending for {folder}. Use make paper-assay-nitro-repair.")
        verify_hashes(plan["inputs"])
        verify_hashes(plan["outputs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "transformer", "reinvent"])
    parser.add_argument("--apply", action="store_true", help="Actually delete the audited stale files")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.apply)
    else:
        if args.apply:
            parser.error("--apply is only valid with prepare")
        run(args.action, args.device)


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import repair_assay_nitro as repair


def touch_files(folder, names):
    paths = []
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n")
        paths.append(path)
    return paths


def test_reinvent_cleanup_preserves_raw_pool_not_stale_selections(tmp_path):
    touch_files(tmp_path, [
        "baseline/prior_samples.csv", "baseline/prior_samples_scored.csv",
        "selection/good_smiles.smi", "models/random_tuned.model",
        "samples/samples_random_neon_lambda_1.csv", "scores/summary.csv",
        "random_neon_lambda_1_samples.csv", "base_samples.csv",
    ])
    deleted = {p.relative_to(tmp_path).as_posix() for p in repair.deletion_paths("reinvent", 13, tmp_path)}
    assert "baseline/prior_samples.csv" not in deleted
    assert len(deleted) == 7


@pytest.mark.parametrize("seed,redo_ne", [(13, False), (23, False), (29, True), (43, True)])
def test_transformer_cleanup_preserves_verified_arms(tmp_path, seed, redo_ne):
    touch_files(tmp_path, [
        "base_samples.csv", "bad_samples.csv", "positive_samples.csv", "positive_training_history.csv",
        "bad.pt", "bad_training_history.csv", "summary.csv", "initial_samples.csv", "good_smiles.smi",
        "random_finetune_samples.csv", "random.pt", "random_training_history.csv",
        "neon_last_block_output_lambda_1.0_samples.csv",
        "random_neon_last_block_output_lambda_1.0_samples.csv",
        "random_neon_norm_matching_last_block_output.json",
    ])
    deleted = {p.name for p in repair.deletion_paths("transformer", seed, tmp_path)}
    assert {"base_samples.csv", "bad_samples.csv", "positive_samples.csv", "summary.csv"} <= deleted
    assert "initial_samples.csv" not in deleted
    assert "good_smiles.smi" not in deleted
    assert "random_finetune_samples.csv" not in deleted
    assert "random.pt" not in deleted
    assert ("bad.pt" in deleted) == redo_ne
    assert ("neon_last_block_output_lambda_1.0_samples.csv" in deleted) == redo_ne
    assert ("random_neon_norm_matching_last_block_output.json" in deleted) == redo_ne


def test_development_cleanup_does_not_touch_base_generator(tmp_path):
    source = tmp_path / "base/base.pt"
    touch_files(tmp_path, ["base/base.pt", "objective/seed_11/bad_samples.csv"])
    assert repair.deletion_paths("transformer", 11, tmp_path / "objective/seed_11") == [
        tmp_path / "objective/seed_11/bad_samples.csv"]
    assert source.exists()


def test_reset_removes_uncommitted_files_only(tmp_path):
    good, partial = touch_files(tmp_path, ["verified_samples.csv", "half_written.csv"])
    plan = {"inputs": {}, "retained": repair.snapshot([good])}
    repair.write_json(tmp_path / repair.MARKER, plan)
    repair.reset_unfinished(tmp_path, plan)
    assert good.exists()
    assert not partial.exists()
    assert (tmp_path / repair.MARKER).exists()


def test_changed_input_aborts_before_reset(tmp_path):
    good, partial = touch_files(tmp_path, ["verified.csv", "partial.csv"])
    plan = {"inputs": {}, "retained": repair.snapshot([good])}
    good.write_text("changed")
    with pytest.raises(ValueError, match="provenance"):
        repair.reset_unfinished(tmp_path, plan)
    assert partial.exists()


def test_symlinks_are_not_followed(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    touch_files(target, ["keep.csv"])
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        repair.regular_files(link)


def test_ordinary_runner_refuses_pending_repair(tmp_path):
    repair.write_json(tmp_path / repair.MARKER, {"status": "cleaned"})
    with pytest.raises(RuntimeError, match="paper-assay-nitro-repair"):
        repair.refuse_pending_repair(tmp_path)


def test_prepare_is_dry_by_default_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("results/assay")
    folder = root / "seed_13"
    touch_files(folder, ["base_samples.csv", "positive_samples.csv", "random_finetune_samples.csv"])
    touch_files(root / "seed_11", ["summary.csv"])
    touch_files(root / "seed_47", ["positive_samples.csv"])
    dependency = touch_files(tmp_path, ["scoring.py"])[0]
    with patch.object(repair, "ROOTS", {"transformer": root}), \
         patch.object(repair, "STALE_SEEDS", (13,)), \
         patch.object(repair, "input_paths", return_value=[dependency]):
        repair.prepare(False)
        assert (folder / "positive_samples.csv").exists()
        assert not (folder / repair.MARKER).exists()
        repair.prepare(True)
        assert not (folder / "positive_samples.csv").exists()
        assert (folder / "random_finetune_samples.csv").exists()
        assert (root / "seed_47/positive_samples.csv").exists()
        (folder / "positive_samples.csv").write_text("new run")
        repair.prepare(True)
        assert (folder / "positive_samples.csv").read_text() == "new run"
        plan = json.loads((folder / repair.MARKER).read_text())
        assert plan["status"] == "cleaned"
        assert str(folder / "positive_samples.csv") in plan["deleted"]


def test_selection_validation_rejects_nitro_in_positive_set(tmp_path):
    (tmp_path / "bad_smiles.smi").write_text("C[N+](=O)[O-]\n")
    (tmp_path / "good_smiles.smi").write_text("CCO\n")
    (tmp_path / "random_smiles.smi").write_text("CC\n")
    repair.validate_selections(tmp_path, reinvent=False)
    (tmp_path / "good_smiles.smi").write_text("C[N+](=O)[O-]\n")
    with pytest.raises(ValueError, match="Selection violates"):
        repair.validate_selections(tmp_path, reinvent=False)


@pytest.mark.parametrize("keep_ne", [True, False])
def test_transformer_repair_reuses_valid_samples_and_original_stage_seeds(tmp_path, monkeypatch, keep_ne):
    import pandas as pd

    from neon_molgen import checkpoint, model, scoring, train
    from scripts import run_objective_from_base as runner

    monkeypatch.chdir(tmp_path)
    seed = 13 if keep_ne else 29
    folder = tmp_path / "seed"
    folder.mkdir()
    (folder / "bad_smiles.smi").write_text("C[N+](=O)[O-]\n")
    (folder / "good_smiles.smi").write_text("CCO\n")
    (folder / "random_smiles.smi").write_text("CC\n")
    config = {
        "base_results_dir": str(tmp_path / "base"), "objective": {"name": "assay_interference_removal"},
        "training": {"bad_epochs": 1, "positive_epochs": 1, "batch_size": 256, "learning_rate": 0.0005},
        "sampling": {"eval_samples": 10000, "batch_size": 512, "max_len": 150, "temperature": 0.75},
        "neon": {"lambda_values": [0.5, 0.75, 1.0], "parameter_scopes": [
            {"name": "last_block_output", "include": ["transformer.layers.3.", "output."]}]},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setitem(repair.CONFIGS, "transformer", config_path)
    frame = pd.DataFrame({"smiles": ["CC"], "valid": [True]})
    source = tmp_path / "base" / f"seed_{seed}"
    source.mkdir(parents=True)
    frame.to_csv(source / "base_samples.csv", index=False)
    frame.to_csv(folder / "random_finetune_samples.csv", index=False)
    if keep_ne:
        for lam in config["neon"]["lambda_values"]:
            for name in ("neon", "random_neon"):
                frame.to_csv(folder / f"{name}_last_block_output_lambda_{lam}_samples.csv", index=False)
    retained = repair.snapshot(folder.glob("*_samples.csv"))
    stages, evaluations = [], []

    def fake_train(*args, **kwargs):
        stages.append(kwargs)
        return object(), [{"epoch": 1, "loss_mean": 0.1, "n_batches": 1}]

    def fake_eval(name, *args, **kwargs):
        evaluations.append((name, kwargs))
        frame.to_csv(kwargs["output_dir"] / f"{name}_samples.csv", index=False)
        return {"model": name}

    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *a, **kw: (object(), object(), {"sampling": {}}))
    monkeypatch.setattr(runner, "load_reference_canonical_smiles", lambda *a: set())
    monkeypatch.setattr(runner, "score_cached_samples", lambda f: f)
    monkeypatch.setattr(model, "clone_model", lambda m: object())
    monkeypatch.setattr(model, "negative_extrapolate", lambda *a, **kw: object())
    monkeypatch.setattr(model, "parameter_update_norm", lambda *a, **kw: 2.0)
    monkeypatch.setattr(train, "train_model_with_history", fake_train)
    monkeypatch.setattr(runner, "evaluate_model", fake_eval)
    monkeypatch.setattr(scoring, "summarize_scores", lambda *a, **kw: {"n_sampled": kw["n_sampled"]})
    repair.repair_transformer(seed, folder, "cpu")
    repair.verify_hashes(retained)
    assert len(evaluations) == (2 if keep_ne else 8)
    assert {stage["seed"] for stage in stages} == ({seed + 1, seed + 2} if keep_ne else {seed + 1, seed + 2, seed + 3})
    assert all(stage["epochs"] == 1 and stage["learning_rate"] == 0.0005 for stage in stages)
    for name, kwargs in evaluations:
        assert kwargs["sampling_seed"] == runner.derive_seed(seed, f"sample:{name}")
    assert len(pd.read_csv(folder / "summary.csv")) == 10


def test_completed_seed_rejects_changed_scoring_source(tmp_path):
    source = touch_files(tmp_path, ["scoring.py"])[0]
    repair.write_json(tmp_path / repair.MARKER, {"status": "complete", "inputs": repair.snapshot([source]), "outputs": {}})
    repair.refuse_pending_repair(tmp_path)
    source.write_text("changed definition")
    with pytest.raises(ValueError, match="provenance"):
        repair.refuse_pending_repair(tmp_path)

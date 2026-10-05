import json

import numpy as np
import pandas as pd
import pytest

from neon_molgen.paper_data import (
    CALIBRATION,
    CALIBRATION_MODES,
    MW_BANDS,
    REINVENT_BASE,
    load_mw_calibration_comparisons,
    load_reference,
    load_reinvent_base,
    novelty_aware_usable_yield,
    replace_reinvent_base,
    replace_reinvent_base_diversity,
)

SEEDS = (13, 17)


def test_novelty_aware_yield_counts_unique_acceptable_outputs(tmp_path):
    sample = tmp_path / "samples.csv"
    pd.DataFrame({
        "canonical_smiles": ["CCO", "CCO", "CCN", "CCC", "CCCl", None, "", "CCO"],
        "valid": [True, True, True, True, True, False, True, True],
        "reactive_hit": [0, 0, 0, 0, 1, 0, 0, 1],
    }).to_csv(sample, index=False)
    reference = tmp_path / "reference.smi"
    reference.write_text("CCN\n\nCCN\n")
    assert novelty_aware_usable_yield(
        sample, liability_column="reactive_hit", reference=load_reference(reference),
    ) == 2 / 8


def test_novelty_aware_yield_rejects_missing_flags(tmp_path):
    sample = tmp_path / "samples.csv"
    pd.DataFrame({"canonical_smiles": ["CCO"], "valid": [True]}).to_csv(sample, index=False)
    with pytest.raises(ValueError, match="reactive_hit"):
        novelty_aware_usable_yield(sample, liability_column="reactive_hit", reference=set())


def test_plot_loader_reconstructs_missing_yields_without_changing_saved_rows(tmp_path, monkeypatch):
    from notebooks import paper_plot_data_loaders as loaders

    monkeypatch.setattr(loaders, "ROOT", tmp_path)
    monkeypatch.setattr(loaders, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(loaders, "OBJECTIVES", {"reactive": loaders.OBJECTIVES["reactive"]})
    meta = loaders.OBJECTIVES["reactive"]
    for architecture in ["rnn", "transformer"]:
        directory = loaders.RESULTS / "objectives" / meta[f"{architecture}_dir"]
        directory.mkdir(parents=True)
        metrics = directory / "objective_metrics.csv"
        pd.DataFrame({
            "seed": [13, 17, 11], "model": ["base", "base", "base"],
            "reactive_hit_fraction": [0.0, 0.0, 0.0],
            "usable_yield": [np.nan, 0.875, np.nan],
        }).to_csv(metrics, index=False)
        reference = loaders.RESULTS / f"guacamol_{architecture}_base/reference_canonical_smiles.smi"
        reference.parent.mkdir(parents=True)
        reference.write_text("CCN\n")
        samples = directory / "seed_13/base_samples.csv"
        samples.parent.mkdir(parents=True)
        pd.DataFrame({
            "canonical_smiles": ["CCO", "CCO", "CCN", "CCCl"],
            "valid": [True, True, True, True], "reactive_hit": [0, 0, 0, 1],
        }).to_csv(samples, index=False)
    before = {path: path.read_bytes() for path in loaders.RESULTS.rglob("objective_metrics.csv")}
    frame = loaders.load_guacamol_liability(seeds=list(SEEDS), objectives=["reactive"])
    assert len(frame) == 4
    assert frame.loc[frame.seed.eq(13), "usable_yield"].tolist() == [0.25, 0.25]
    assert frame.loc[frame.seed.eq(17), "usable_yield"].tolist() == [0.875, 0.875]
    assert set(frame.seed) == set(SEEDS)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.fixture
def baseline(tmp_path):
    folder = tmp_path / REINVENT_BASE
    folder.mkdir(parents=True)
    (folder / "evaluation_complete.json").write_text(json.dumps({
        "definition": "reinvent_base_fixed_budget_v1",
        "n_sampled": 10000,
        "seeds": SEEDS,
        "objectives": {"chelator": "chelator_hit"},
    }))
    frame = pd.DataFrame({
        "seed": SEEDS, "model": "base", "objective": "chelator",
        "n_sampled": 10000, "n_valid": [9800, 9810],
        "usable_yield": [0.88, 0.89], "chelator_hit_fraction": [0.10, 0.09],
    })
    frame.to_csv(folder / "objective_metrics.csv", index=False)
    return tmp_path, folder, frame


def test_base_overlay_preserves_edited_rows_and_source(baseline):
    root, _, base = baseline
    old_base = base.assign(n_sampled=50000, n_valid=49000)
    edited = base.assign(model="neon_lambda_1", usable_yield=0.95)
    original = pd.concat([old_base, edited], ignore_index=True)
    before = original.copy(deep=True)
    result = replace_reinvent_base(original, root, objective="chelator", seeds=SEEDS)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(result[result.model.ne("base")].reset_index(drop=True), edited)
    pd.testing.assert_frame_equal(result[result.model.eq("base")].reset_index(drop=True), base)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "budget", "nan"])
def test_reject_invalid_base_data(baseline, problem):
    root, folder, frame = baseline
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "missing":
        frame = frame.iloc[:1]
    elif problem == "budget":
        frame.loc[0, "n_sampled"] = 50000
    else:
        frame.loc[0, "usable_yield"] = np.nan
    frame.to_csv(folder / "objective_metrics.csv", index=False)
    with pytest.raises(ValueError):
        load_reinvent_base(root, objective="chelator", seeds=SEEDS)


def test_missing_marker_does_not_fall_back(baseline):
    root, folder, _ = baseline
    (folder / "evaluation_complete.json").unlink()
    with pytest.raises(FileNotFoundError, match="paper-reinvent-base-budget"):
        load_reinvent_base(root, objective="chelator", seeds=SEEDS)


def test_statistics_uses_budget_matched_base(baseline, monkeypatch):
    from scripts import analyze_paper_statistics as statistics

    root, _, base = baseline
    config = statistics.GENERATOR_CONFIG["REINVENT prior"]
    monkeypatch.setattr(statistics, "GENERATOR_CONFIG", {"REINVENT prior": config})
    monkeypatch.setattr(statistics, "OBJECTIVES", {"chelator": statistics.OBJECTIVES["chelator"]})
    monkeypatch.setattr(statistics, "PAPER_SEEDS", list(SEEDS))
    folder = root / config["root"] / "chelator_replicates"
    folder.mkdir(parents=True)
    historical = pd.concat([
        base.assign(model=model, n_sampled=50000 if model == "base" else 10000,
                    usable_yield=0.80 if model == "base" else 0.95)
        for model in config["models"].values()
    ])
    historical.to_csv(folder / "objective_metrics.csv", index=False)
    endpoints = statistics.load_liability_endpoints(root)
    observed = endpoints[endpoints.model.eq("base") & endpoints.metric.eq("usable_yield")]
    assert observed.value.tolist() == [0.88, 0.89]


def test_diversity_overlay_does_not_mix_cluster_spaces(baseline):
    root, folder, frame = baseline
    diversity = frame[["seed", "model", "n_valid"]].assign(
        unique_scaffold_fraction=[0.87, 0.88], cluster_entropy=[2.0, 3.0],
    )
    path = folder / "analysis/diversity/diversity_metrics.csv"
    path.parent.mkdir(parents=True)
    diversity.to_csv(path, index=False)
    edited = diversity.assign(model="neon_lambda_1", unique_scaffold_fraction=0.90)
    result = replace_reinvent_base_diversity(pd.concat([diversity, edited]), root, seeds=SEEDS)
    assert result[result.model.eq("base")].cluster_entropy.isna().all()
    assert result[result.model.eq("base")].unique_scaffold_fraction.tolist() == [0.87, 0.88]
    pd.testing.assert_frame_equal(result[result.model.ne("base")].reset_index(drop=True), edited)
    diversity.loc[0, "n_valid"] -= 1
    diversity.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        replace_reinvent_base_diversity(edited, root, seeds=SEEDS)


@pytest.fixture
def calibration(tmp_path):
    frame = pd.MultiIndex.from_product(
        [["rnn", "transformer", "reinvent"], SEEDS, MW_BANDS, CALIBRATION_MODES],
        names=["generator", "seed", "band", "mode"],
    ).to_frame(index=False)
    frame = frame.assign(fcd=1.2, fdd=0.03, n_query=900, n_reference=900,
                         common_size=900, n_overlap=80)
    path = tmp_path / CALIBRATION
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)
    return tmp_path, path, frame


def test_complete_calibration(calibration):
    root, _, frame = calibration
    pd.testing.assert_frame_equal(load_mw_calibration_comparisons(root, seeds=SEEDS), frame)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unequal", "varying", "overlap"])
def test_reject_inconsistent_calibration(calibration, problem):
    root, path, frame = calibration
    if problem == "missing":
        frame = frame.iloc[1:]
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "unequal":
        frame.loc[1, "n_reference"] = 5000
    elif problem == "varying":
        frame.loc[2, ["n_query", "n_reference", "common_size"]] = 800
    else:
        frame.loc[0, "n_overlap"] = 901
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_mw_calibration_comparisons(root, seeds=SEEDS)

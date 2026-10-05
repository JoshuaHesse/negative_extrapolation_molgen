from pathlib import Path

import pytest

from scripts import build_zenodo_archive as archive


def test_archive_includes_new_paper_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    manifest = {"seed_result_groups": [], "confirmatory_seeds": [13, 17]}
    for root in archive.result_roots(manifest):
        (tmp_path / root).mkdir(parents=True)
    for relative in archive.SUPPLEMENTARY_RESULT_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    required = [
        Path("results/external/reinvent4/base_evaluation_10000/seed_13/base_samples.csv"),
        Path("results/analysis/fcd_calibration_size_check/comparison_metrics.csv"),
        Path("results/analysis/reinvent_base_budget_check/assay_scoring_audit.csv"),
    ]
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sample\n")
    excluded = Path("results/external/reinvent4/base_evaluation_10000/seed_11/base_samples.csv")
    (tmp_path / excluded).parent.mkdir(parents=True)
    (tmp_path / excluded).write_text("development\n")
    selected = set(archive.collect_files(manifest))
    assert set(required).issubset(selected)
    assert excluded not in selected


@pytest.mark.parametrize("name", ["data.zip", "data.tar.zst"])
def test_archive_root_name(name):
    assert archive.archive_root_name(name) == "data"


@pytest.mark.parametrize("name", ["../data.zip", "data.tar", ".zip"])
def test_archive_root_name_rejects_invalid_names(name):
    with pytest.raises(ValueError):
        archive.archive_root_name(name)


def make_staging(tmp_path):
    root_name = "paper_data"
    root = tmp_path / root_name
    payload = root / "results/sample.csv"
    payload.parent.mkdir(parents=True)
    payload.write_text("smiles\nCCO\n")
    (root / "SHA256SUMS").write_text(f"{archive.sha256(payload)}  results/sample.csv\n")
    for name in ("README.md", "MANIFEST.tsv", "archive_metadata.json", "reproducibility_manifest.json"):
        (root / name).write_text("{}\n")
    return root_name, payload


def test_zip_archive_round_trip(tmp_path):
    root_name, _ = make_staging(tmp_path)
    destination = tmp_path / "data.zip"
    archive.create_archive(tmp_path, root_name, destination, 1)
    archive.verify_archive(destination, root_name, 1)


def test_zip_archive_rejects_changed_payload(tmp_path):
    root_name, payload = make_staging(tmp_path)
    payload.write_text("different data\n")
    destination = tmp_path / "data.zip"
    archive.create_archive(tmp_path, root_name, destination, 1)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        archive.verify_archive(destination, root_name, 1)


def test_zip_archive_rejects_unlisted_payload(tmp_path):
    root_name, _ = make_staging(tmp_path)
    (tmp_path / root_name / "results/extra.csv").write_text("extra\n")
    destination = tmp_path / "data.zip"
    archive.create_archive(tmp_path, root_name, destination, 1)
    with pytest.raises(RuntimeError, match="payload differs"):
        archive.verify_archive(destination, root_name, 2)


def test_zip_compression_level_rejected(tmp_path):
    root_name, _ = make_staging(tmp_path)
    with pytest.raises(ValueError, match="compression level"):
        archive.create_archive(tmp_path, root_name, tmp_path / "data.zip", 10)

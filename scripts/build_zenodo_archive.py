from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPRODUCIBILITY_MANIFEST = ROOT / "configs" / "reproducibility_manifest.json"

SUPPLEMENTARY_RESULT_ROOTS = (
    "results/development/guacamol_rnn_reactive_epoch_sensitivity_seed_11",
    "results/development/guacamol_transformer_reactive_epoch_sensitivity_seed_11",
    "results/development/reinvent_reactive_epoch_sensitivity_seed_11",
    "results/development/semlaflow_parameter_selection",
    "results/paper_statistics",
    "results/publication",
    "results/cleanup_manifests",
)

EXCLUDED_SUFFIXES = {
    ".ckpt",
    ".joblib",
    ".log",
    ".model",
    ".pkl",
    ".prior",
    ".pth",
    ".safetensors",
}
EXCLUDED_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    "tb_bad",
    "tb_positive",
    "tb_random",
}
EXCLUDED_PREFIXES = (
    "results/external/semlaflow/four_liability_joint_replicates/analysis/smiles_samples/",
)
SEED_PATTERN = re.compile(r"^seed_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the project-owned Zenodo data archive.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "zenodo_release",
        help="Directory receiving the archive and its checksum.",
    )
    parser.add_argument(
        "--archive-name",
        default="negative_extrapolation_molgen_data_v1.tar.zst",
        help="Compressed archive filename.",
    )
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def result_roots(manifest: dict) -> list[str]:
    roots = {group["root"] for group in manifest["seed_result_groups"]}
    roots.update(SUPPLEMENTARY_RESULT_ROOTS)
    return sorted(roots)


def is_allowed_base_checkpoint(relative: Path, confirmatory_seeds: set[int]) -> bool:
    if relative.name != "base.pt" or len(relative.parts) != 4:
        return False
    if relative.parts[:2] not in {
        ("results", "guacamol_rnn_base"),
        ("results", "guacamol_transformer_base"),
    }:
        return False
    match = SEED_PATTERN.match(relative.parts[2])
    return bool(match and int(match.group(1)) in confirmatory_seeds)


def include_file(
    relative: Path,
    confirmatory_seeds: set[int],
    development_root: bool,
) -> bool:
    relative_text = relative.as_posix()
    if any(relative_text.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.name.startswith("events.out.tfevents"):
        return False

    if not development_root:
        for part in relative.parts:
            match = SEED_PATTERN.match(part)
            if match and int(match.group(1)) not in confirmatory_seeds:
                return False

    suffix = relative.suffix.lower()
    if suffix == ".pt":
        return is_allowed_base_checkpoint(relative, confirmatory_seeds)
    return suffix not in EXCLUDED_SUFFIXES


def collect_files(manifest: dict) -> list[Path]:
    confirmatory_seeds = set(manifest["confirmatory_seeds"])
    selected: set[Path] = set()
    for root_text in result_roots(manifest):
        root = ROOT / root_text
        if not root.is_dir():
            raise FileNotFoundError(f"Required archive root is missing: {root_text}")
        development_root = root_text.startswith("results/development/")
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(ROOT)
            if include_file(relative, confirmatory_seeds, development_root):
                selected.add(relative)
    return sorted(selected, key=lambda path: path.as_posix())


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def write_release_metadata(
    staging_root: Path,
    selected: list[Path],
    checksums: dict[Path, str],
    manifest: dict,
    commit: str,
) -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    total_size = sum((ROOT / relative).stat().st_size for relative in selected)

    readme = f"""# Negative Extrapolation for Molecular Generation: Data Archive

This archive contains the project-owned generated samples, molecular scores,
selection sets, analysis outputs, and reusable GuacaMol base checkpoints used
for the associated manuscript.

- Code repository: https://github.com/JoshuaHesse/negative_extrapolation_molgen
- Code commit: `{commit}`
- Data DOI: https://doi.org/{manifest['project_data_doi']}
- Confirmatory seeds: {', '.join(map(str, manifest['confirmatory_seeds']))}
- Generated: {generated_at}
- Payload files: {len(selected)}
- Uncompressed payload size: {format_bytes(total_size)}

Extract this archive at the root of a checkout of the recorded code commit. The
stored `results/` paths then match the analysis and figure-generation commands
documented in `docs/reproduction.md`.

Validate the extracted payload from this directory with:

```bash
sha256sum -c SHA256SUMS
```

The archive deliberately excludes third-party repositories, training datasets,
and pretrained weights. Their exact source revisions and input checksums are
recorded in `reproducibility_manifest.json`. It also excludes transient tuned
and extrapolated checkpoints, logs, TensorBoard events, exploratory seed 11 from
confirmatory experiments, and experiments not reported in the manuscript.
Development seed 11 is retained only for the explicitly identified parameter-
selection and epoch-sensitivity studies.
"""
    (staging_root / "README.md").write_text(readme)
    shutil.copy2(REPRODUCIBILITY_MANIFEST, staging_root / "reproducibility_manifest.json")

    with (staging_root / "MANIFEST.tsv").open("w") as handle:
        handle.write("sha256\tsize_bytes\tpath\n")
        for relative in selected:
            handle.write(
                f"{checksums[relative]}\t{(ROOT / relative).stat().st_size}\t{relative.as_posix()}\n"
            )

    with (staging_root / "SHA256SUMS").open("w") as handle:
        for relative in selected:
            handle.write(f"{checksums[relative]}  {relative.as_posix()}\n")

    metadata = {
        "archive_version": 1,
        "code_repository": "https://github.com/JoshuaHesse/negative_extrapolation_molgen",
        "code_commit": commit,
        "project_data_doi": manifest["project_data_doi"],
        "confirmatory_seeds": manifest["confirmatory_seeds"],
        "generated_at_utc": generated_at,
        "payload_file_count": len(selected),
        "payload_size_bytes": total_size,
        "third_party_assets_included": False,
    }
    (staging_root / "archive_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def stage_files(staging_root: Path, selected: list[Path]) -> None:
    for index, relative in enumerate(selected, start=1):
        source = ROOT / relative
        destination = staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        if index % 500 == 0:
            print(f"staged {index}/{len(selected)} files", flush=True)


def create_archive(staging_parent: Path, root_name: str, archive_path: Path, level: int) -> None:
    tar_command = [
        "tar",
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "-C",
        str(staging_parent),
        "-cf",
        "-",
        root_name,
    ]
    zstd_command = ["zstd", "-T0", f"-{level}", "-f", "-o", str(archive_path)]
    with subprocess.Popen(tar_command, stdout=subprocess.PIPE) as tar_process:
        if tar_process.stdout is None:
            raise RuntimeError("tar stdout pipe was not created")
        zstd_result = subprocess.run(zstd_command, stdin=tar_process.stdout, check=False)
        tar_process.stdout.close()
        tar_status = tar_process.wait()
    if tar_status != 0:
        raise subprocess.CalledProcessError(tar_status, tar_command)
    if zstd_result.returncode != 0:
        raise subprocess.CalledProcessError(zstd_result.returncode, zstd_command)


def verify_archive(archive_path: Path, root_name: str, expected_file_count: int) -> None:
    listing = subprocess.check_output(
        ["tar", "--zstd", "-tf", str(archive_path)], text=True
    ).splitlines()
    archived_files = [name for name in listing if name and not name.endswith("/")]
    expected = expected_file_count + 5
    if len(archived_files) != expected:
        raise RuntimeError(
            f"Archive verification failed: found {len(archived_files)} files, expected {expected}"
        )
    required = {
        f"{root_name}/README.md",
        f"{root_name}/MANIFEST.tsv",
        f"{root_name}/SHA256SUMS",
        f"{root_name}/archive_metadata.json",
        f"{root_name}/reproducibility_manifest.json",
    }
    missing = sorted(required.difference(archived_files))
    if missing:
        raise RuntimeError(f"Archive verification failed; missing metadata: {missing}")


def main() -> None:
    args = parse_args()
    manifest = json.loads(REPRODUCIBILITY_MANIFEST.read_text())
    selected = collect_files(manifest)
    total_size = sum((ROOT / relative).stat().st_size for relative in selected)
    print(f"Selected {len(selected)} files ({format_bytes(total_size)})")
    if args.dry_run:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root_name = args.archive_name.removesuffix(".tar.zst")
    staging_parent = output_dir / ".staging"
    staging_root = staging_parent / root_name
    if staging_parent.exists():
        shutil.rmtree(staging_parent)
    staging_root.mkdir(parents=True)

    print("Computing payload checksums...", flush=True)
    checksums: dict[Path, str] = {}
    for index, relative in enumerate(selected, start=1):
        checksums[relative] = sha256(ROOT / relative)
        if index % 250 == 0:
            print(f"checksummed {index}/{len(selected)} files", flush=True)

    write_release_metadata(staging_root, selected, checksums, manifest, git_commit())
    stage_files(staging_root, selected)

    archive_path = output_dir / args.archive_name
    print(f"Compressing {archive_path}...", flush=True)
    create_archive(staging_parent, root_name, archive_path, args.compression_level)
    print("Verifying archive structure...", flush=True)
    verify_archive(archive_path, root_name, len(selected))

    archive_checksum = sha256(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{archive_checksum}  {archive_path.name}\n")
    if not args.keep_staging:
        shutil.rmtree(staging_parent)

    print(f"Archive: {archive_path}")
    print(f"Archive size: {format_bytes(archive_path.stat().st_size)}")
    print(f"SHA-256: {archive_checksum}")
    print(f"Checksum file: {checksum_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "reproducibility_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit publication reproducibility inputs.")
    parser.add_argument("--inputs", action="store_true", help="Verify external repositories and input hashes.")
    parser.add_argument("--results", action="store_true", help="Verify confirmatory result completion artifacts.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(repository), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def static_audit(errors: list[str]) -> None:
    required = [
        "README.md",
        "LICENSE",
        "Makefile",
        "Dockerfile",
        "docs/reproduction.md",
        "notebooks/260612_main_paper_figures.ipynb",
        "notebooks/260617_si_figures_and_statistics.ipynb",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing public file: {relative}")

    for path in sorted((ROOT / "configs").rglob("*.json")):
        try:
            json.loads(path.read_text())
        except Exception as error:
            errors.append(f"invalid JSON {path.relative_to(ROOT)}: {error}")

    for folder in ("neon_molgen", "scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            try:
                ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as error:
                errors.append(f"invalid Python {path.relative_to(ROOT)}: {error}")

    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook.get("cells", [])):
            if not cell.get("id"):
                errors.append(f"notebook cell ID missing: {path.name} cell {index}")
            if cell.get("cell_type") != "code":
                if "outputs" in cell or "execution_count" in cell:
                    errors.append(f"code-only field on non-code cell: {path.name} cell {index}")
                continue
            if cell.get("outputs"):
                errors.append(f"notebook output retained: {path.name} cell {index}")
            if cell.get("execution_count") is not None:
                errors.append(f"notebook execution count retained: {path.name} cell {index}")

    make_text = (ROOT / "Makefile").read_text()
    references = set(
        re.findall(
            r"(?:scripts/[A-Za-z0-9_./-]+\.py|configs/[A-Za-z0-9_./-]+\.(?:json|toml|csv|smi))",
            make_text,
        )
    )
    for relative in sorted(references):
        if not (ROOT / relative).is_file():
            errors.append(f"Makefile references missing file: {relative}")

    targets = set(re.findall(r"^([A-Za-z0-9_.%-]+):", make_text, flags=re.MULTILINE))
    docs = (ROOT / "README.md").read_text() + "\n" + "\n".join(
        path.read_text() for path in sorted((ROOT / "docs").glob("*.md"))
    )
    for target in sorted(set(re.findall(r"\bmake ([A-Za-z0-9_.-]+)", docs))):
        if target not in targets:
            errors.append(f"documentation references missing Make target: {target}")


def input_audit(manifest: dict, errors: list[str]) -> None:
    for item in manifest["repositories"]:
        path = ROOT / item["path"]
        if not path.is_dir():
            errors.append(f"missing repository: {item['name']} ({item['path']})")
            continue
        try:
            commit = git_value(path, "rev-parse", "HEAD")
        except (OSError, subprocess.CalledProcessError):
            errors.append(f"not a Git repository: {item['path']}")
            continue
        if commit != item["commit"]:
            errors.append(f"wrong {item['name']} commit: {commit} != {item['commit']}")

    for item in manifest["input_files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            errors.append(f"missing input: {item['name']} ({item['path']})")
            continue
        observed = sha256(path)
        if observed != item["sha256"]:
            errors.append(f"checksum mismatch: {item['path']} ({observed})")


def result_audit(manifest: dict, errors: list[str]) -> None:
    seeds = manifest["confirmatory_seeds"]
    for group in manifest["seed_result_groups"]:
        for seed in seeds:
            relative = Path(group["root"]) / group["pattern"].format(seed=seed)
            if not (ROOT / relative).is_file():
                errors.append(f"missing result: {group['name']} seed {seed} ({relative})")
    for relative in manifest["analysis_files"]:
        if not (ROOT / relative).is_file():
            errors.append(f"missing analysis artifact: {relative}")


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text())
    errors: list[str] = []
    static_audit(errors)
    if args.inputs:
        input_audit(manifest, errors)
    if args.results:
        result_audit(manifest, errors)

    scopes = ["source"]
    if args.inputs:
        scopes.append("inputs")
    if args.results:
        scopes.append("results")
    if errors:
        print(f"Reproducibility audit failed ({', '.join(scopes)}):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Reproducibility audit passed: {', '.join(scopes)}")


if __name__ == "__main__":
    main()

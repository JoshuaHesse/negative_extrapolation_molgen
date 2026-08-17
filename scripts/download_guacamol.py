from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}/versions/{version}"

DATASETS = {
    "train": {
        "article_id": 7322228,
        "version": 2,
        "filename": "guacamol_v1_train.smiles",
        "md5": "05ad85d871958a05c02ab51a4fde8530",
        "fallback_urls": [
            "https://huggingface.co/datasets/katielink/GuacaMol/resolve/main/guacamol_v1_train.smiles",
            "https://raw.githubusercontent.com/iktos/generation-under-synthetic-constraint/master/data/guacamol_v1_train.smiles",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(DATASETS),
        default=["train"],
        help="GuacaMol split files to fetch.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def figshare_download_url(dataset: dict, *, timeout: int) -> str:
    url = FIGSHARE_API.format(
        article_id=dataset["article_id"],
        version=dataset["version"],
    )
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    files = payload.get("files", [])
    if not files:
        raise RuntimeError(f"No files found in Figshare metadata for article {dataset['article_id']}")
    for item in files:
        if item.get("name") == dataset["filename"]:
            return item["download_url"]
    return files[0]["download_url"]


def download_url(url: str, destination: Path, *, timeout: int) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def candidate_urls(dataset: dict, *, timeout: int) -> list[str]:
    urls = []
    try:
        urls.append(figshare_download_url(dataset, timeout=timeout))
    except Exception as error:
        print(f"Could not resolve Figshare URL for {dataset['filename']}: {error}")
    urls.extend(dataset.get("fallback_urls", []))
    return urls


def fetch_split(split: str, output_dir: Path, *, force: bool, timeout: int) -> Path:
    dataset = DATASETS[split]
    output_path = output_dir / dataset["filename"]
    expected_md5 = dataset["md5"]
    if output_path.exists() and not force:
        observed_md5 = md5sum(output_path)
        if observed_md5 == expected_md5:
            print(f"{output_path} already exists and md5 matches.")
            return output_path
        raise ValueError(
            f"{output_path} exists but md5 is {observed_md5}, expected {expected_md5}. "
            "Use --force to replace it."
        )

    urls = candidate_urls(dataset, timeout=timeout)
    if not urls:
        raise RuntimeError(f"No download URLs available for split '{split}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        print(f"Downloading {split} from {url}")
        with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            download_url(url, tmp_path, timeout=timeout)
            observed_md5 = md5sum(tmp_path)
            if observed_md5 != expected_md5:
                raise ValueError(f"md5 {observed_md5} != expected {expected_md5}")
            tmp_path.replace(output_path)
            print(f"Wrote {output_path}")
            return output_path
        except Exception as error:
            errors.append(f"{url}: {error}")
            tmp_path.unlink(missing_ok=True)
            print(f"Failed: {error}")
    raise RuntimeError("All GuacaMol download attempts failed:\n" + "\n".join(errors))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for split in args.splits:
        fetch_split(split, output_dir, force=args.force, timeout=args.timeout)


if __name__ == "__main__":
    main()

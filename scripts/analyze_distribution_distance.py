from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer
from scipy import linalg
from sklearn.cluster import MiniBatchKMeans

SAMPLE_SUFFIX = "_samples.csv"
REFERENCE_FILES = {
    "base": "base_samples.csv",
    "good": "good_smiles.smi",
    "bad": "bad_smiles.smi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/objectives/guacamol_rnn_liability_removal")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--min-size", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cluster-fit-size", type=int, default=5000)
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--skip-fcd", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed seed/model/reference comparisons from the partial CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model names to analyze. Defaults to every *_samples.csv file.",
    )
    parser.add_argument(
        "--reference-smiles",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional reference SMILES file. Can be passed multiple times.",
    )
    return parser.parse_args()


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode()).hexdigest()
    return int(digest[:16], 16) % (2**32)


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def unique_valid_smiles(smiles: list[str]) -> list[str]:
    canonical = []
    seen = set()
    for smi in smiles:
        can = canonicalize_smiles(str(smi).strip())
        if can and can not in seen:
            seen.add(can)
            canonical.append(can)
    return canonical


def read_sample_smiles(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "valid" in frame.columns:
        frame = frame[frame["valid"]]
    column = "canonical_smiles" if "canonical_smiles" in frame.columns else "smiles"
    return unique_valid_smiles(frame[column].dropna().astype(str).tolist())


def read_smi(path: Path) -> list[str]:
    smiles = [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]
    return unique_valid_smiles(smiles)


def sample_smiles(smiles: list[str], n: int, *, key: str) -> list[str]:
    if len(smiles) <= n:
        return list(smiles)
    rng = np.random.default_rng(stable_seed(key))
    indices = rng.choice(len(smiles), size=n, replace=False)
    return [smiles[int(index)] for index in indices]


def fdd_descriptor_vector(smiles: str) -> list[float] | None:
    """Five-descriptor FDD vector from Özçelik and Grisoni, J. Cheminf. 2025."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        float(Crippen.MolLogP(mol)),
        float(Descriptors.MolWt(mol)),
        float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
    ]


def physchem_descriptor_vector(smiles: str) -> list[float] | None:
    """Extended interpretable physicochemical vector retained for sensitivity analysis."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumHBA(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(Chem.GetFormalCharge(mol)),
        float(sascorer.calculateScore(mol)),
    ]


def descriptor_matrix(smiles: list[str], vector_fn) -> np.ndarray:
    rows = [row for smi in smiles if (row := vector_fn(smi)) is not None]
    return np.asarray(rows, dtype=float)


FDD_DESCRIPTOR_MINS = np.array([-3.0, 0.0, 0.0, 0.0, 0.0])
FDD_DESCRIPTOR_MAXES = np.array([10.0, 1000.0, 10.0, 10.0, 250.0])
FDD_DESCRIPTOR_RANGES = FDD_DESCRIPTOR_MAXES - FDD_DESCRIPTOR_MINS


def scale_fdd_descriptors(x: np.ndarray) -> np.ndarray:
    return (x - FDD_DESCRIPTOR_MINS) / FDD_DESCRIPTOR_RANGES


def frechet_distance(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-6) -> float:
    """FCD-library-style Fréchet distance used by Özçelik and Grisoni's FDD script."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mu_x = np.atleast_1d(np.mean(x, axis=0))
    mu_y = np.atleast_1d(np.mean(y, axis=0))
    cov_x = np.atleast_2d(np.cov(x.T))
    cov_y = np.atleast_2d(np.cov(y.T))
    diff = mu_x - mu_y
    covmean = linalg.sqrtm(cov_x.dot(cov_y))
    is_real = np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)
    if not np.isfinite(covmean).all() or not is_real:
        offset = np.eye(cov_x.shape[0]) * eps
        covmean = linalg.sqrtm((cov_x + offset).dot(cov_y + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            return float("nan")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov_x) + np.trace(cov_y) - 2.0 * np.trace(covmean))


def fdd(smiles_a: list[str], smiles_b: list[str]) -> float:
    """Fréchet Descriptor Distance matching Özçelik and Grisoni's released implementation."""
    x = scale_fdd_descriptors(descriptor_matrix(smiles_a, fdd_descriptor_vector))
    y = scale_fdd_descriptors(descriptor_matrix(smiles_b, fdd_descriptor_vector))
    return frechet_distance(x, y)


def physchem_frechet_10d(smiles_a: list[str], smiles_b: list[str]) -> float:
    return frechet_distance(
        descriptor_matrix(smiles_a, physchem_descriptor_vector),
        descriptor_matrix(smiles_b, physchem_descriptor_vector),
    )


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint_smiles(smiles: list[str]):
    fingerprints = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fingerprints.append(MORGAN_GENERATOR.GetFingerprint(mol))
    return fingerprints


def nearest_tanimoto_stats(smiles_a: list[str], smiles_b: list[str]) -> dict[str, float]:
    """Nearest-neighbor Jaccard/Tanimoto distance to a reference set.

    For binary Morgan fingerprints, Jaccard similarity and Tanimoto similarity are
    equivalent. The distance columns below are 1 - nearest-neighbor Tanimoto, so
    lower means closer to the reference under a fixed sampling budget.
    """
    fps_a = fingerprint_smiles(smiles_a)
    fps_b = fingerprint_smiles(smiles_b)
    if not fps_a or not fps_b:
        return {
            "nn_tanimoto_mean": float("nan"),
            "nn_tanimoto_median": float("nan"),
            "nn_tanimoto_p90": float("nan"),
            "nn_tanimoto_ge_0_5_fraction": float("nan"),
            "nn_tanimoto_ge_0_7_fraction": float("nan"),
            "nn_jaccard_distance_mean": float("nan"),
            "nn_jaccard_distance_median": float("nan"),
        }
    nearest = np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fp, fps_b)) for fp in fps_a],
        dtype=float,
    )
    distances = 1.0 - nearest
    return {
        "nn_tanimoto_mean": float(np.mean(nearest)),
        "nn_tanimoto_median": float(np.median(nearest)),
        "nn_tanimoto_p90": float(np.quantile(nearest, 0.90)),
        "nn_tanimoto_ge_0_5_fraction": float(np.mean(nearest >= 0.5)),
        "nn_tanimoto_ge_0_7_fraction": float(np.mean(nearest >= 0.7)),
        "nn_jaccard_distance_mean": float(np.mean(distances)),
        "nn_jaccard_distance_median": float(np.median(distances)),
    }


def fingerprint_array(smiles: list[str]) -> np.ndarray:
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = MORGAN_GENERATOR.GetFingerprint(mol)
        row = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, row)
        rows.append(row)
    return np.asarray(rows, dtype=np.uint8)


def distribution_from_labels(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_clusters).astype(float)
    if counts.sum() == 0:
        return np.full(n_clusters, 1.0 / n_clusters)
    return counts / counts.sum()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def fit_base_cluster_model(
    base_smiles: list[str],
    *,
    cluster_fit_size: int,
    n_clusters: int,
    seed: int,
) -> tuple[MiniBatchKMeans | None, np.ndarray | None]:
    base_array = fingerprint_array(base_smiles)
    if len(base_array) < max(2, n_clusters):
        return None, None
    rng = np.random.default_rng(seed)
    fit_size = min(cluster_fit_size, len(base_array))
    fit_idx = rng.choice(len(base_array), size=fit_size, replace=False) if fit_size < len(base_array) else np.arange(len(base_array))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=1024,
        n_init="auto",
    )
    kmeans.fit(base_array[fit_idx])
    base_distribution = distribution_from_labels(kmeans.predict(base_array), n_clusters)
    return kmeans, base_distribution


def cluster_js_to_reference(
    smiles_a: list[str],
    smiles_b: list[str],
    *,
    kmeans: MiniBatchKMeans | None,
    n_clusters: int,
) -> float:
    if kmeans is None:
        return float("nan")
    array_a = fingerprint_array(smiles_a)
    array_b = fingerprint_array(smiles_b)
    if len(array_a) == 0 or len(array_b) == 0:
        return float("nan")
    dist_a = distribution_from_labels(kmeans.predict(array_a), n_clusters)
    dist_b = distribution_from_labels(kmeans.predict(array_b), n_clusters)
    return js_divergence(dist_a, dist_b)


def build_fcd(device: str, jobs: int):
    try:
        from fcd_torch import FCD
    except ImportError:
        return None, "fcd_torch is not installed"
    try:
        return FCD(device=device, n_jobs=jobs), None
    except TypeError:
        try:
            return FCD(device=device), None
        except Exception as error:  # pragma: no cover - depends on optional package internals
            return None, str(error)
    except Exception as error:  # pragma: no cover - depends on optional package internals
        return None, str(error)


def fcd_distance(
    fcd_model,
    smiles_a: list[str],
    smiles_b: list[str],
    cache: dict[str, object],
    *,
    key_a: str,
    key_b: str,
) -> float:
    if fcd_model is None:
        return float("nan")
    try:
        if key_a not in cache:
            cache[key_a] = fcd_model.precalc(smiles_a)
        if key_b not in cache:
            cache[key_b] = fcd_model.precalc(smiles_b)
        return float(fcd_model.metric(cache[key_b], cache[key_a]))
    except Exception as error:
        print(
            f"FCD failed for {key_a!r} versus {key_b!r}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return float("nan")


def discover_seed_dirs(results_dir: Path, seeds: set[int] | None = None) -> list[Path]:
    seed_dirs = sorted(
        path
        for path in results_dir.glob("seed_*")
        if path.is_dir()
        and (seeds is None or int(path.name.removeprefix("seed_")) in seeds)
    )
    if seed_dirs:
        return seed_dirs
    if list(results_dir.glob(f"*{SAMPLE_SUFFIX}")):
        return [results_dir]
    raise ValueError(f"No seed directories or sample files found in {results_dir}")


def seed_from_dir(seed_dir: Path) -> int:
    if seed_dir.name.startswith("seed_"):
        return int(seed_dir.name.removeprefix("seed_"))
    return 0


def model_name_from_sample(path: Path) -> str:
    return path.name[: -len(SAMPLE_SUFFIX)]


def parse_extra_references(values: list[str]) -> dict[str, Path]:
    references = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH for --reference-smiles, got {value!r}")
        name, path = value.split("=", 1)
        references[name] = Path(path)
    return references


def reference_smiles(seed_dir: Path, extra_references: dict[str, Path]) -> dict[str, list[str]]:
    refs = {}
    for name, filename in REFERENCE_FILES.items():
        path = seed_dir / filename
        if not path.exists():
            continue
        refs[name] = read_sample_smiles(path) if filename.endswith(".csv") else read_smi(path)
    for name, path in extra_references.items():
        resolved = path if path.is_absolute() else seed_dir / path
        refs[name] = read_smi(resolved)
    return refs


def sample_files(seed_dir: Path, requested_models: set[str] | None) -> list[Path]:
    paths = sorted(seed_dir.glob(f"*{SAMPLE_SUFFIX}"))
    if requested_models is None:
        return paths
    return [path for path in paths if model_name_from_sample(path) in requested_models]


def summarize(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        metrics.to_csv(output_dir / "distribution_distance_metrics.csv", index=False)
        return
    metrics.to_csv(output_dir / "distribution_distance_metrics.csv", index=False)
    numeric = [
        column
        for column in metrics.select_dtypes(include="number").columns
        if column not in {"seed", "n_model_available", "n_reference_available", "n_compared"}
    ]
    aggregate = metrics.groupby(["model", "reference"])[numeric].agg(["mean", "std", "count"])
    aggregate.columns = ["_".join(str(part) for part in column if part) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output_dir / "distribution_distance_summary.csv", index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(metrics: pd.DataFrame, output_dir: Path, fcd_available: bool, fcd_error: str | None) -> None:
    summary = pd.read_csv(output_dir / "distribution_distance_summary.csv")
    models = [
        "base",
        "positive",
        "neon_lambda_0.5",
        "neon_lambda_0.75",
        "neon_lambda_1.0",
        "random_neon_lambda_1.0",
    ]
    references = ["base", "good", "bad"]
    display = summary[summary["model"].isin(models) & summary["reference"].isin(references)].copy()
    display["model_sort"] = display["model"].map({model: i for i, model in enumerate(models)})
    display["reference_sort"] = display["reference"].map({name: i for i, name in enumerate(references)})
    display = display.sort_values(["reference_sort", "model_sort"])
    keep = [
        "reference",
        "model",
        "fdd_mean",
        "fdd_std",
        "fcd_mean",
        "fcd_std",
        "nn_jaccard_distance_mean_mean",
        "nn_tanimoto_mean_mean",
        "cluster_js_divergence_mean",
        "fdd_count",
    ]
    display = display[[column for column in keep if column in display.columns]]
    lines = [
        "# Distribution Distance Analysis",
        "",
        f"FCD available: `{fcd_available}`",
    ]
    if fcd_error:
        lines.extend(["", f"FCD note: `{fcd_error}`"])
    lines.extend(
        [
            "",
            "Distances are computed on equal-size valid unique molecule samples for each",
            "model/reference comparison. Lower values mean the generated set is closer to",
            "the reference distribution.",
            "",
            markdown_table(display),
            "",
            "Interpretation: compare NEON to positive fine-tuning against `good`, `bad`,",
            "and `base`. A targeted-erasure result should move away from `bad` while",
            "remaining closer to `base` than positive fine-tuning, or preserving more",
            "diversity at comparable objective gain.",
            "",
        ]
    )
    (output_dir / "distribution_distance_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir or results_dir / "analysis" / "distribution_distance")
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_models = set(args.models) if args.models else None
    extra_references = parse_extra_references(args.reference_smiles)

    fcd_model = None
    fcd_error = None
    if not args.skip_fcd:
        fcd_model, fcd_error = build_fcd(args.device, args.jobs)

    partial_path = output_dir / "distribution_distance_metrics.partial.csv"
    rows: list[dict[str, object]] = []
    completed_keys: set[tuple[int, str, str]] = set()
    if args.resume and partial_path.is_file():
        partial = pd.read_csv(partial_path)
        required = {"seed", "model", "reference"}
        if not required.issubset(partial.columns):
            raise ValueError(
                f"Cannot resume from {partial_path}: missing columns "
                f"{sorted(required - set(partial.columns))}"
            )
        rows = partial.to_dict("records")
        completed_keys = {
            (int(row["seed"]), str(row["model"]), str(row["reference"]))
            for row in rows
        }
        print(
            f"Resuming from {len(completed_keys)} completed comparisons in {partial_path}",
            flush=True,
        )
    fcd_cache: dict[str, object] = {}
    comparisons_done = len(completed_keys)
    for seed_dir in discover_seed_dirs(
        results_dir,
        set(args.seeds) if args.seeds is not None else None,
    ):
        seed = seed_from_dir(seed_dir)
        refs = reference_smiles(seed_dir, extra_references)
        if not refs:
            continue
        kmeans, _base_cluster_distribution = fit_base_cluster_model(
            refs["base"],
            cluster_fit_size=args.cluster_fit_size,
            n_clusters=args.n_clusters,
            seed=args.seed + seed,
        ) if "base" in refs else (None, None)
        for sample_path in sample_files(seed_dir, requested_models):
            model = model_name_from_sample(sample_path)
            model_smiles = read_sample_smiles(sample_path)
            for ref_name, ref_smiles in refs.items():
                comparison_key = (seed, model, ref_name)
                if comparison_key in completed_keys:
                    continue
                n_compared = min(args.sample_size, len(model_smiles), len(ref_smiles))
                row = {
                    "seed": seed,
                    "model": model,
                    "reference": ref_name,
                    "n_model_available": len(model_smiles),
                    "n_reference_available": len(ref_smiles),
                    "n_compared": n_compared,
                    "fdd": np.nan,
                    "physchem_frechet_10d": np.nan,
                    "fcd": np.nan,
                    "nn_tanimoto_mean": np.nan,
                    "nn_tanimoto_median": np.nan,
                    "nn_tanimoto_p90": np.nan,
                    "nn_tanimoto_ge_0_5_fraction": np.nan,
                    "nn_tanimoto_ge_0_7_fraction": np.nan,
                    "nn_jaccard_distance_mean": np.nan,
                    "nn_jaccard_distance_median": np.nan,
                    "cluster_js_divergence": np.nan,
                }
                if n_compared >= args.min_size:
                    left_key = f"{seed}:{model}:model:{ref_name}:{n_compared}"
                    right_key = f"{seed}:{model}:reference:{ref_name}:{n_compared}"
                    sampled_model = sample_smiles(model_smiles, n_compared, key=left_key)
                    sampled_ref = (
                        sampled_model
                        if model == ref_name == "base"
                        else sample_smiles(ref_smiles, n_compared, key=right_key)
                    )
                    row["fdd"] = fdd(sampled_model, sampled_ref)
                    row["physchem_frechet_10d"] = physchem_frechet_10d(sampled_model, sampled_ref)
                    row.update(nearest_tanimoto_stats(sampled_model, sampled_ref))
                    row["cluster_js_divergence"] = cluster_js_to_reference(
                        sampled_model,
                        sampled_ref,
                        kmeans=kmeans,
                        n_clusters=args.n_clusters,
                    )
                    row["fcd"] = fcd_distance(
                        fcd_model,
                        sampled_model,
                        sampled_ref,
                        fcd_cache,
                        key_a=left_key,
                        key_b=left_key if model == ref_name == "base" else right_key,
                    )
                rows.append(row)
                comparisons_done += 1
                if args.flush_every > 0 and comparisons_done % args.flush_every == 0:
                    partial = pd.DataFrame(rows)
                    partial.to_csv(partial_path, index=False)
                    print(
                        "completed "
                        f"{comparisons_done} comparisons; latest seed={seed} "
                        f"model={model} reference={ref_name} n={n_compared}",
                        flush=True,
                    )

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics = (
            metrics.drop_duplicates(["seed", "model", "reference"], keep="last")
            .sort_values(["seed", "model", "reference"])
            .reset_index(drop=True)
        )
        metrics.to_csv(partial_path, index=False)
    summarize(metrics, output_dir)
    write_report(metrics, output_dir, fcd_model is not None, fcd_error)
    print(f"Wrote distribution distance analysis to {output_dir}")


if __name__ == "__main__":
    main()

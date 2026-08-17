from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import MiniBatchKMeans

SAMPLE_SUFFIX = "_samples.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/guacamol_rnn_qed")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--internal-sample-size", type=int, default=2000)
    parser.add_argument("--internal-random-pairs", type=int, default=200000)
    parser.add_argument("--cluster-fit-size", type=int, default=5000)
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--sphere-similarity-threshold", type=float, default=0.65)
    parser.add_argument(
        "--sphere-sample-size",
        type=int,
        default=0,
        help="Optional cap for sphere-exclusion molecules per model. 0 means use all valid molecules.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser.parse_args()


def model_name_from_sample(path: Path) -> str:
    return path.name[: -len(SAMPLE_SUFFIX)]


def read_valid_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    valid = frame[frame["valid"]].copy()
    valid = valid[valid["canonical_smiles"].notna()]
    valid = valid[valid["canonical_smiles"].astype(str).ne("")]
    return valid.reset_index(drop=True)


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if len(counts) == 0:
        return float("nan")
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def scaffold_metrics(mols: list[Chem.Mol]) -> dict[str, float]:
    scaffolds = [scaffold_smiles(mol) for mol in mols]
    scaffolds = [scaffold for scaffold in scaffolds if scaffold]
    if not scaffolds:
        return {
            "unique_scaffold_fraction": np.nan,
            "scaffold_entropy": np.nan,
            "scaffold_entropy_normalized": np.nan,
            "effective_scaffolds": np.nan,
            "top1_scaffold_fraction": np.nan,
            "top10_scaffold_fraction": np.nan,
            "top50_scaffold_fraction": np.nan,
            "scaffold_simpson_diversity": np.nan,
        }
    counts = pd.Series(scaffolds).value_counts().to_numpy()
    entropy = entropy_from_counts(counts)
    probabilities = counts / counts.sum()
    return {
        "unique_scaffold_fraction": float(len(counts) / len(mols)) if mols else np.nan,
        "scaffold_entropy": entropy,
        "scaffold_entropy_normalized": float(entropy / math.log(len(counts))) if len(counts) > 1 else 0.0,
        "effective_scaffolds": float(math.exp(entropy)),
        "top1_scaffold_fraction": float(counts[:1].sum() / counts.sum()),
        "top10_scaffold_fraction": float(counts[:10].sum() / counts.sum()),
        "top50_scaffold_fraction": float(counts[:50].sum() / counts.sum()),
        "scaffold_simpson_diversity": float(1.0 - np.square(probabilities).sum()),
    }


def fingerprint_array(
    mols: list[Chem.Mol],
    generator,
) -> tuple[list[DataStructs.ExplicitBitVect], np.ndarray]:
    fps = [generator.GetFingerprint(mol) for mol in mols]
    array = np.zeros((len(fps), 2048), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, array[i])
    return fps, array


def unique_morgan_feature_count(mols: list[Chem.Mol], sparse_generator) -> int:
    features = set()
    for mol in mols:
        sparse = sparse_generator.GetSparseCountFingerprint(mol)
        features.update(sparse.GetNonzeroElements())
    return len(features)


def sphere_exclusion_count(
    fps: list[DataStructs.ExplicitBitVect],
    *,
    similarity_threshold: float,
) -> int:
    centers: list[DataStructs.ExplicitBitVect] = []
    for fp in fps:
        if not centers:
            centers.append(fp)
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fp, centers)
        if max(similarities) < similarity_threshold:
            centers.append(fp)
    return len(centers)


def sample_indices(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 0 or n <= size:
        return np.arange(n)
    return rng.choice(n, size=size, replace=False)


def internal_similarity_metrics(
    fps: list[DataStructs.ExplicitBitVect],
    *,
    sample_size: int,
    random_pairs: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = len(fps)
    if n < 2:
        return {
            "pairwise_tanimoto_mean": np.nan,
            "pairwise_tanimoto_distance_mean": np.nan,
            "internal_nn_tanimoto_mean": np.nan,
            "internal_nn_tanimoto_median": np.nan,
            "close_neighbor_gt_0.7_fraction": np.nan,
            "close_neighbor_gt_0.8_fraction": np.nan,
        }

    idx = sample_indices(n, sample_size, rng)
    sample_fps = [fps[int(i)] for i in idx]
    m = len(sample_fps)

    pair_count = min(random_pairs, m * (m - 1) // 2)
    similarities = []
    if pair_count:
        left = rng.integers(0, m, size=pair_count)
        right = rng.integers(0, m - 1, size=pair_count)
        right = right + (right >= left)
        for i, j in zip(left, right, strict=False):
            similarities.append(DataStructs.TanimotoSimilarity(sample_fps[int(i)], sample_fps[int(j)]))

    nearest = []
    for i, fp in enumerate(sample_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, sample_fps)
        sims[i] = -1.0
        nearest.append(max(sims))
    nearest_values = np.asarray(nearest, dtype=float)
    pair_values = np.asarray(similarities, dtype=float)
    return {
        "pairwise_tanimoto_mean": float(pair_values.mean()) if len(pair_values) else np.nan,
        "pairwise_tanimoto_distance_mean": float(1.0 - pair_values.mean())
        if len(pair_values)
        else np.nan,
        "internal_nn_tanimoto_mean": float(nearest_values.mean()),
        "internal_nn_tanimoto_median": float(np.median(nearest_values)),
        "close_neighbor_gt_0.7_fraction": float((nearest_values > 0.7).mean()),
        "close_neighbor_gt_0.8_fraction": float((nearest_values > 0.8).mean()),
    }


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


def cluster_metrics(
    array: np.ndarray,
    *,
    kmeans: MiniBatchKMeans,
    base_distribution: np.ndarray,
) -> dict[str, float]:
    labels = kmeans.predict(array)
    distribution = distribution_from_labels(labels, kmeans.n_clusters)
    entropy = entropy_from_counts(np.bincount(labels, minlength=kmeans.n_clusters))
    return {
        "occupied_cluster_fraction": float((distribution > 0).mean()),
        "cluster_entropy": entropy,
        "cluster_entropy_normalized": float(entropy / math.log(kmeans.n_clusters)),
        "effective_clusters": float(math.exp(entropy)),
        "top1_cluster_fraction": float(distribution.max()),
        "top5_cluster_fraction": float(np.sort(distribution)[-5:].sum()),
        "js_divergence_from_base_clusters": js_divergence(distribution, base_distribution),
    }


def load_model_data(
    results_dir: Path,
    requested_models: set[str] | None,
) -> dict[str, dict]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sparse_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    data = {}
    for path in sorted(results_dir.glob(f"*{SAMPLE_SUFFIX}")):
        model = model_name_from_sample(path)
        if requested_models is not None and model not in requested_models:
            continue
        frame = read_valid_frame(path)
        mols = [mol for smi in frame["canonical_smiles"] if (mol := mol_from_smiles(str(smi))) is not None]
        fps, array = fingerprint_array(mols, generator)
        data[model] = {
            "path": path,
            "frame": frame,
            "mols": mols,
            "fps": fps,
            "array": array,
            "unique_morgan_features": unique_morgan_feature_count(mols, sparse_generator),
        }
    if "base" not in data:
        raise ValueError(f"Could not find base_samples.csv in {results_dir}")
    return data


def discover_result_dirs(
    results_dir: Path,
    seeds: set[int] | None = None,
) -> list[Path]:
    seed_dirs = sorted(
        path
        for path in results_dir.glob("seed_*")
        if path.is_dir()
        and (seeds is None or int(path.name.removeprefix("seed_")) in seeds)
    )
    if seed_dirs:
        return seed_dirs
    return [results_dir]


def seed_from_dir(results_dir: Path) -> int:
    if results_dir.name.startswith("seed_"):
        return int(results_dir.name.removeprefix("seed_"))
    return 0


def write_markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze_one_dir(
    results_dir: Path,
    *,
    requested_models: set[str] | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    seed = seed_from_dir(results_dir)
    rng = np.random.default_rng(args.seed + seed)

    data = load_model_data(
        results_dir,
        requested_models,
    )
    base_array = data["base"]["array"]
    fit_idx = sample_indices(len(base_array), args.cluster_fit_size, rng)
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed + seed,
        batch_size=1024,
        n_init="auto",
    )
    kmeans.fit(base_array[fit_idx])
    base_labels = kmeans.predict(base_array)
    base_distribution = distribution_from_labels(base_labels, args.n_clusters)

    rows = []
    for model, item in sorted(data.items()):
        model_rng = np.random.default_rng(args.seed + seed + sum(ord(char) for char in model))
        sphere_fps = item["fps"]
        if args.sphere_sample_size and len(sphere_fps) > args.sphere_sample_size:
            sphere_idx = sample_indices(len(sphere_fps), args.sphere_sample_size, model_rng)
            sphere_fps = [sphere_fps[int(index)] for index in sphere_idx]
        row = {
            "seed": seed,
            "model": model,
            "n_valid": int(len(item["mols"])),
            "sphere_exclusion_n": int(len(sphere_fps)),
            "sphere_exclusion_circles": int(
                sphere_exclusion_count(
                    sphere_fps,
                    similarity_threshold=args.sphere_similarity_threshold,
                )
            ),
            "unique_morgan_features": int(item["unique_morgan_features"]),
            "unique_morgan_features_per_valid": (
                float(item["unique_morgan_features"] / len(item["mols"])) if item["mols"] else np.nan
            ),
        }
        row.update(scaffold_metrics(item["mols"]))
        row.update(
            internal_similarity_metrics(
                item["fps"],
                sample_size=args.internal_sample_size,
                random_pairs=args.internal_random_pairs,
                rng=model_rng,
            )
        )
        row.update(
            cluster_metrics(
                item["array"],
                kmeans=kmeans,
                base_distribution=base_distribution,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir or results_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_models = set(args.models) if args.models else None

    metrics = pd.concat(
        [
            analyze_one_dir(seed_dir, requested_models=requested_models, args=args)
            for seed_dir in discover_result_dirs(
                results_dir,
                set(args.seeds) if args.seeds is not None else None,
            )
        ],
        ignore_index=True,
    )
    metrics.to_csv(output_dir / "diversity_metrics.csv", index=False)

    delta_rows = []
    for seed, seed_metrics in metrics.groupby("seed"):
        base_row = seed_metrics[seed_metrics["model"].eq("base")].iloc[0]
        for _, row in seed_metrics.iterrows():
            if row["model"] == "base":
                continue
            delta = {"seed": seed, "model": row["model"]}
            for column in metrics.select_dtypes(include="number").columns:
                if column in {"seed", "n_valid", "sphere_exclusion_n"}:
                    continue
                delta[f"delta_{column}"] = row[column] - base_row[column]
            delta_rows.append(delta)
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(output_dir / "diversity_delta_metrics.csv", index=False)

    numeric_columns = [
        column for column in metrics.select_dtypes(include="number").columns if column != "seed"
    ]
    summary = metrics.groupby("model")[numeric_columns].agg(["mean", "std", "count"])
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns]
    summary.reset_index().to_csv(output_dir / "diversity_summary.csv", index=False)

    interesting = [
        "base",
        "positive",
        "neon_lambda_0.25",
        "neon_lambda_0.5",
        "neon_lambda_0.75",
        "neon_lambda_1.0",
        "neon_last_block_output_lambda_0.5",
        "neon_last_block_output_lambda_0.75",
        "neon_last_block_output_lambda_1.0",
        "random_finetune",
        "random_neon_lambda_0.5",
        "random_neon_last_block_output_lambda_0.5",
        "random_neon_last_block_output_lambda_0.75",
        "random_neon_last_block_output_lambda_1.0",
    ]
    display_cols = [
        "model",
        "unique_scaffold_fraction",
        "scaffold_entropy_normalized",
        "top10_scaffold_fraction",
        "sphere_exclusion_circles",
        "unique_morgan_features",
        "unique_morgan_features_per_valid",
        "pairwise_tanimoto_distance_mean",
        "internal_nn_tanimoto_mean",
        "close_neighbor_gt_0.7_fraction",
        "occupied_cluster_fraction",
        "effective_clusters",
        "top5_cluster_fraction",
        "js_divergence_from_base_clusters",
    ]
    display_models = list(args.models) if args.models else interesting
    display_source = metrics
    if metrics["seed"].nunique() > 1:
        display_source = metrics.groupby("model", as_index=False)[
            [column for column in display_cols if column != "model"]
        ].mean(numeric_only=True)
    display = display_source[display_source["model"].isin(display_models)][display_cols].copy()
    display["sort_key"] = display["model"].map({model: i for i, model in enumerate(display_models)})
    display = display.sort_values("sort_key").drop(columns=["sort_key"])

    report = f"""# Diversity Analysis

Input: `{results_dir}`

The K-means base-space metrics fit {args.n_clusters} clusters on base generated
Morgan fingerprints, then measure how each optimized generator occupies that
same base fingerprint space.

For replicate directories, metrics are computed separately per `seed_*`
directory and the table below reports model-wise means.

`sphere_exclusion_circles` is a greedy count of structurally separated Morgan
fingerprint centers using a Tanimoto similarity threshold of
{args.sphere_similarity_threshold:.2f}. `unique_morgan_features` counts unique
unhashed sparse Morgan features in the valid generated set.

{write_markdown_table(display)}

Interpretation guide:

- Higher unique scaffold fraction, scaffold entropy, pairwise Tanimoto distance,
  sphere-exclusion circles, Morgan features, occupied cluster fraction, and
  effective clusters indicate broader diversity.
- Lower top scaffold/cluster fractions indicate less concentration.
- Higher internal nearest-neighbor similarity or close-neighbor fraction
  indicates more local redundancy.
- Higher JS divergence from base clusters means the generator moved farther from
  the base distribution.
"""
    (output_dir / "diversity_report.md").write_text(report)
    print(report)
    print(f"Wrote diversity analysis to {output_dir}")


if __name__ == "__main__":
    main()

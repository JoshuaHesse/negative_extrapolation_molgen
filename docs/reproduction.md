# Reproduction Guide

## 1. Restore Inputs

Clone the repository, then extract the project-owned data archive into a
temporary directory and copy only its `results/` tree into the checkout. Do
not strip the archive's top-level directory directly into the checkout,
because the archive also contains release metadata named `README.md`.

```bash
mkdir -p /tmp/ne-data
tar --zstd -xf negative_extrapolation_molgen_data_v1.tar.zst -C /tmp/ne-data
cp -a /tmp/ne-data/negative_extrapolation_molgen_data_v1/results/. results/
```

Download third-party source trees and checkpoints separately using the pinned
commands in `data_manifest.md`.

## 2. Build Environments

```bash
make build
make audit
make reinvent-build
make semlaflow-build
make semlaflow-env-check
make audit-inputs
```

MLflow is disabled by default. To enable remote tracking, export
`MLFLOW_TRACKING_URI` before invoking a target. Alternatively,
`NEON_ENABLE_MLFLOW=1` explicitly enables MLflow's default local backend.

## 3. Regenerate Models and Samples

The broad confirmatory entry point is:

```bash
make paper-control-refresh
```

This runs the GuacaMol, REINVENT4, and SemlaFlow confirmatory generation
workflows. Seed-level workflows skip outputs whose completion artifacts already
exist. For resource scheduling, the component targets in the README are
preferable because the model families use different images and runtimes.

The SemlaFlow 50,000-molecule baseline pools are deliberately regenerated
rather than distributed in the project archive. After restoring the public
SemlaFlow assets, generate all ten seed-specific pools with:

```bash
make semlaflow-geom-drugs-50000-baseline-replicates
```

This writes
`results/external/semlaflow/geom_drugs_50000_seed_<seed>/` for seeds 13, 17,
19, 23, 29, 31, 37, 41, 43, and 47. The downstream SemlaFlow target consumes
these directories. The Zenodo archive contains the exact selected
training/validation molecules and final sample-level results used in the paper,
so regenerating the large baseline pools is unnecessary when reproducing only
the reported analyses and figures.

## 4. Regenerate Analyses

```bash
make paper-post-control-analysis
make paper-fcd-distances
make paper-statistics
make publication-si-tables
make audit-results
```

FCD requires a CUDA-capable device in the current workflow. CPU-heavy analyses
can be constrained with `ANALYSIS_CPUS` and `ANALYSIS_CPUSET`; FCD worker count
is controlled by `FCD_JOBS`.

## 5. Regenerate Figures

Execute the public notebooks after restoring the result archive:

```bash
make figures
```

This runs, in order:

1. `notebooks/260612_main_paper_figures.ipynb`
2. `notebooks/260617_si_figures_and_statistics.ipynb`

Figures are exported as PNG, SVG, and PDF into the ignored
`notebooks/figures/` directory. The notebooks do not define inferential tests or
new chemistry metrics.

## Determinism and Pairing

All final comparisons use the same ten seeds for each method. GPU kernels and
third-party samplers may not be bitwise deterministic across software and
hardware stacks. Re-analysis of the archived samples reproduces the exact
reported tables; regeneration from checkpoints is intended to reproduce the
reported distributional behavior. The released GuacaMol objective runner uses
separate deterministic seeds for each fine-tuning stage and each sampled model,
so adding or omitting another evaluated method does not advance the sampled
model's random-number stream.

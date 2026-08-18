# Reproduction Guide

## 1. Restore Inputs

Clone the repository, then restore the project-owned data archive so that its
`results/` and `data/` directories sit at the repository root. Download
third-party source trees and checkpoints separately as listed in
`data_manifest.md`.

## 2. Build Environments

```bash
make build
make audit
make reinvent-build
make semlaflow-build
make semlaflow-env-check
make audit-inputs
```

Use `NEON_DISABLE_MLFLOW=1` unless an MLflow server is configured.

## 3. Regenerate Models and Samples

The broad confirmatory entry point is:

```bash
NEON_DISABLE_MLFLOW=1 make paper-control-refresh
```

This runs the GuacaMol, REINVENT4, and SemlaFlow confirmatory generation
workflows. Seed-level workflows skip outputs whose completion artifacts already
exist. For resource scheduling, the component targets in the README are
preferable because the model families use different images and runtimes.

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
reported distributional behavior.

# Negative Extrapolation for Molecular Generators

This repository contains the implementation and reproducibility workflows for a
study of **negative extrapolation (NE)** as a parameter-space edit for molecular
generators. The method fine-tunes a pretrained generator on molecules containing
an unwanted feature and reverses the resulting parameter update:

```text
delta_bad = theta_bad - theta_base
theta_NE  = theta_base - lambda * delta_bad
```

The study asks whether this edit can reduce defined medicinal-chemistry
liabilities while preserving validity, novelty, and structural diversity. It
evaluates three model families:

- a PyTorch SMILES RNN trained on GuacaMol;
- a PyTorch causal SMILES Transformer trained on GuacaMol;
- published REINVENT4 and SemlaFlow pretrained models.

The four confirmatory objectives are reactive motifs, metal-binding motifs,
charged motifs, and assay-interference motifs. Their exact SMARTS definitions
are versioned in [`neon_molgen/scoring.py`](neon_molgen/scoring.py).

## Repository Scope

The public repository contains the code, experiment configurations, Docker
environments, analysis scripts, and final figure notebooks needed for the
reported study. It intentionally excludes:

- the manuscript source;
- generated molecule tables and derived analysis outputs;
- trained checkpoints and third-party pretrained weights;
- exploratory experiments that are not reported in the study.

Those boundaries keep the code reviewable and avoid redistributing third-party
models. See [`docs/data_manifest.md`](docs/data_manifest.md) for the expected
Zenodo archive layout and original model sources.

Reader-facing text uses **metal-binding motifs**. Existing configuration names,
result paths, and metric columns retain the earlier identifier `chelator` (for
example, `chelator_hit`) so the released data remain compatible with the
analysis code. Historical machine-readable names also retain `neon` (for
example, `neon_lambda_1.0`); the manuscript and public documentation use the
method name **NE**.

## Layout

```text
neon_molgen/   Core models, training, sampling, scoring, and model arithmetic
scripts/       Reproducible experiment and analysis entry points
configs/       Confirmatory and development configurations
notebooks/     Final main-text and supplementary figure notebooks
docs/          Reproduction protocol, controls, and data manifest
results/       Downloaded/generated study outputs; ignored by Git
data/          Downloaded GuacaMol data; ignored by Git
external/      Third-party source trees and weights; ignored by Git
```

## Environment

Docker is the supported execution path. GPU workflows require the NVIDIA
Container Toolkit.

```bash
make build
make test
make lint
make audit
make reinvent-build
make semlaflow-build
make semlaflow-env-check
```

The core Python package can also be installed locally:

```bash
python -m pip install -e '.[dev]'
```

REINVENT4 and SemlaFlow are not vendored. Place their source trees and official
weights under the paths listed in [`docs/data_manifest.md`](docs/data_manifest.md)
before building the corresponding images.

MLflow logging is optional. Set `MLFLOW_TRACKING_URI` to use a tracking server,
or set `NEON_DISABLE_MLFLOW=1` to disable logging.

## Reproducing the Experiments

Final analyses use ten paired confirmatory seeds:

```text
13, 17, 19, 23, 29, 31, 37, 41, 43, 47
```

Seed 11 and the scope-development seeds 5, 7, and 11 are development data and
are excluded from confirmatory statistics.

### GuacaMol RNN and Transformer

```bash
make guacamol-download
make guacamol-train-rnn-base
make guacamol-train-transformer-base
make paper-guacamol-qed-replicates
make paper-guacamol-liability-replicates
```

The Transformer liability experiments use the final block plus output layer.
The fully crossed scope-development screen can be regenerated with:

```bash
make paper-transformer-scope-development-data
```

### REINVENT4

After installing REINVENT4 and its published prior:

```bash
make paper-reinvent-liability-replicates
```

### SemlaFlow

After installing SemlaFlow and its published GEOM-Drugs checkpoint/data:

```bash
make semlaflow-geom-drugs-50000-baseline-replicates
make semlaflow-four-liability-positive-corrected-replicates
make semlaflow-four-liability-joint-replicates-posebusters
make semlaflow-four-liability-structural-analysis
```

Transient extrapolated checkpoints are deleted after scoring. The retained
selection files, configurations, training histories, and norm metadata permit
the model edits to be reconstructed.

### Analysis and Figures

With the released result archive restored under `results/`:

```bash
make paper-post-control-analysis
make paper-fcd-distances
make paper-statistics
make publication-si-tables
make audit-results
make figures
```

Generated statistical tables are written to
`results/publication/tables/`. The notebooks load the scripted CSV outputs and
only perform plotting transformations:

- `notebooks/260612_main_paper_figures.ipynb`
- `notebooks/260617_si_figures_and_statistics.ipynb`

See [`docs/reproduction.md`](docs/reproduction.md) for workflow details and
expected outputs.
The audited package versions are listed in
[`docs/software_environment.md`](docs/software_environment.md).
The exact input revisions, checksums, confirmatory seeds, and expected result
artifacts are defined in
[`configs/reproducibility_manifest.json`](configs/reproducibility_manifest.json).

## Controls and Statistics

Random NE controls are learned by fine-tuning on size-matched random samples;
they are not synthetic Gaussian directions. Their update is globally L2
norm-matched to the corresponding bad-set update within the exact parameter
scope. SemlaFlow additionally includes random-corrected and liability-free
positive-corrected directions. Exact definitions and audit metadata are in
[`docs/confirmatory_control_protocol.md`](docs/confirmatory_control_protocol.md).

The independently trained seed is the experimental unit. Confirmatory
comparisons use paired mean differences, 95% Student-t confidence intervals,
exact two-sided sign-flip tests, and prespecified Holm correction families.

## Data and Checkpoints

The Git repository is not sufficient by itself to rerun every analysis because
large generated samples and checkpoints are distributed separately. The data
manifest distinguishes project-owned study outputs, third-party inputs, and
transient checkpoints that are deliberately not archived.

## License

This repository is released under the [MIT License](LICENSE). The licenses and
citation requirements of GuacaMol, REINVENT4, SemlaFlow, PoseBusters, RDKit,
FCD, and their pretrained assets apply independently.

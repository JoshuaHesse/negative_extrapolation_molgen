# Negative Extrapolation for Targeted Liability Erasure in Molecular Generators

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

REINVENT4 and SemlaFlow are not vendored. The pinned clone, download, and file
placement commands are listed in
[`docs/data_manifest.md`](docs/data_manifest.md). Run those commands before
building the corresponding images.

MLflow logging is disabled by default. Opt in by setting `MLFLOW_TRACKING_URI`
to a tracking server, or set `NEON_ENABLE_MLFLOW=1` to use MLflow's default
local backend. `MLFLOW_EXPERIMENT_NAME` optionally overrides the experiment
name.

## Reproducing the Experiments

Final analyses use ten paired confirmatory seeds:

```text
13, 17, 19, 23, 29, 31, 37, 41, 43, 47
```

Seed 11 and the scope-development seeds 5, 7, and 11 are development data and
are excluded from confirmatory statistics.

### GuacaMol RNN and Transformer

#### Repairing Legacy Assay-Interference Results

For the audited legacy result tree only, the following one-time cleanup removes
assay-interference results affected by the old nitro SMARTS. It preserves raw
base pools, verified Transformer evaluations, and seed 47. It does not change
other objectives or the separate 10k REINVENT base evaluation.

```bash
make paper-assay-nitro-plan   # inspect the deletion summary without changing files
make paper-assay-nitro-clean  # delete stale results; record paths and hashes
make paper-assay-nitro-repair # regenerate affected arms, audit, rebuild statistics
```

This is not required for fresh runs with the current scoring code. The repair
uses the original training settings, lambdas, and sampling seeds. Completed seeds
are checksum-verified and skipped on restart; an interrupted seed restarts from
its preserved inputs. Input/configuration changes cause an error rather than
silently mixing incompatible results. Disposable edited checkpoints are not
retained. Cleanup manifests are in `results/cleanup_manifests/assay_nitro_v2`.
Stale development-seed-11 outputs are removed but are not regenerated. Figures
remain notebook-generated; rerun the figure notebooks after the repair finishes.
Existing manuscript PDFs and release archives are not replaced by these commands.

#### Standard Runs

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

For a budget-matched reevaluation of the unedited REINVENT prior, run:

```bash
make paper-reinvent-base-budget
```

This samples 10,000 attempts per confirmatory seed using the existing prior
and the same sampling settings and seed offset as the edited models. A single
base evaluation per seed is scored for all four objectives. It also computes
base scaffold/diversity metrics, with the usual `ANALYSIS_CPUS` and
`ANALYSIS_CPUSET` limits. Sampling resumes from completed, checksum-verified
outputs. Results are written separately to
`results/external/reinvent4/base_evaluation_10000`; the original 50,000-sample
training pools, selected fine-tuning sets, and edited-model results are not
modified. The paper notebooks and statistics loader require this completed
evaluation and replace only the historical Base rows when reading the results.
They do not silently fall back to the 50,000-sample evaluation. The target does
not overwrite existing figures or statistics. Fresh sampling is necessary because the old native REINVENT
CSVs omit invalid and duplicate attempts, so selecting 10,000 saved rows is
not equivalent to requesting 10,000 generations.

### SemlaFlow

After installing SemlaFlow and its published GEOM-Drugs checkpoint/data:

```bash
make semlaflow-geom-drugs-50000-baseline-replicates
make semlaflow-four-liability-cne-replicates
make semlaflow-four-liability-joint-replicates-posebusters
make semlaflow-four-liability-structural-analysis
```

The first target regenerates the 50,000-molecule baseline pool independently
for each confirmatory seed under
`results/external/semlaflow/geom_drugs_50000_seed_<seed>/`. These large,
reconstructible baseline pools are not included in the Zenodo archive. The
archive instead retains the exact selected training/validation molecules,
sampled confirmatory outputs, configurations, and analysis inputs used for the
publication.

Transient extrapolated checkpoints are deleted after scoring. The retained
selection files, configurations, training histories, and norm metadata permit
the model edits to be reconstructed.

### Analysis and Figures

With the released result archive restored under `results/`:

```bash
make paper-post-control-analysis
make paper-reinvent-base-budget
make paper-fcd-distances
make paper-fcd-mw-calibration-size-check
make paper-statistics
make publication-si-tables
make audit-results
make figures
```

The SI molecular-weight calibration uses equal query/reference counts across
all MW bands within each generator/seed. The sample-size analysis
also retains the original and pair-matched calculations. All are drawn from
the saved base pools, and query/reference overlap is allowed; they illustrate
chemical-space restrictions rather than universal FCD/FDD thresholds.
`paper-fcd-mw-calibration-size-check` uses CPU by default and writes
`results/analysis/fcd_calibration_size_check/comparison_metrics.csv`.

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
The project-owned Zenodo data bundle can be assembled with
`make zenodo-archive`; see [`docs/data_manifest.md`](docs/data_manifest.md) for
its inclusion and exclusion policy.
The target writes `negative_extrapolation_molgen_data_v1.zip` and its SHA-256
checksum under `results/zenodo_release/`; every payload checksum is verified
before the build reports success.
The supporting dataset is published at
[`10.5281/zenodo.23165725`](https://doi.org/10.5281/zenodo.23165725).

## Controls and Statistics

Random NE controls are learned by fine-tuning on size-matched random samples;
they are not synthetic Gaussian directions. Their update is globally L2
norm-matched to the corresponding bad-set update within the exact parameter
scope. SemlaFlow CNE subtracts a direction learned from liability-free valid
molecules from the bad-set direction before extrapolation. A random-corrected
control is globally norm-matched to CNE independently within each seed. Exact
definitions and audit metadata are in
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

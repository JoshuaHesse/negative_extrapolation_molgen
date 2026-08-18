# Software Environments

The repository uses three Docker environments because the published external
models have different framework requirements. The versions below were queried
from the release images during the final repository audit. Exact sampled outputs
are distributed for byte-for-byte re-analysis because older runs were built from
the then-current versions of previously unpinned direct dependencies.

| Package | Core GuacaMol/analysis | REINVENT4 | SemlaFlow |
|---|---:|---:|---:|
| Python | 3.11.10 | 3.11.14 | 3.11.10 |
| PyTorch | 2.5.1+cu124 | 2.9.1+cu128 | 2.5.1+cu124 |
| NumPy | 2.1.2 | 1.26.4 | 1.26.2 |
| pandas | 2.3.3 | 2.3.3 | 2.2.2 |
| SciPy | 1.17.1 | 1.17.1 | 1.11.4 |
| RDKit | 2026.03.5 | 2026.03.3 | 2026.03.3 |
| Lightning | not used | not used | 2.6.5 |
| PoseBusters | not used | not used | 0.6.5 |

Core direct dependencies are pinned in `requirements.txt`. The REINVENT4 image
is installed with the upstream `install.py` workflow, while SemlaFlow-specific
compatibility pins are recorded in `Dockerfile.semlaflow`.

Base-image digests are pinned in the Dockerfiles. Exact external Git revisions
and input checksums are recorded in `configs/reproducibility_manifest.json`.
The figure environment additionally pins `ipykernel==6.29.4` and
`nbconvert==7.16.3` in `requirements-notebooks.txt`.

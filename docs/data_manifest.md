# Data and Model Manifest

Large or externally licensed files are deliberately excluded from Git.

## Project-Owned Study Archive

The Zenodo deposit should preserve repository-relative paths and include:

- GuacaMol acquisition metadata or the training file when redistribution is
  appropriate;
- `results/guacamol_rnn_base/` and `results/guacamol_transformer_base/`;
- GuacaMol QED and four-liability outputs under `results/objectives/`;
- Transformer scope-development and epoch-sensitivity outputs under
  `results/development/`;
- the four REINVENT4 liability outputs under `results/external/reinvent4/`;
- SemlaFlow 50,000-sample base generations and confirmatory joint-liability
  outputs under `results/external/semlaflow/`;
- `results/paper_statistics/` and all scripted analysis subdirectories;
- selection CSVs, samples, summaries, histories, run configurations, and
  norm-matching metadata.

Derived NE checkpoints are not required in the archive. They can be recreated
from the base checkpoint, reproducible tuned update, parameter scope, lambda,
and retained metadata. A local deletion audit is kept at
`results/cleanup_manifests/derived_checkpoint_cleanup_2026-08-17.tsv`.

## Third-Party Inputs

Do not copy third-party repositories or pretrained weights into GitHub or the
project-owned Zenodo release unless their redistribution terms explicitly allow
it. Restore them under these paths:

```text
external/REINVENT4/
external/REINVENT4/priors/reinvent.prior
external/semla-flow/
external/semla-flow/assets/models/geom-drugs/200epochs.ckpt
external/semla-flow/assets/data/geom-drugs/smol/
```

Use the official REINVENT4 and SemlaFlow repositories/releases corresponding to
the manuscript. Record release tags, download URLs, checksums, and access dates
in the Zenodo metadata before deposition.

## Not Publicly Distributed

- manuscript source in `paper/`;
- abandoned active-learning, docking, antibiotic-prediction, Mol2Mol,
  LinkInvent, FlowMol, held-out-motif, and residual-direction experiments;
- transient generated checkpoints;
- local MLflow stores and container caches.

Non-paper exploratory material remains in the dated private archive created
during cleanup and is not needed to reproduce the study.

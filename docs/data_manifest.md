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

Use the exact revisions and SHA-256 checksums in
`configs/reproducibility_manifest.json`; verify them with `make audit-inputs`.

- REINVENT4: commit `d082b365713771c7e6de2b7053fb3444bccc0918` from
  `https://github.com/MolecularAI/REINVENT4.git`. Public priors are distributed
  at `https://doi.org/10.5281/zenodo.15641296`.
- SemlaFlow: commit `3f43103d3af138b86dbe9f29fe8085e83f9a6283` from
  `https://github.com/rssrwn/semla-flow.git`. The GEOM-Drugs checkpoint and
  processed data are distributed through the Google Drive folder linked in the
  upstream README.

These revisions and files were verified on 2026-08-18. Add the project-owned
Zenodo DOI here before manuscript submission. Until that archive is deposited,
an independent user can test the code but cannot reproduce the exact archived
sample-level analyses from GitHub alone.

## Not Publicly Distributed

- manuscript source in `paper/`;
- abandoned active-learning, docking, antibiotic-prediction, Mol2Mol,
  LinkInvent, FlowMol, held-out-motif, and residual-direction experiments;
- transient generated checkpoints;
- local MLflow stores and container caches.

Non-paper exploratory material remains in the dated private archive created
during cleanup and is not needed to reproduce the study.

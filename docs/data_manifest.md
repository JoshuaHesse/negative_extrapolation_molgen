# Data and Model Manifest

Large or externally licensed files are deliberately excluded from Git.

## Project-Owned Study Archive

The Zenodo deposit should preserve repository-relative paths and include:

- GuacaMol acquisition metadata or the training file when redistribution is
  appropriate;
- `results/guacamol_rnn_base/` and `results/guacamol_transformer_base/`;
- GuacaMol QED and four-liability outputs under `results/objectives/`;
- compact Transformer scope-development summaries under `results/objectives/`
  and epoch-sensitivity outputs under `results/development/`;
- the four REINVENT4 liability outputs under `results/external/reinvent4/`;
- SemlaFlow confirmatory joint-liability selections, sample-level outputs, and
  analyses under `results/external/semlaflow/`; the reconstructible
  50,000-sample base-generation pools are intentionally excluded;
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
  `https://github.com/MolecularAI/REINVENT4.git`. The study uses
  `reinvent.prior` from immutable Zenodo record
  `https://doi.org/10.5281/zenodo.15641297`. The concept DOI resolves to newer
  releases that no longer contain this exact file.
- SemlaFlow: commit `3f43103d3af138b86dbe9f29fe8085e83f9a6283` from
  `https://github.com/rssrwn/semla-flow.git`. The GEOM-Drugs checkpoint and
  processed data are distributed through the Google Drive folder linked in the
  upstream README.

The following commands reproduce the third-party source setup from a clean
checkout:

```bash
mkdir -p external
git clone https://github.com/MolecularAI/REINVENT4.git external/REINVENT4
git -C external/REINVENT4 checkout d082b365713771c7e6de2b7053fb3444bccc0918
mkdir -p external/REINVENT4/priors
curl -L --fail \
  https://zenodo.org/api/records/15641297/files/reinvent.prior/content \
  -o external/REINVENT4/priors/reinvent.prior

git clone https://github.com/rssrwn/semla-flow.git external/semla-flow
git -C external/semla-flow checkout 3f43103d3af138b86dbe9f29fe8085e83f9a6283
python3 -m pip install --target /tmp/ne-gdown gdown==5.2.0
PYTHONPATH=/tmp/ne-gdown python3 -m gdown --folder \
  https://drive.google.com/drive/folders/1rHi5JzN05bsGRGQUcWRmDu-Ilfoa9EAT \
  --remaining-ok -O /tmp/semlaflow-assets
mkdir -p external/semla-flow/assets/data/geom-drugs/smol \
  external/semla-flow/assets/models/geom-drugs
cp /tmp/semlaflow-assets/data/geom-drugs/smol/{train,val,test}.smol \
  external/semla-flow/assets/data/geom-drugs/smol/
cp /tmp/semlaflow-assets/models/geom-drugs/200epochs.ckpt \
  external/semla-flow/assets/models/geom-drugs/
```

Run `make guacamol-download` and then `make audit-inputs`. The audit verifies
both repository revisions and every downloaded file hash.

After the SemlaFlow source, GEOM-Drugs checkpoint, and processed data have been
restored, regenerate the seed-specific 50,000-molecule baseline pools with:

```bash
make semlaflow-geom-drugs-50000-baseline-replicates
```

The target uses the ten confirmatory seeds and writes each pool to
`results/external/semlaflow/geom_drugs_50000_seed_<seed>/`. These baseline
generations are large but fully reconstructible from the public checkpoint and
are therefore not duplicated in the project-owned Zenodo deposit. The deposit
does include the exact selected `.smol` training and validation files used for
the reported SemlaFlow model edits.

These revisions and files were verified on 2026-08-27. The project-owned data
archive has the reserved DOI
[`10.5281/zenodo.21991931`](https://doi.org/10.5281/zenodo.21991931). The record
remains a Zenodo draft until publication. Until it is published, an independent
user can test the code but cannot retrieve the exact archived sample-level
analyses from GitHub alone.

Build the project-owned upload bundle with:

```bash
make zenodo-archive-dry-run
make zenodo-archive
```

The builder selects the confirmatory paper results, records the archive version,
writes per-file SHA-256 checksums, excludes third-party and transient artifacts,
and verifies the resulting `.tar.zst` archive. The upload-ready files are written
under `results/zenodo_release/`. The code repository is versioned separately;
the publication release tag provides the frozen software snapshot.

## Not Publicly Distributed

- manuscript source in `paper/`;
- abandoned active-learning, docking, antibiotic-prediction, Mol2Mol,
  LinkInvent, FlowMol, held-out-motif, and residual-direction experiments;
- transient generated checkpoints;
- local MLflow stores and container caches.

Non-paper exploratory material remains in the dated private archive created
during cleanup and is not needed to reproduce the study.

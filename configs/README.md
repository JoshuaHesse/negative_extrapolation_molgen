# Experiment Configurations

- `guacamol_*_base.json`: independently trained base-model replicates.
- `objectives/guacamol_*_qed_from_base.json`: QED control experiments.
- `objectives/guacamol_*_{reactive,chelator,charged_motif,assay_interference}*`:
  four confirmatory liability objectives.
- `objectives/guacamol_transformer_chelator_scope_development_from_base.json`:
  development-only Transformer parameter-scope screen.
- `reinvent/reinvent4_external_model.json` and the three objective overrides:
  confirmatory REINVENT4 experiments.

The legacy term `chelator` is retained in machine-readable identifiers; it is
reported as **metal-binding motifs** in reader-facing text.

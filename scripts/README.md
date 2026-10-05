# Command-Line Entry Points

Scripts are grouped by purpose:

- `train_base_models.py`, `run_objective_from_base.py`: GuacaMol training and
  objective editing;
- `run_reinvent_reactive_replicates.py`, `reinvent_negative_extrapolate.py`:
  REINVENT4 editing and checkpoint arithmetic;
- `run_semlaflow_probe.py`, `run_semlaflow_neon.py`: SemlaFlow generation,
  fine-tuning, and corrected NE;
- `analyze_*`, `summarize_*`, `recompute_fdd_columns.py`: chemistry, diversity,
  distance, PoseBusters, and statistical analyses;
- `export_si_tables.py`: generated supplementary tables.
- `evaluate_reinvent_base_budget.py`: independent 10,000-attempt prior
  evaluations used as the fixed-budget baseline;
- `analyze_fcd_calibration_size_check.py`: sample-count-controlled
  molecular-weight calibration;
- `audit_reinvent_assay_scoring.py`, `repair_assay_nitro.py`: scoring
  consistency checks and explicit repair of legacy nitro-SMARTS results;
- `audit_reproducibility.py`, `build_zenodo_archive.py`: input/result auditing
  and the checksum-verified Zenodo ZIP bundle.

Prefer the named Make targets in the root README. They record the exact seeds,
model scopes, sample sizes, and output paths used in the study.

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

Prefer the named Make targets in the root README. They record the exact seeds,
model scopes, sample sizes, and output paths used in the study.

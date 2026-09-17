# Confirmatory Control Protocol

## Seed policy

Final endpoint analyses use paired seeds `13, 17, 19, 23, 29, 31, 37, 41, 43, 47`. Seed `11` was used for method and hyperparameter development and is excluded from confirmatory tables, statistics, and figures.

## Trained-random negative extrapolation

Random negative-extrapolation controls use learned updates, not synthetic Gaussian vectors. For each base checkpoint and objective, a separate model copy is fine-tuned on a size-matched random subset from the corresponding base-generation pool with the same optimization settings used to learn the bad-set direction.

Let `delta_bad = theta_bad - theta_base` and `delta_random = theta_random - theta_base`. Within the exact parameter scope used by the corresponding NE experiment, the random update is globally L2 norm-matched to the bad update:

```text
scale_random = ||delta_bad||_2 / ||delta_random||_2
theta_random_ne = theta_base - lambda * scale_random * delta_random
```

The norm is calculated once over all floating-point tensors in scope. It is not matched independently per tensor. This preserves the learned random-data direction while controlling total parameter displacement.

## SemlaFlow corrected negative extrapolation

SemlaFlow CNE uses a liability-free task vector as a validity-correction
direction. Let `delta_free = theta_free - theta_base`, where `theta_free` is
fine-tuned on valid molecules that contain none of the targeted motifs:

```text
v_cne = delta_bad - delta_free
theta_cne = theta_base - lambda * v_cne
```

The liability-free direction is intended to remove shared changes associated
with continued training on valid chemistry from the bad-set direction. It is
used as a correction, not as a separate enrichment objective. CNE and standard
NE use the reported lambda without an additional norm scale.

The random-corrected mechanistic control uses

```text
v_random_control = delta_bad - delta_random
scale_control = ||v_cne||_2 / ||v_random_control||_2
theta_random_control = theta_base - lambda * scale_control * v_random_control
```

This matches the random-corrected direction to CNE's global full-model L2 norm
within each replicate seed. Pure random NE is separately matched to standard NE
using the first definition above; no scale factor is shared across replicates.

## Persistent audit metadata

GuacaMol runs write `random_neon_norm_matching*.json`. REINVENT writes the trained-random definition and norm factors to `neon_scope_summary.json` and a completion marker. SemlaFlow writes raw task-vector norms and the random-control-to-CNE scale factor to `models/neon_scope_summary.json`; this file is retained even when transient model checkpoints are deleted. Legacy SemlaFlow artifact paths use `positive_corrected` for CNE and `norm_matched_random_corrected` for its control; these identifiers are retained solely for compatibility with the released result archive.

For GuacaMol seeds whose original bad checkpoint was removed during storage cleanup, the bad model is deterministically reconstructed from the saved selection and original optimization settings. Standard NE and random NE are both resampled from that same reconstructed direction, avoiding a comparison between an old NE direction and a newly reconstructed control direction. Reconstructed tuned checkpoints and transient extrapolated checkpoints may be deleted after scoring; the retained inputs and metadata permit reconstruction. REINVENT likewise deletes transient random-NE and transfer-learning checkpoints after their outputs and audit metadata have been written.

## Reproduction targets

The aggregate refresh target is:

```bash
make paper-control-refresh
```

MLflow logging is disabled by default, so an unavailable tracking server cannot
interrupt the confirmatory computation. To opt in, set `MLFLOW_TRACKING_URI` to
the server URL before running the target. Set `NEON_ENABLE_MLFLOW=1` instead to
use MLflow's default local backend.

The component targets are documented in the Makefile and can be run independently to manage GPU availability.

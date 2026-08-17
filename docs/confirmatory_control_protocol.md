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

## SemlaFlow corrected control

SemlaFlow additionally compares positive-corrected NE against a trained-random corrected control:

```text
v_positive = delta_bad - delta_positive
v_random = delta_bad - delta_random
scale_corrected_random = ||v_positive||_2 / ||v_random||_2
```

The random-corrected checkpoint applies `scale_corrected_random * v_random`, so both corrected directions have the same global full-model L2 norm at a given lambda. Pure random NE is separately matched to standard NE using the first definition above.

## Persistent audit metadata

GuacaMol runs write `random_neon_norm_matching*.json`. REINVENT writes the trained-random definition and norm factors to `neon_scope_summary.json` and a completion marker. SemlaFlow writes all raw norms, reference norms, and scale factors to `models/neon_scope_summary.json`; this file is retained even when transient model checkpoints are deleted.

For GuacaMol seeds whose original bad checkpoint was removed during storage cleanup, the bad model is deterministically reconstructed from the saved selection and original optimization settings. Standard NE and random NE are both resampled from that same reconstructed direction, avoiding a comparison between an old NE direction and a newly reconstructed control direction. Reconstructed tuned checkpoints and transient extrapolated checkpoints may be deleted after scoring; the retained inputs and metadata permit reconstruction. REINVENT likewise deletes transient random-NE and transfer-learning checkpoints after their outputs and audit metadata have been written.

## Reproduction targets

The aggregate refresh target is:

```bash
make paper-control-refresh
```

This aggregate target disables optional MLflow logging by default so an unavailable
tracking server cannot interrupt the confirmatory computation. To retain MLflow
tracking when the configured server is reachable, run
`NEON_DISABLE_MLFLOW=0 make paper-control-refresh`.

The component targets are documented in the Makefile and can be run independently to manage GPU availability.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from reinvent.models.meta_data import update_model_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create REINVENT NEON checkpoints by extrapolating away from a bad fine-tuned checkpoint."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--bad", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--random-controls",
        action="store_true",
        help="Also write a globally norm-matched random-NE checkpoint.",
    )
    parser.add_argument(
        "--random-trained",
        default=None,
        help=(
            "Checkpoint fine-tuned on a matched random molecular subset. Required with "
            "--random-controls; synthetic random vectors are not used."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--scope-name",
        default="full_model",
        help="Name for the parameter scope used in metadata and scope_summary.json.",
    )
    parser.add_argument(
        "--include-prefixes",
        nargs="*",
        default=None,
        help=(
            "Optional tensor-name prefixes to update. If omitted, all tensors are updated. "
            "Example for REINVENT LSTM last layer plus output: "
            "_rnn.weight_ih_l2 _rnn.weight_hh_l2 _rnn.bias_ih_l2 _rnn.bias_hh_l2 _linear."
        ),
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def network_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "decorator" in checkpoint:
        return checkpoint["decorator"]["state"]
    if "network_state" in checkpoint:
        return checkpoint["network_state"]
    if "network" in checkpoint:
        return checkpoint["network"]
    raise KeyError(f"Could not find a known REINVENT network state key in {checkpoint.keys()}.")


def format_lambda(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def in_scope(key: str, prefixes: list[str] | None) -> bool:
    if not prefixes:
        return True
    return any(key.startswith(prefix) for prefix in prefixes)


def global_delta_norm(deltas: dict[str, torch.Tensor], scoped_keys: list[str]) -> float:
    squared_norm = 0.0
    for key in scoped_keys:
        delta = deltas[key].detach().to(dtype=torch.float32, device="cpu")
        squared_norm += float(torch.sum(delta * delta))
    return squared_norm**0.5


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    bad_path = Path(args.bad)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_checkpoint = load_checkpoint(base_path)
    bad_checkpoint = load_checkpoint(bad_path)
    random_checkpoint = load_checkpoint(Path(args.random_trained)) if args.random_trained else None
    base_state = network_state(base_checkpoint)
    bad_state = network_state(bad_checkpoint)
    random_state = network_state(random_checkpoint) if random_checkpoint is not None else None

    if args.random_controls and random_state is None:
        legacy_note = (
            " The legacy --random-seed synthetic-vector control has been disabled."
            if args.random_seed is not None
            else ""
        )
        raise ValueError(
            "--random-controls requires --random-trained with a checkpoint learned "
            f"from matched random molecular samples.{legacy_note}"
        )

    if set(base_state) != set(bad_state):
        missing_bad = sorted(set(base_state) - set(bad_state))
        missing_base = sorted(set(bad_state) - set(base_state))
        raise ValueError(
            "Base and bad checkpoints do not have matching tensor keys. "
            f"Missing in bad: {missing_bad[:10]}; missing in base: {missing_base[:10]}"
        )
    if random_state is not None and set(base_state) != set(random_state):
        missing_random = sorted(set(base_state) - set(random_state))
        missing_base = sorted(set(random_state) - set(base_state))
        raise ValueError(
            "Base and random-trained checkpoints do not have matching tensor keys. "
            f"Missing in random: {missing_random[:10]}; missing in base: {missing_base[:10]}"
        )

    for key in base_state:
        if not torch.is_tensor(base_state[key]) or not torch.is_tensor(bad_state[key]):
            raise TypeError(f"Network entry {key!r} is not a tensor in both checkpoints.")
        if base_state[key].shape != bad_state[key].shape:
            raise ValueError(
                f"Shape mismatch for {key}: base={tuple(base_state[key].shape)}, "
                f"bad={tuple(bad_state[key].shape)}"
            )
        if random_state is not None and base_state[key].shape != random_state[key].shape:
            raise ValueError(
                f"Shape mismatch for {key}: base={tuple(base_state[key].shape)}, "
                f"random={tuple(random_state[key].shape)}"
            )

    scoped_keys = [
        key
        for key, value in base_state.items()
        if in_scope(key, args.include_prefixes) and value.is_floating_point()
    ]
    if not scoped_keys:
        raise ValueError(
            f"Scope {args.scope_name!r} selected no tensors. "
            f"Available keys: {sorted(base_state)[:20]}"
        )

    bad_deltas = {
        key: (bad_state[key] - base_state[key]) if key in scoped_keys else torch.zeros_like(base_state[key])
        for key in base_state
    }
    deltas = bad_deltas

    scope_summary = {
        "scope_name": args.scope_name,
        "include_prefixes": args.include_prefixes,
        "random_trained": args.random_trained,
        "n_total_tensors": len(base_state),
        "n_scoped_tensors": len(scoped_keys),
        "scoped_keys": scoped_keys,
        "excluded_keys": [key for key in base_state if key not in scoped_keys],
    }
    (output_dir / "neon_scope_summary.json").write_text(json.dumps(scope_summary, indent=2))

    random_deltas: dict[str, torch.Tensor] = {}
    random_direction_scale = None
    if args.random_controls:
        random_deltas = {
            key: (random_state[key] - base_state[key]) if key in scoped_keys else torch.zeros_like(base_state[key])
            for key in base_state
        }
        neon_direction_norm = global_delta_norm(deltas, scoped_keys)
        random_direction_norm = global_delta_norm(random_deltas, scoped_keys)
        if random_direction_norm == 0.0:
            raise ValueError("Random fine-tuning produced a zero parameter update.")
        random_direction_scale = neon_direction_norm / random_direction_norm
        scope_summary["random_control_definition"] = "trained_random_global_scope_l2_v1"
        scope_summary["neon_direction_norm"] = neon_direction_norm
        scope_summary["random_direction_raw_norm"] = random_direction_norm
        scope_summary["random_direction_scale"] = random_direction_scale
        scope_summary["random_direction_applied_norm"] = random_direction_norm * random_direction_scale
        (output_dir / "neon_scope_summary.json").write_text(json.dumps(scope_summary, indent=2))

    for scale in args.lambda_values:
        neon_checkpoint = load_checkpoint(base_path)
        neon_state = network_state(neon_checkpoint)
        for key in base_state:
            if key in scoped_keys:
                neon_state[key] = base_state[key] - scale * deltas[key]
            else:
                neon_state[key] = base_state[key]

        metadata = neon_checkpoint.get("metadata")
        if isinstance(metadata, dict):
            metadata.setdefault("comments", []).append(
                f"NEON extrapolation lambda={scale} scope={args.scope_name} from {bad_path.name}"
            )

        neon_checkpoint = update_model_data(
            neon_checkpoint,
            comment=f"NEON lambda={scale} scope={args.scope_name}",
        )
        output_path = output_dir / f"neon_lambda_{format_lambda(scale)}.model"
        torch.save(neon_checkpoint, output_path)
        print(f"Wrote {output_path}")

        if args.random_controls:
            random_checkpoint = load_checkpoint(base_path)
            random_state = network_state(random_checkpoint)
            for key in base_state:
                if key in scoped_keys:
                    random_state[key] = (
                        base_state[key]
                        - scale * random_direction_scale * random_deltas[key]
                    )
                else:
                    random_state[key] = base_state[key]

            metadata = random_checkpoint.get("metadata")
            if isinstance(metadata, dict):
                metadata.setdefault("comments", []).append(
                    f"Trained-random globally norm-matched control lambda={scale} scope={args.scope_name}"
                )
            random_checkpoint = update_model_data(
                random_checkpoint,
                comment=f"Trained-random norm-matched NE control lambda={scale} scope={args.scope_name}",
            )
            random_output_path = output_dir / f"random_neon_lambda_{format_lambda(scale)}.model"
            torch.save(random_checkpoint, random_output_path)
            print(f"Wrote {random_output_path}")


if __name__ == "__main__":
    main()

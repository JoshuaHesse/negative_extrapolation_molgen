from __future__ import annotations

import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd

_UNAVAILABLE_BACKENDS: set[tuple[str, str]] = set()
_WARNED_BACKENDS: set[tuple[str, str]] = set()


def _environment_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def mlflow_enabled(config: dict[str, Any]) -> bool:
    # Tracking is opt-in. Keep the disable flag as a backward-compatible
    # override for existing private workflows.
    if _environment_flag("NEON_DISABLE_MLFLOW"):
        return False
    mlflow_config = config.get("mlflow", {})
    return bool(
        _environment_flag("NEON_ENABLE_MLFLOW")
        or mlflow_config.get("enabled", False)
        or mlflow_config.get("tracking_uri")
        or os.getenv("MLFLOW_TRACKING_URI")
    )


def mlflow_backend(config: dict[str, Any]) -> tuple[str, str]:
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI") or ""
    experiment = (
        mlflow_config.get("experiment")
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or "neon_molecular_generation"
    )
    return str(tracking_uri), str(experiment)


def mlflow_required(config: dict[str, Any]) -> bool:
    return bool(config.get("mlflow", {}).get("required", False))


def mark_mlflow_unavailable(config: dict[str, Any], error: Exception) -> None:
    if mlflow_required(config):
        raise error
    backend = mlflow_backend(config)
    _UNAVAILABLE_BACKENDS.add(backend)
    if backend not in _WARNED_BACKENDS:
        uri, experiment = backend
        warnings.warn(
            "MLflow tracking is unavailable; continuing without remote tracking for "
            f"this process (uri={uri or 'default'}, experiment={experiment}): "
            f"{type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED_BACKENDS.add(backend)


def get_mlflow(config: dict[str, Any]):
    if not mlflow_enabled(config):
        return None
    if mlflow_backend(config) in _UNAVAILABLE_BACKENDS:
        return None
    try:
        import mlflow
    except ImportError as error:
        wrapped = ImportError(
            "MLflow tracking is enabled, but the mlflow package is not installed. "
            "Install it in the container or remove the MLflow opt-in setting."
        )
        if mlflow_required(config):
            raise wrapped from error
        mark_mlflow_unavailable(config, wrapped)
        return None

    tracking_uri, experiment = mlflow_backend(config)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(experiment)
    except Exception as error:
        mark_mlflow_unavailable(config, error)
        return None
    return mlflow


class SafeRunContext:
    """Keep optional MLflow lifecycle failures from aborting local experiments."""

    def __init__(self, config: dict[str, Any], context: Any):
        self.config = config
        self.context = context
        self.entered = False

    def __enter__(self):
        try:
            value = self.context.__enter__()
            self.entered = True
            return value
        except Exception as error:
            mark_mlflow_unavailable(self.config, error)
            return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self.entered:
            return False
        try:
            return bool(self.context.__exit__(exc_type, exc_value, traceback))
        except Exception as error:
            mark_mlflow_unavailable(self.config, error)
            # Never suppress an exception raised by the experiment body.
            return False


def start_run(config: dict[str, Any], run_name: str):
    mlflow = get_mlflow(config)
    if mlflow is None:
        return nullcontext(None)
    try:
        return SafeRunContext(config, mlflow.start_run(run_name=run_name))
    except Exception as error:
        mark_mlflow_unavailable(config, error)
        return nullcontext(None)


def flatten_params(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    params = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            params.update(flatten_params(value, name))
        elif isinstance(value, (list, tuple)):
            params[name] = ",".join("null" if item is None else str(item) for item in value)
        elif value is None:
            params[name] = "null"
        elif isinstance(value, (str, int, float, bool)):
            params[name] = value
        else:
            params[name] = str(value)
    return params


def log_params(config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    mlflow = get_mlflow(config)
    if mlflow is None:
        return
    params = flatten_params({key: value for key, value in config.items() if key != "mlflow"})
    if extra:
        params.update(flatten_params(extra))
    # MLflow has a practical per-run param limit. Keep the config artifact as
    # the source of truth and use params for search/filter fields.
    for key, value in params.items():
        try:
            mlflow.log_param(key[:250], value)
        except Exception as error:
            mark_mlflow_unavailable(config, error)
            return


def log_metrics_from_mapping(
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    prefix: str = "",
    step: int | None = None,
) -> None:
    mlflow = get_mlflow(config)
    if mlflow is None:
        return
    for key, value in metrics.items():
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and pd.notna(value):
            name = f"{prefix}{key}" if prefix else key
            try:
                mlflow.log_metric(name[:250], float(value), step=step)
            except Exception as error:
                mark_mlflow_unavailable(config, error)
                return


def log_table_metrics(
    config: dict[str, Any],
    frame: pd.DataFrame,
    *,
    row_key: str,
    metric_prefix: str = "",
) -> None:
    for _, row in frame.iterrows():
        prefix = f"{metric_prefix}{row[row_key]}/"
        log_metrics_from_mapping(config, row.to_dict(), prefix=prefix)


def log_artifacts(config: dict[str, Any], paths: list[str | Path]) -> None:
    mlflow = get_mlflow(config)
    if mlflow is None:
        return
    for path in paths:
        path = Path(path)
        if path.exists() and path.is_file():
            try:
                mlflow.log_artifact(str(path))
            except Exception as error:
                mark_mlflow_unavailable(config, error)
                return

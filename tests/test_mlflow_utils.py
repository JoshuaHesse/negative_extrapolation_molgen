from neon_molgen.mlflow_utils import mlflow_enabled


def test_mlflow_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NEON_ENABLE_MLFLOW", raising=False)
    monkeypatch.delenv("NEON_DISABLE_MLFLOW", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert not mlflow_enabled({})
    assert not mlflow_enabled({"mlflow": {"enabled": False, "tracking_uri": ""}})


def test_mlflow_can_be_enabled_by_environment(monkeypatch):
    monkeypatch.delenv("NEON_DISABLE_MLFLOW", raising=False)
    monkeypatch.setenv("NEON_ENABLE_MLFLOW", "1")

    assert mlflow_enabled({})


def test_tracking_uri_is_an_opt_in(monkeypatch):
    monkeypatch.delenv("NEON_ENABLE_MLFLOW", raising=False)
    monkeypatch.delenv("NEON_DISABLE_MLFLOW", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example:8085")

    assert mlflow_enabled({})
    assert mlflow_enabled({"mlflow": {"tracking_uri": "file:///tmp/mlruns"}})


def test_disable_override_has_precedence(monkeypatch):
    monkeypatch.setenv("NEON_ENABLE_MLFLOW", "1")
    monkeypatch.setenv("NEON_DISABLE_MLFLOW", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example:8085")

    assert not mlflow_enabled({"mlflow": {"enabled": True}})

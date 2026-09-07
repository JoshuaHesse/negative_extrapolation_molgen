from scripts.run_objective_from_base import derive_seed


def test_derived_seed_is_stable_and_namespaced():
    assert derive_seed(13, "sample:positive") == derive_seed(13, "sample:positive")
    assert derive_seed(13, "sample:positive") != derive_seed(13, "sample:bad")
    assert derive_seed(13, "sample:positive") != derive_seed(17, "sample:positive")


def test_derived_seed_fits_torch_seed_range():
    value = derive_seed(47, "sample:neon_lambda_1.0")
    assert 0 <= value < 2**31

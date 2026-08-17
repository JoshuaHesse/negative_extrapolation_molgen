import math

from neon_molgen.scoring import score_smiles, summarize_scores


def test_confirmatory_liability_patterns_match_expected_examples() -> None:
    scores = score_smiles(
        [
            "CC(=O)Cl",
            "Oc1ccccc1O",
            "C[N+](C)(C)C",
            "[O-][N+](=O)c1ccccc1",
            "CCO",
        ]
    )

    assert scores.loc[0, "reactive_hit"] == 1.0
    assert scores.loc[1, "chelator_hit"] == 1.0
    assert scores.loc[2, "charged_motif_hit"] == 1.0
    assert scores.loc[3, "assay_interference_hit"] == 1.0
    assert scores.loc[4, "four_liability_hit"] == 0.0


def test_usable_yield_uses_original_sampling_budget() -> None:
    scores = score_smiles(["CCO", "CCO", "CC(=O)Cl", "not-a-smiles"])
    summary = summarize_scores(
        scores,
        reference_canonical_smiles={"CC(=O)Cl"},
        liability_free_column="reactive_hit",
        n_sampled=10,
    )

    assert summary["n_sampled"] == 10.0
    assert summary["n_valid"] == 3.0
    assert summary["n_valid_unique"] == 2.0
    assert summary["n_valid_unique_novel"] == 1.0
    assert summary["n_valid_unique_novel_liability_free"] == 1.0
    assert math.isclose(summary["usable_yield"], 0.1)

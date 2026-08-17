import pandas as pd

from scripts.run_reinvent_reactive_replicates import select_binary_sets


def test_binary_selection_is_balanced_capped_and_deterministic() -> None:
    scores = pd.DataFrame(
        {
            "canonical_smiles": [f"mol-{index}" for index in range(12)],
            "valid": [True] * 11 + [False],
            "reactive_hit": [1.0] * 4 + [0.0] * 8,
        }
    )
    selection = {
        "bad_where": "reactive_hit",
        "good_where": "reactive_hit",
        "bad_value": 1.0,
        "good_value": 0.0,
        "balance": True,
        "max_smiles": 3,
        "min_smiles": 2,
    }

    bad_a, good_a, metadata = select_binary_sets(scores, selection, seed=13)
    bad_b, good_b, _ = select_binary_sets(scores, selection, seed=13)

    assert len(bad_a) == len(good_a) == 3
    assert metadata["n_bad_raw"] == 4
    assert metadata["n_good_raw"] == 7
    assert bad_a["canonical_smiles"].tolist() == bad_b["canonical_smiles"].tolist()
    assert good_a["canonical_smiles"].tolist() == good_b["canonical_smiles"].tolist()

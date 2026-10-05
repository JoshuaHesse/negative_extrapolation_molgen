from unittest import TestCase

import numpy as np
import pandas as pd

from scripts.analyze_fcd_calibration_size_check import make_comparisons, validate_original


class CalibrationSamplingTests(TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"smiles": [f"mol_{i}" for i in range(1000)], "mw": np.arange(1000)})
        self.plan = make_comparisons(self.frame, seed=13, cap=500, minimum=50)

    def test_pair_matched_retains_query_and_matches_reference(self):
        original = {r["band"]: r for r in self.plan if r["mode"] == "original"}
        for row in self.plan:
            if row["mode"] != "pair_matched":
                continue
            old = original[row["band"]]
            self.assertEqual(row["query"], old["query"])
            self.assertEqual(len(row["query"]), len(row["reference"]))
            self.assertEqual(row["reference"], old["reference"][:len(row["query"])])

    def test_common_size_is_identical_across_all_bands_and_sides(self):
        common = [r for r in self.plan if r["mode"] == "common_size"]
        self.assertEqual(len(common), 8)
        self.assertEqual({len(r[key]) for r in common for key in ["query", "reference"]}, {100})
        self.assertTrue(all(len(set(r["query"])) == 100 for r in common))
        self.assertTrue(all(r["reference"] == common[0]["reference"] for r in common))

    def test_deterministic_sampling(self):
        self.assertEqual(self.plan, make_comparisons(self.frame, 13, 500, 50))

    def test_fails_for_underpowered_band(self):
        with self.assertRaisesRegex(ValueError, "minimum"):
            make_comparisons(self.frame, 13, 500, 150)

    def test_original_input_validation(self):
        original = pd.DataFrame([
            {**{k: v for k, v in row.items() if k not in ["query", "reference"]},
             "n_compared": len(row["query"]), "n_reference": len(row["reference"])}
            for row in self.plan if row["mode"] == "original"
        ])
        validate_original(self.plan, original)
        original.loc[0, "n_reference"] = 999
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_original(self.plan, original)

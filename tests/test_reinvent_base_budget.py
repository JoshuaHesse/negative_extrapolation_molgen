from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from scripts import evaluate_reinvent_base_budget as evaluation


class BaseBudgetTests(TestCase):
    def test_requested_attempts_are_the_denominator(self):
        from neon_molgen.scoring import score_smiles

        scores = score_smiles(["CCO", "CCO", "CC(=O)Cl", "not-a-smiles"])
        metrics = evaluation.summarize_objectives(scores, seed=13, n_sampled=10)
        self.assertEqual(set(metrics.objective), set(evaluation.OBJECTIVES))
        self.assertTrue(metrics.n_sampled.eq(10).all())
        reactive = metrics.set_index("objective").loc["reactive"]
        self.assertAlmostEqual(reactive.usable_yield, 0.1)
        self.assertEqual(reactive.model, "base")
        self.assertEqual(reactive.seed, 13)

    def test_sampling_resumes_without_overwriting_original_pools(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            prior = root / "prior.model"
            prior.write_bytes(b"test checkpoint")
            args = Namespace(output_dir=root / "evaluation", prior=prior, eval_samples=10, device="cpu")
            original = root / "original_pool.csv"
            original.write_text("original training pool\n")

            def sample(command):
                self.assertEqual(command[:3], ["reinvent", "--seed", "1013"])
                config = Path(command[-1]).read_text()
                self.assertIn("num_smiles = 10", config)
                self.assertIn("unique_molecules = true", config)
                self.assertIn("randomize_smiles = true", config)
                pd.DataFrame({"SMILES": ["CCO", "CC(=O)Cl"]}).to_csv(
                    args.output_dir / "seed_13/prior_samples.pending.csv", index=False,
                )

            with patch.object(evaluation, "run_command", side_effect=sample) as run:
                first = evaluation.evaluate_seed(args, 13, evaluation.file_hash(prior))
                second = evaluation.evaluate_seed(args, 13, evaluation.file_hash(prior))
                self.assertEqual(run.call_count, 1)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(original.read_text(), "original training pool\n")
            self.assertEqual(len(pd.read_csv(args.output_dir / "seed_13/base_samples.csv")), 2)
            self.assertTrue(first.n_sampled.eq(10).all())
            args.eval_samples = 20
            with self.assertRaisesRegex(ValueError, "Configuration changed"):
                evaluation.evaluate_seed(args, 13, evaluation.file_hash(prior))

    def test_modified_complete_output_is_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            args = Namespace(output_dir=root, prior=root / "prior.model", eval_samples=10, device="cpu")

            def sample(command):
                pd.DataFrame({"SMILES": ["CCO"]}).to_csv(
                    root / "seed_13/prior_samples.pending.csv", index=False,
                )

            with patch.object(evaluation, "run_command", side_effect=sample):
                evaluation.evaluate_seed(args, 13, "test hash")
            (root / "seed_13/prior_samples.csv").write_text("SMILES\nCCC\n")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                evaluation.evaluate_seed(args, 13, "test hash")

    def test_cannot_count_more_rows_than_requested_attempts(self):
        with self.assertRaisesRegex(ValueError, "exceed"):
            evaluation.summarize_objectives(pd.DataFrame({"smiles": ["C", "CC"]}), 13, 1)

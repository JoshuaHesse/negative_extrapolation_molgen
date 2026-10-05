from unittest import TestCase

from neon_molgen.scoring import make_liability_patterns
from scripts.audit_reinvent_assay_scoring import score_records


class AssayScoringAuditTests(TestCase):
    def test_nitro_only_molecule_is_not_motif_free(self):
        patterns = make_liability_patterns()["assay_interference"]
        frame = score_records(["O=[N+]([O-])c1ccccc1", "CCO"], patterns, {})
        self.assertEqual(frame.current_hit.tolist(), [True, False])
        self.assertEqual(frame.nitro_hit.tolist(), [True, False])
        self.assertFalse(frame.without_nitro_hit.any())

    def test_cache_retains_duplicate_occurrences(self):
        cache = {}
        patterns = make_liability_patterns()["assay_interference"]
        frame = score_records(["CCO", "CCO"], patterns, cache)
        self.assertEqual(len(cache), 1)
        self.assertEqual(len(frame), 2)
        self.assertEqual(len(frame.drop_duplicates("canonical_smiles")), 1)

    def test_unparseable_saved_molecule_fails(self):
        with self.assertRaisesRegex(ValueError, "cannot be parsed"):
            score_records(["not-a-smiles"], make_liability_patterns()["assay_interference"], {})

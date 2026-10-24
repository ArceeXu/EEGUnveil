import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from src.offline import search_openbmi_lambda as search


class LambdaSearchTests(unittest.TestCase):
    def test_missing_baseline_is_never_scheduled(self):
        args = SimpleNamespace(baseline_root=Path("baseline"), search_root=Path("search"))
        with patch.object(search, "complete", return_value=False), \
             patch.object(Path, "exists", return_value=False):
            jobs = search.planned_jobs(args, {})
        self.assertEqual(len(jobs), 12)
        self.assertNotIn(.75, [job[0] for job in jobs])

    def rows(self):
        return [dict(lambda_value=lam, seed=seed, normal=.8, bci_blind=.6,
                     overall=.65, score=.7, **{"lambda": lam})
                for lam in search.LAMBDAS for seed in search.SEEDS]

    def test_selection_uses_seed_mean_and_predeclared_tie(self):
        rows = self.rows()
        summaries, best = search.aggregate(rows)
        self.assertEqual(best["lambda"], .75)
        self.assertEqual(len(summaries), 5)
        for row in rows:
            if row["lambda"] == .625:
                row["score"] = .72
            if row["lambda"] == .9375:
                row["score"] = .99 if row["seed"] == 42 else .5
        self.assertEqual(search.aggregate(rows)[1]["lambda"], .625)

    def test_missing_seed_is_not_silently_averaged(self):
        with self.assertRaises(ValueError):
            search.aggregate(self.rows()[:-1])

    def test_protocol_or_split_mismatch_rejected(self):
        split = {"train": [1], "val": [2], "test": [3]}
        config = dict(search.PROTOCOL, seed=42, mix_lambda=.75,
                      effective_mix_lambda=.75, subject_split=split)
        metrics = {"subject_macro_accuracy": .7,
                   "report": {"group_accuracy": {"normal": .8, "bci_blind": .6}}}
        with patch.object(search, "complete", return_value=True):
            with patch.object(search, "read_json", side_effect=[config, split, metrics]):
                row = search.validate_result(Path("unused"), split, .75, 42)
                self.assertAlmostEqual(row["score"], .7)
            with patch.object(search, "read_json", return_value=dict(config, batch_size=16)):
                with self.assertRaises(ValueError):
                    search.validate_result(Path("unused"), split, .75, 42)
            with patch.object(search, "read_json", side_effect=[config, {}]):
                with self.assertRaises(ValueError):
                    search.validate_result(Path("unused"), split, .75, 42)


if __name__ == "__main__":
    unittest.main()

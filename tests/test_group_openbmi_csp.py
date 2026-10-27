import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.tools.group_openbmi_csp import group_name, load_training, process_file, score_training


class GroupTests(unittest.TestCase):
    def test_threshold_boundary(self):
        self.assertEqual(group_name(0.69), "bci_blind")
        self.assertEqual(group_name(0.70), "normal")
        self.assertEqual(group_name(0.71), "normal")
        with self.assertRaises(ValueError):
            group_name(float("nan"))

    def test_each_fold_fits_only_its_training_trials(self):
        y = np.tile([0, 1], 10)
        X = np.zeros((20, 2, 200))
        X[:, 0, :] = np.arange(20)[:, None]
        seen = []

        class SpyPipeline:
            def fit(self, data, labels):
                self.ids = set(data[:, 0, 0].astype(int))
                self.assertion = len(labels)

            def predict(self, data):
                ids = data[:, 0, 0].astype(int)
                assert self.ids.isdisjoint(ids)
                assert len(self.ids) == 16
                seen.extend(ids)
                return ids % 2

        with patch("src.tools.group_openbmi_csp.sosfiltfilt", side_effect=lambda sos, x, axis: x), \
             patch("src.tools.group_openbmi_csp.make_pipeline", side_effect=lambda *a: SpyPipeline()):
            score, scores, pred, folds = score_training(X, y, 100, components=2)
        self.assertEqual(score, 1)
        self.assertEqual(sorted(seen), list(range(20)))
        np.testing.assert_array_equal(pred, y)
        np.testing.assert_array_equal(np.bincount(folds), [4] * 5)

    def test_real_pipeline_synthetic_separable_data(self):
        rng = np.random.default_rng(42)
        y = np.tile([0, 1], 20)
        X = rng.normal(size=(40, 4, 200)) * 1e-6
        X[y == 0, 0] *= 5
        X[y == 1, 1] *= 5
        score, _, _, _ = score_training(X, y, 100, components=2)
        self.assertGreater(score, 0.9)

    def test_export_preserves_data_and_never_reads_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "train.npz"
            # Invalid sibling file would fail if the grouping code tried loading it.
            (root / "test.npz").write_bytes(b"must never be read")
            X = np.random.default_rng(1).normal(size=(20, 4, 200)).astype(np.float32)
            payload = dict(X=X, y=np.tile([0, 1], 10), sfreq=100,
                           subject_id=np.ones(20, dtype=int), session_id=np.ones(20, dtype=int),
                           trial_id=np.arange(20), source_run=np.full(20, "EEG_MI_train"),
                           channel_names=np.array(["C3", "C4", "Cz", "FCz"]),
                           metadata_json=json.dumps({"mode": "official"}))
            np.savez(path, **payload)
            args = Namespace(folds=5, seed=42, components=2, low=8., high=30., threshold=.7)
            row = process_file(path, root / "groups", args)
            with np.load(root / "groups" / row["grouped_train"]) as grouped:
                np.testing.assert_array_equal(grouped["X"], X)
                np.testing.assert_array_equal(grouped["y"], payload["y"])
                self.assertTrue(np.all(grouped["group_label"] == int(row["group"] == "bci_blind")))
            payload["source_run"][0] = "EEG_MI_test"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "official training"):
                load_training(path)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.tools.group_openbmi_csp import (
    MOTOR_CHANNELS, group_name, load_training, process_subject, score_training,
)


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
            X = np.random.default_rng(1).normal(size=(20, 20, 400)).astype(np.float32)
            payload = dict(X=X, y=np.tile([0, 1], 10), sfreq=100,
                           subject_id=np.ones(20, dtype=int), session_id=np.ones(20, dtype=int),
                           trial_id=np.arange(20), source_run=np.full(20, "EEG_MI_train"),
                           channel_names=np.array(MOTOR_CHANNELS),
                           metadata_json=json.dumps({"mode": "official", "tmin": 0.0}))
            paths = []
            for session in (1, 2):
                folder = root / f"ses-{session:02d}"
                folder.mkdir()
                (folder / "test.npz").write_bytes(b"must never be read")
                path = folder / "train.npz"
                np.savez(path, **dict(payload, session_id=np.full(20, session)))
                paths.append(path)
            args = Namespace(folds=5, seed=42, components=2, low=8., high=30., threshold=.7)
            predicted = payload["y"].copy()
            with patch("src.tools.group_openbmi_csp.score_training", side_effect=[
                (.6, [.6] * 5, predicted, np.repeat(np.arange(5), 4)),
                (.8, [.8] * 5, predicted, np.repeat(np.arange(5), 4)),
            ]):
                subject, rows = process_subject(paths, root / "groups", args)
            self.assertAlmostEqual(subject["mean_cv_accuracy"], .7)
            self.assertEqual(subject["group"], "normal")
            self.assertEqual(len(rows), 2)
            for row in rows:
                with np.load(root / "groups" / row["grouped_train"]) as grouped:
                    np.testing.assert_array_equal(grouped["X"], X)
                    np.testing.assert_array_equal(grouped["y"], payload["y"])
                    self.assertTrue(np.all(grouped["group_label"] == 0))
            with self.assertRaisesRegex(ValueError, "session 1"):
                process_subject(paths[:1], root / "incomplete", args)
            payload["source_run"][0] = "EEG_MI_test"
            np.savez(paths[0], **payload)
            with self.assertRaisesRegex(ValueError, "official training"):
                load_training(paths[0])


if __name__ == "__main__":
    unittest.main()

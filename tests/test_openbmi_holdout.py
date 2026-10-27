"""Subject isolation and allocation checks; no EEG files are loaded."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.offline import run_openbmi_holdout as holdout
from src.offline import train_v65_distill as core


class HoldoutTests(unittest.TestCase):
    def test_fixed_split_and_reject_overlap(self):
        report = Path(__file__).resolve().parents[1] / "Data/OpenBMI/groups/csp70_20/report.json"
        if not report.is_file():
            self.skipTest("Local group report not present")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            split = holdout.get_split(report, path, 2026)
            self.assertEqual([len(split[k]) for k in ("train", "val", "test")], [32, 11, 11])
            self.assertEqual(split, holdout.get_split(report, path, 2026))
            self.assertFalse(set(split["train"]) & set(split["test"]))
            split["test"][0] = split["train"][0]
            path.write_text(json.dumps(split), encoding="utf-8")
            with self.assertRaises(ValueError):
                holdout.get_split(report, path, 2026)

    def test_dataset_reuses_arrays_but_does_not_mutate_input(self):
        X = np.ones((4, 2, 8), dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int64)
        feats = np.ones((4, 2), dtype=np.float32)
        scalers = [core.StandardScaler().fit(X[:, ch]) for ch in range(2)]
        ds = core.EEGDataset(X, y, feats, scalers)
        self.assertTrue(np.shares_memory(ds.X, X))
        ds[0]
        np.testing.assert_array_equal(X, np.ones_like(X))

    def test_windows_source_remapped_to_mounted_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / "OpenBMI" / "example.mat"
            expected.parent.mkdir()
            expected.touch()
            with patch.dict(core.RUN_CONFIG, {"data_root": tmp}):
                actual = core.resolve_data_path(r"D:\project\Data\OpenBMI\example.mat")
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

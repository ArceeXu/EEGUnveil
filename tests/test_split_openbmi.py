"""Run: python -m unittest discover -s tests -p test_split_openbmi.py"""
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.io import savemat

from src.tools.split_openbmi import read_run, split_file, stratified_indices


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "sess01_subj01_EEG_MI.mat"
        self.signal = np.arange(500 * 2).reshape(500, 2).astype(float)
        self.block = dict(x=self.signal, t=np.arange(8) * 50 + 1, fs=10,
                          y_dec=np.tile([1, 2], 4), chan=np.array(["C3", "C4"], dtype=object))
        self.block["class"] = np.array([["1", "right"], ["2", "left"]], dtype=object)
        savemat(self.path, dict(EEG_MI_train=self.block, EEG_MI_test=self.block))

    def test_event_alignment_labels_and_units(self):
        data = read_run(self.path, "EEG_MI_train")
        self.assertEqual(data["X"].shape, (8, 2, 40))
        np.testing.assert_allclose(data["X"][0], self.signal[1:41].T * 1e-6, rtol=1e-6)
        np.testing.assert_array_equal(data["y"], np.tile([1, 0], 4))
        self.assertEqual(read_run(self.path, "EEG_MI_train", sfreq=5)["X"].shape, (8, 2, 20))

    def test_bad_bounds_fail(self):
        self.block["t"][-1] = 490
        savemat(self.path, {"EEG_MI_train": self.block})
        with self.assertRaisesRegex(ValueError, "bounds"):
            read_run(self.path, "EEG_MI_train")

    def test_partition_reproducible_complete_balanced(self):
        y = np.tile([0, 1], 100)
        a, b = stratified_indices(y, 0.2, 42)
        c, d = stratified_indices(y, 0.2, 42)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), set(range(200)))
        np.testing.assert_array_equal(np.bincount(y[b]), [20, 20])

    def test_export_provenance_and_overwrite_protection(self):
        for mode, expected in [("official", (8, 8)), ("random", (12, 4))]:
            args = Namespace(mode=mode, test_size=0.25, seed=42,
                             tmin=0., tmax=4., sfreq=None)
            output = self.root / mode
            split_file(self.path, output, 1, 1, args)
            folder = output / "sub-01/ses-01"
            identities = []
            for name, count in zip(["train", "test"], expected):
                with np.load(folder / f"{name}.npz", allow_pickle=False) as data:
                    self.assertEqual(len(data["y"]), count)
                    identities.append(set(zip(data["source_run"], data["trial_id"])))
                    self.assertTrue(np.all(data["subject_id"] == 1))
            self.assertFalse(identities[0] & identities[1])
            self.assertEqual(len(identities[0] | identities[1]), 16)
            with self.assertRaises(FileExistsError):
                split_file(self.path, output, 1, 1, args)


if __name__ == "__main__":
    unittest.main()

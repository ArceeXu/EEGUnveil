import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.offline import train_v65_distill as training


class MixedTrainingTests(unittest.TestCase):
    def test_natural_without_distillation(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(32, 8, 128)).astype(np.float32)
        y = np.tile([0, 1], 16)
        groups = np.array([0] * 8 + [1] * 16 + [0, 0, 1, 1] * 2)
        with patch.multiple(
            training, DEVICE=torch.device("cpu"), EPOCHS=1, BATCH_SIZE=8,
            NUM_WORKERS=0, GLOBAL_STOP_ACC=2.0, GLOBAL_STOP_FLAG=False,
            GLOBAL_BEST_ACC=-1.0, GLOBAL_BEST_ARTIFACTS={}, RUN_CONFIG={},
            SAMPLING_MODE="natural", USE_DISTILLATION=False,
        ):
            with patch.object(training, "MixedGroupBatchSampler",
                              side_effect=AssertionError("Natural mode must not rebalance groups")), \
                 patch.object(training, "ShallowConvNetTeacher",
                              side_effect=AssertionError("Teacher must be disabled")):
                accuracy, _, predictions, _ = training.train_fold_pipeline(
                    X[:24], y[:24], X[24:], y[24:], 0, X.shape,
                    groups[:24], groups[24:],
                )
            self.assertEqual(len(predictions), 8)
            self.assertTrue(np.isfinite(accuracy))
            self.assertAlmostEqual(training.RUN_CONFIG["effective_mix_lambda"], 1 / 3)

    def test_batches_preserve_majority_and_classes(self):
        y = np.tile([0, 1], 50)
        groups = np.repeat([0, 1], [20, 80])
        sampler = training.MixedGroupBatchSampler(y, groups, 16, 0.75)
        for indices in sampler:
            self.assertEqual(np.sum(groups[indices] == 0), 12)
            self.assertEqual(np.sum(groups[indices] == 1), 4)
            for g in (0, 1):
                selected = np.array(indices)[groups[indices] == g]
                self.assertEqual(np.sum(y[selected] == 0), np.sum(y[selected] == 1))
        for coefficient in (0, 0.5, 1):
            with self.assertRaises(ValueError):
                training.MixedGroupBatchSampler(y, groups, 16, coefficient)

    def test_teacher_student_one_epoch(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(32, 8, 128)).astype(np.float32)
        y = np.tile([0, 1], 16)
        groups = np.tile([0, 0, 1, 1], 8)
        # Each fold's CSP/scalers are fitted through the production pipeline.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch.multiple(
                training, DEVICE=torch.device("cpu"), EPOCHS=1, BATCH_SIZE=8,
                NUM_WORKERS=0, GLOBAL_STOP_ACC=2.0, GLOBAL_STOP_FLAG=False,
                GLOBAL_BEST_ACC=-1.0, GLOBAL_BEST_ARTIFACTS={},
                ARTIFACTS_DIR=Path(tmp), RUN_CONFIG={},
            ):
                accuracy, labels, predictions, report = training.train_fold_pipeline(
                    X[:24], y[:24], X[24:], y[24:], 0, X.shape,
                    groups[:24], groups[24:],
                )
                self.assertEqual(len(predictions), 8)
                self.assertTrue(np.isfinite(accuracy))
                self.assertEqual(set(report["group_accuracy"]), {"normal", "bci_blind"})
                self.assertEqual(training.GLOBAL_BEST_ARTIFACTS["model_state"]["head.layers.7.weight"].shape[0], 2)
                training.save_global_artifacts()
                self.assertTrue((Path(tmp) / "best_model.pth").is_file())


if __name__ == "__main__":
    unittest.main()

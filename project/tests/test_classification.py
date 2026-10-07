from __future__ import annotations

import unittest

import numpy as np

from classification.features import calibrate_empty, image_features, signal_features
from classification.models import binary_metrics, grouped_folds, select_columns, size_class
from classification.predict import predict_from_features
from data_loader.bmid_loader import bmid_frequencies_hz
from reconstruction.bmid_geometry import generation_from_path, remove_low_rank, remove_rotational_mean, time_gate
from tools.build_classification_features import attach_twins
from reconstruction.das import das_coherent_from_frequency
from reconstruction.reconstruction_manager import build_circular_antenna_array
from tools.train_tumor_models import balanced_fair_mask


class TestSignalFeatures(unittest.TestCase):
    def setUp(self):
        self.freqs = bmid_frequencies_hz()
        rng = np.random.default_rng(1)
        self.raw = rng.normal(size=(self.freqs.size, 24)) + 1j * rng.normal(size=(self.freqs.size, 24))

    def test_calibrate_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            calibrate_empty(self.raw, self.raw[:, :10])

    def test_symmetric_signal_has_no_rotational_residual_or_harmonics(self):
        column = np.exp(-1j * 2 * np.pi * self.freqs * 1e-9)[:, None]
        symmetric = np.repeat(column, 24, axis=1)
        feats = signal_features(self.raw, symmetric, self.freqs)
        self.assertAlmostEqual(feats["sig_rot_residual"], 0.0, places=12)
        self.assertAlmostEqual(feats["sig_ang_h1"], 0.0, places=12)
        self.assertAlmostEqual(feats["sig_ant_cv"], 0.0, places=12)

    def test_one_bright_antenna_raises_asymmetry(self):
        cal = np.ones((self.freqs.size, 24), dtype=complex)
        cal[:, 5] *= 4.0
        feats = signal_features(self.raw, cal, self.freqs)
        self.assertGreater(feats["sig_rot_residual"], 0.05)
        self.assertGreater(feats["sig_ang_h1"], 0.01)
        self.assertGreater(feats["sig_ant_max_over_median"], 10.0)

    def test_features_are_finite(self):
        feats = signal_features(self.raw, calibrate_empty(self.raw, 0.9 * self.raw), self.freqs)
        self.assertTrue(all(np.isfinite(value) for value in feats.values()))
        self.assertEqual(sum(name.startswith("sig_band") for name in feats), 21)


class TestImageFeatures(unittest.TestCase):
    def test_top_spot_position_matches_blob(self):
        span = (-0.06, 0.06)
        axis = np.linspace(span[0], span[1], 64)
        xx, yy = np.meshgrid(axis, axis)
        image = np.exp(-(((xx - 0.02) ** 2 + (yy + 0.015) ** 2) / (2 * 0.005**2))) + 0.02
        feats = image_features(image, span, span, "t")
        self.assertAlmostEqual(feats["t_x_cm"], 2.0, delta=0.4)
        self.assertAlmostEqual(feats["t_y_cm"], -1.5, delta=0.4)
        self.assertTrue(all(key.startswith("t_") for key in feats))
        self.assertTrue(all(np.isfinite(value) for value in feats.values()))


class TestModelHelpers(unittest.TestCase):
    def test_size_class_edges(self):
        self.assertEqual(size_class(1.0), "small")
        self.assertEqual(size_class(2.0), "small")
        self.assertEqual(size_class(3.0), "medium")
        self.assertEqual(size_class(4.0), "medium")
        self.assertEqual(size_class(6.0), "large")

    def test_grouped_folds_never_split_a_phantom(self):
        groups = np.repeat(np.arange(20), 6)
        y = np.tile([0, 1], 60)
        for train, test in grouped_folds(y, groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_select_columns_by_prefix(self):
        names = ["sig_a", "d4_cal_x", "das_raw_y", "sig_b"]
        self.assertEqual(select_columns(names, ("sig_",)), [0, 3])

    def test_binary_metrics_counts(self):
        row = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.2]))
        self.assertEqual((row["tp"], row["fp"], row["tn"], row["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(row["auc"], 0.75)


class TestCoherentDasAndGeometry(unittest.TestCase):
    def test_point_target_is_found(self):
        freqs = bmid_frequencies_hz()
        positions = build_circular_antenna_array(72, 0.1, span_deg=355.0)
        target = np.array([0.02, -0.015])
        delay = 2.0 * np.linalg.norm(positions - target, axis=1) / 3e8
        s = np.exp(-2j * np.pi * freqs[:, None] * delay[None, :])
        grid = np.linspace(-0.06, 0.06, 64)
        image = das_coherent_from_frequency(s, freqs, positions, grid, grid)
        iy, ix = np.unravel_index(np.argmax(image), image.shape)
        self.assertAlmostEqual(grid[ix], 0.02, delta=0.004)
        self.assertAlmostEqual(grid[iy], -0.015, delta=0.004)

    def test_rotational_mean_removes_symmetric_part(self):
        data = np.repeat(np.arange(5, dtype=complex)[:, None], 8, axis=1)
        data[:, 3] += 1.0
        out = remove_rotational_mean(data)
        np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)
        self.assertEqual(int(np.argmax(np.abs(out[0]))), 3)

    def test_low_rank_removal_drops_shared_pattern(self):
        rng = np.random.default_rng(0)
        shared = np.outer(rng.normal(size=50), np.ones(8)).astype(complex)
        out = remove_low_rank(shared, 1)
        self.assertLess(np.linalg.norm(out), 1e-9 * np.linalg.norm(shared))

    def test_time_gate_removes_early_echo(self):
        freqs = bmid_frequencies_hz()
        early = np.exp(-2j * np.pi * freqs * 0.3e-9)[:, None]
        late = np.exp(-2j * np.pi * freqs * 2.0e-9)[:, None]
        gated = time_gate(early + late, freqs, 1.0e-9)
        self.assertLess(np.abs(np.vdot(early, gated)) / np.abs(np.vdot(late, gated)), 0.05)

    def test_twin_baselines_are_same_phantom_session_healthy(self):
        records = [
            {"index": i, "scan_id": i + 1, "phant_id": "P", "n_session": 1.0, "group": g}
            for i, g in enumerate(["tumor", "healthy_fib", "healthy_fib", "healthy_fib", "healthy_fib", "adipose_only"])
        ] + [{"index": 6, "scan_id": 7, "phant_id": "Q", "n_session": 1.0, "group": "healthy_fib"}]
        kept = attach_twins(records, k=3)
        self.assertEqual([r["index"] for r in kept], [0, 1, 2, 3, 4])
        for record in kept:
            self.assertNotIn(record["index"], record["twin_indices"])
            self.assertTrue(set(record["twin_indices"]) <= {1, 2, 3, 4})
            self.assertEqual(len(record["twin_indices"]), 3)

    def test_generation_from_path(self):
        self.assertEqual(generation_from_path("x/fd_data_gen_two_s11.mat"), "gen2")
        self.assertEqual(generation_from_path("metadata_gen_three.mat"), "gen3")
        self.assertIsNone(generation_from_path("scan.s1p"))

    def test_signal_prefix(self):
        freqs = bmid_frequencies_hz()
        cal = np.ones((freqs.size, 8), dtype=complex)
        feats = signal_features(cal, cal, freqs, prefix="rsig")
        self.assertTrue(all(name.startswith("rsig_") for name in feats))


class TestFairSet(unittest.TestCase):
    def test_sessions_are_balanced(self):
        table = {
            "has_tumor": np.array([1, 1, 1, 0, 1, 0, 0, 0, 1]),
            "session": np.array(["a", "a", "a", "a", "b", "b", "b", "b", "c"]),
        }
        keep = balanced_fair_mask(table)
        for session in "abc":
            sel = keep & (table["session"] == session)
            self.assertEqual(int(table["has_tumor"][sel].sum()), int((table["has_tumor"][sel] == 0).sum()))
        self.assertEqual(int(keep.sum()), 4)

    def test_predict_from_features(self):
        from sklearn.dummy import DummyClassifier, DummyRegressor

        X, y = np.zeros((4, 1)), np.array([0, 1, 1, 1])
        bundle = {
            "feature_names": ["f"],
            "size_classes": ("small", "medium", "large"),
            "models": {
                "detection": {"columns": [0], "estimator": DummyClassifier(strategy="prior").fit(X, y), "threshold": 0.8},
                "size_class": {"columns": [0], "estimator": DummyClassifier(strategy="constant", constant=1).fit(X, y)},
                "diameter": {"columns": [0], "estimator": [DummyRegressor(constant=2.5, strategy="constant").fit(X, y)]},
            },
        }
        feats = {"f": 0.0, "imgl_peak_x_cm": 0.5, "imgl_peak_y_cm": -1.0}
        out = predict_from_features(feats, bundle, "gen3")
        self.assertAlmostEqual(out["tumor_probability"], 0.75)
        self.assertFalse(out["tumor_flag"])
        self.assertEqual(out["size_class"], "medium")
        self.assertEqual(out["diameter_cm"], 2.5)
        self.assertEqual(out["image_spot_cm"], (0.5, -1.0))
        self.assertTrue(out["image_spot_reliable"])
        self.assertFalse(predict_from_features(feats, bundle, "gen2")["image_spot_reliable"])


if __name__ == "__main__":
    unittest.main()

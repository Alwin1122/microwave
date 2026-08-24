"""Tests for beamformer selection modes."""

from __future__ import annotations

import unittest

import numpy as np

from quality.beamformer_selector import select_best_beamformer


class TestBeamformerSelector(unittest.TestCase):
    def test_prefer_dmas_d4_mode(self):
        images = {
            "DAS": np.ones((16, 16)) * 3.0,
            "DMAS": np.ones((16, 16)) * 2.0,
            "DMAS-D4": np.ones((16, 16)) * 0.5,
        }
        timings = {"DAS": 0.01, "DMAS": 0.02, "DMAS-D4": 0.03}
        selected, metrics = select_best_beamformer(images, timings, mode="prefer_dmas_d4")
        self.assertEqual(selected, "DMAS-D4")
        self.assertEqual(metrics["DMAS-D4"].get("selection_mode"), "prefer_dmas_d4")

    def test_tumor_gt_mode_picks_closer_roi(self):
        das = np.zeros((32, 32), dtype=float)
        das[15:17, 15:17] = 5.0  # center
        dmas_d4 = np.zeros((32, 32), dtype=float)
        dmas_d4[22:26, 22:26] = 4.0  # near GT quadrant
        images = {"DAS": das, "DMAS-D4": dmas_d4}
        timings = {"DAS": 0.01, "DMAS-D4": 0.02}
        selected, metrics = select_best_beamformer(
            images,
            timings,
            mode="tumor_gt",
            x_span=(-0.06, 0.06),
            y_span=(-0.06, 0.06),
            tumor_xy_m=(0.0225, 0.0225),
            prefer_off_center=True,
        )
        self.assertEqual(selected, "DMAS-D4")
        self.assertIn("tumor_gt_distance_m", metrics["DMAS-D4"])
        self.assertLess(
            metrics["DMAS-D4"]["tumor_gt_distance_m"],
            metrics["DAS"]["tumor_gt_distance_m"],
        )


if __name__ == "__main__":
    unittest.main()

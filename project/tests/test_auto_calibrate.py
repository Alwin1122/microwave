"""Tests for automatic reconstruction calibration."""

from __future__ import annotations

import unittest

import numpy as np

from reconstruction.auto_calibrate import (
    GeometryCandidate,
    auto_calibrate_reconstruction,
    build_geometry_search_space,
)
from reconstruction.reconstruction_manager import (
    ReconstructionConfig,
    build_circular_antenna_array,
)


def _point_target(freqs, ant_pos, target_xy, wave_speed=3e8):
    target = np.asarray(target_xy, dtype=float)
    distances = np.linalg.norm(ant_pos - target[None, :], axis=1)
    round_trip = 2.0 * distances / wave_speed
    phase = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * round_trip[None, :])
    envelope = np.exp(-((freqs[:, None] - 4.5e9) ** 2) / (2.0 * (1.0e9) ** 2))
    return envelope * phase


class TestAutoCalibrate(unittest.TestCase):
    def test_quick_search_space_is_smaller(self):
        base = ReconstructionConfig()
        quick = build_geometry_search_space(base, quick=True)
        full = build_geometry_search_space(base, quick=False)
        self.assertLess(len(quick), len(full))
        self.assertGreater(len(quick), 10)

    def test_auto_calibrate_finds_near_target(self):
        freqs = np.linspace(1e9, 8e9, 48)
        ant_pos = build_circular_antenna_array(12, radius=0.08)
        target = (0.02, 0.015)
        data = _point_target(freqs, ant_pos, target)
        config = ReconstructionConfig(
            x_span=(-0.05, 0.05),
            y_span=(-0.05, 0.05),
            n_x=32,
            n_y=32,
            antenna_radius=0.08,
            wave_speed=3e8,
            antenna_angle_offset_deg=90.0,  # intentionally wrong starting guess
            antenna_flip_x=True,
        )
        result = auto_calibrate_reconstruction(
            data,
            freqs,
            config,
            tumor_xy_m=target,
            quick=True,
            top_k_full=3,
        )
        self.assertIsNotNone(result.best.tumor_gt_distance_m)
        self.assertLess(result.best.tumor_gt_distance_m, 0.025)
        # Current settings are included; final pick should beat a random far ROI.
        self.assertGreater(len(result.trials), 5)

    def test_geometry_candidate_label(self):
        g = GeometryCandidate(antenna_angle_offset_deg=180.0, antenna_flip_y=True)
        self.assertIn("180", g.label())
        self.assertIn("flip=y", g.label())

    def test_keeps_baseline_when_no_improvement(self):
        freqs = np.linspace(1e9, 8e9, 32)
        ant_pos = build_circular_antenna_array(8, radius=0.08)
        target = (0.01, 0.01)
        data = _point_target(freqs, ant_pos, target)
        config = ReconstructionConfig(
            x_span=(-0.04, 0.04),
            y_span=(-0.04, 0.04),
            n_x=24,
            n_y=24,
            antenna_radius=0.08,
            wave_speed=3e8,
        )
        result = auto_calibrate_reconstruction(
            data,
            freqs,
            config,
            tumor_xy_m=target,
            quick=True,
            top_k_full=2,
            baseline_prefer_off_center=False,
        )
        self.assertIsNotNone(result.baseline)
        if not result.improved:
            self.assertEqual(result.best.beamformer, result.baseline.beamformer)
            self.assertEqual(result.best.geometry, result.baseline.geometry)
        else:
            self.assertLess(
                result.best.tumor_gt_distance_m,
                result.baseline.tumor_gt_distance_m - 0.0005,
            )


if __name__ == "__main__":
    unittest.main()

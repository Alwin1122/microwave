"""Detect → locate pipeline: localization path + gating behaviour."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from automation.detect_locate import DetectLocateResult, detect_and_locate, localize_blind
from data_loader.dataset_info import MicrowaveDataset


def _synthetic_dataset(n_freq: int = 64, n_ant: int = 12) -> MicrowaveDataset:
    freqs = np.linspace(1e9, 8e9, n_freq)
    # Mild frequency ripple so Week 3 / DAS have something to image.
    s = np.exp(1j * np.linspace(0, 2 * np.pi, n_freq)).reshape(-1, 1) * np.ones(n_ant)
    s = s + 0.05 * np.random.default_rng(0).normal(size=s.shape)
    return MicrowaveDataset(
        file_path="synthetic.mat",
        file_name="synthetic.mat",
        file_type="synthetic",
        frequencies=freqs,
        s_parameters=s.astype(complex),
        n_ports=n_ant,
        available_variables=["fd"],
        metadata={
            "dataset_family": "UM-BMID",
            "bmid_generation": "gen1",
            "antenna_radius_m": 0.18,
            "wave_speed_m_per_s": 2.1e8,
            "reconstruction_x_span_m": (-0.06, 0.06),
            "reconstruction_y_span_m": (-0.06, 0.06),
        },
        raw={},
    )


class TestDetectLocate(unittest.TestCase):
    def test_localize_blind_returns_cm_point(self):
        loc = localize_blind(_synthetic_dataset())
        self.assertIn("location_cm", loc)
        self.assertEqual(len(loc["location_cm"]), 2)
        self.assertTrue(np.isfinite(loc["location_cm"][0]))
        self.assertGreater(loc["prominence"], 0.0)

    def test_skips_localization_when_healthy(self):
        dataset = _synthetic_dataset()
        ref = np.asarray(dataset.s_parameters, dtype=complex)

        fake_detection = {
            "tumor_probability": 0.12,
            "tumor_flag": False,
            "image_spot_cm": (0.0, 0.0),
        }

        with patch("automation.detect_locate.predict_scan", return_value=fake_detection), patch(
            "automation.detect_locate.load_models", return_value={"models": {}, "feature_names": []}
        ):
            result = detect_and_locate(dataset, reference=ref, reference_kind="emp")

        self.assertIsInstance(result, DetectLocateResult)
        self.assertFalse(result.tumor_detected)
        self.assertFalse(result.localization_ran)
        self.assertIsNone(result.location_cm)

    def test_localizes_when_tumor_flagged(self):
        dataset = _synthetic_dataset()
        ref = np.asarray(dataset.s_parameters, dtype=complex)
        fake_detection = {
            "tumor_probability": 0.91,
            "tumor_flag": True,
            "image_spot_cm": (1.0, 1.0),
        }

        with patch("automation.detect_locate.predict_scan", return_value=fake_detection), patch(
            "automation.detect_locate.load_models", return_value={"models": {}, "feature_names": []}
        ):
            result = detect_and_locate(dataset, reference=ref, reference_kind="emp")

        self.assertTrue(result.tumor_detected)
        self.assertTrue(result.localization_ran)
        self.assertIsNotNone(result.location_cm)
        self.assertEqual(len(result.location_cm), 2)


if __name__ == "__main__":
    unittest.main()

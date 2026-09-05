"""Unit tests for Module 2 Week 3 time-domain path and Module 3 handoff export."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.touchstone_loader import load_touchstone_dataset
from preprocessing.handover_export import export_module3_handover
from preprocessing.preprocessing_pipeline import PreprocessingConfig, run_preprocessing_pipeline
from preprocessing.week3_time_domain import (
    apply_hamming,
    build_time_axis,
    group_mean_clutter_removal,
    global_normalize,
    hamming_window,
    process_week3_time_domain,
)


class TestWeek3Core(unittest.TestCase):
    def test_hamming_matches_brief_endpoints(self):
        w = hamming_window(5)
        self.assertAlmostEqual(w[0], 0.08, places=5)
        self.assertAlmostEqual(w[-1], 0.08, places=5)
        self.assertAlmostEqual(w[2], 1.0, places=5)

    def test_time_axis_delta_t(self):
        freqs = np.linspace(1e9, 3e9, 11)
        time_s, delta_t, bandwidth, period = build_time_axis(freqs)
        delta_f = float(np.mean(np.diff(freqs)))
        self.assertAlmostEqual(delta_t, 1.0 / (11 * delta_f), places=12)
        self.assertAlmostEqual(bandwidth, 2e9, places=6)
        self.assertAlmostEqual(period, 1.0 / delta_f, places=12)
        self.assertEqual(time_s.shape[0], 11)

    def test_matched_reference_subtract_linearity(self):
        freqs = np.linspace(1e9, 2e9, 16)
        target = np.exp(1j * np.linspace(0, 1, 16))
        reference = 0.25 * np.ones(16, dtype=complex)
        result = process_week3_time_domain(freqs, target, reference_filtered=reference)

        self.assertTrue(result.matched_windowing_performed)
        self.assertTrue(result.time_reference_subtraction_performed)
        expected_freq = apply_hamming(target) - apply_hamming(reference)
        np.testing.assert_allclose(result.s21_reference_subtracted_freq[:, 0], expected_freq)

        # IFFT(ST - SR) == IFFT(ST) - IFFT(SR)
        np.testing.assert_allclose(
            result.time_reference_subtracted[:, 0],
            result.time_hamming[:, 0] - result.time_reference_hamming[:, 0],
            atol=1e-12,
        )

    def test_group_clutter_and_global_normalize(self):
        # (n_time, n_channels)
        signals = np.array(
            [
                [1 + 0j, 1 + 0j, 1 + 0j],
                [2 + 0j, 3 + 0j, 4 + 0j],
            ],
            dtype=complex,
        )
        cleaned, mu = group_mean_clutter_removal(signals, axis=1)
        np.testing.assert_allclose(mu, np.array([1 + 0j, 3 + 0j]))
        np.testing.assert_allclose(cleaned[:, 0], np.array([0 + 0j, -1 + 0j]))

        normalized, alpha = global_normalize(cleaned)
        self.assertAlmostEqual(alpha, 1.0)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 1.0)

    def test_single_channel_skips_clutter(self):
        freqs = np.linspace(1e9, 2e9, 8)
        s21 = np.ones(8, dtype=complex)
        result = process_week3_time_domain(freqs, s21, enable_clutter_removal=True)
        self.assertFalse(result.clutter_removal_performed)
        self.assertTrue(any("clutter" in note.lower() for note in result.notes))


class TestWeek3PipelineAndExport(unittest.TestCase):
    def _write_touchstone(self, name: str, body: str) -> str:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        return path

    def test_touchstone_week3_runs_and_exports(self):
        primary = self._write_touchstone(
            "_test_week3_primary.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.0 0 0 1.0 0.0 0 0 0 0",
                    "1.1 0 0 1.0 0.0 0 0 0 0",
                    "1.2 0 0 1.0 0.0 0 0 0 0",
                    "1.3 0 0 1.0 0.0 0 0 0 0",
                ]
            ),
        )
        reference = self._write_touchstone(
            "_test_week3_reference.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.0 0 0 0.2 0.0 0 0 0 0",
                    "1.1 0 0 0.2 0.0 0 0 0 0",
                    "1.2 0 0 0.2 0.0 0 0 0 0",
                    "1.3 0 0 0.2 0.0 0 0 0 0",
                ]
            ),
        )
        try:
            dataset = load_touchstone_dataset(primary)
            ref = load_touchstone_dataset(reference)
            result = run_preprocessing_pipeline(
                dataset,
                PreprocessingConfig(
                    filter_method="none",
                    enable_week3=True,
                    reference_dataset=ref,
                ),
            )
            self.assertIsNotNone(result.week3_result)
            self.assertTrue(result.week3_result.time_reference_subtraction_performed)
            self.assertTrue(result.validation_report.week3_time_domain_performed)
            self.assertEqual(result.week3_result.time_s.shape[0], 4)

            with tempfile.TemporaryDirectory() as tmp:
                mat_path = os.path.join(tmp, "handoff.mat")
                csv_path = os.path.join(tmp, "handoff.csv")
                written = export_module3_handover(result, mat_path, csv_path=csv_path)
                self.assertTrue(os.path.isfile(written))
                self.assertTrue(os.path.isfile(csv_path))
                self.assertTrue(os.path.isfile(os.path.splitext(mat_path)[0] + ".json"))
                payload = loadmat(mat_path)
                self.assertIn("frequency_Hz", payload)
                self.assertIn("time_s", payload)
                self.assertIn("S21_Hamming", payload)
                self.assertIn("Tx_coordinates", payload)
                self.assertIn("Rx_coordinates", payload)
        finally:
            os.remove(primary)
            os.remove(reference)


if __name__ == "__main__":
    unittest.main()

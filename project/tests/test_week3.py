"""Unit tests for Module 2 Week 3 time-domain path and Module 3 handoff export."""

from __future__ import annotations

import json
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
    circular_tx_rx_coordinates,
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

    def test_time_domain_matches_brief_formulas(self):
        n = 32
        freqs = 2e9 + np.arange(n) * 50e6
        rng = np.random.default_rng(0)
        s21 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        result = process_week3_time_domain(freqs, s21, enable_global_normalize=False)

        k = np.arange(n)
        manual = np.array([np.sum(s21 * np.exp(2j * np.pi * k * m / n)) / n for m in range(n)])
        np.testing.assert_allclose(result.time_filtered[:, 0], manual, atol=1e-12)
        np.testing.assert_allclose(
            result.time_hamming[:, 0],
            np.array([np.sum(s21 * hamming_window(n) * np.exp(2j * np.pi * k * m / n)) / n for m in range(n)]),
            atol=1e-12,
        )
        self.assertAlmostEqual(result.delta_t_s, 1.0 / (n * 50e6))
        np.testing.assert_allclose(result.time_s, np.arange(n) * result.delta_t_s)
        self.assertAlmostEqual(result.period_s, 1.0 / 50e6)
        self.assertAlmostEqual(result.bandwidth_hz, (n - 1) * 50e6, places=3)
        self.assertAlmostEqual(result.time_resolution_s, 1.0 / ((n - 1) * 50e6))
        self.assertIn("Week3 δt≈1/B (s)", result.to_display_dict())

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

    def test_clutter_removed_per_comparable_group_only(self):
        rng = np.random.default_rng(1)
        n_time = 16
        clutter_a = rng.standard_normal(n_time) + 1j * rng.standard_normal(n_time)
        clutter_b = 5 * (rng.standard_normal(n_time) + 1j * rng.standard_normal(n_time))
        signals = np.column_stack([clutter_a] * 3 + [clutter_b] * 2 + [clutter_a]).astype(complex)
        signals[4, 1] += 1.0  # target response on channel 1
        cleaned, mu = group_mean_clutter_removal(signals, axis=1, groups=[[0, 1, 2], [3, 4], [5]])

        self.assertEqual(mu.shape, (n_time, 3))
        np.testing.assert_allclose(mu[:, 0], np.mean(signals[:, :3], axis=1))
        np.testing.assert_allclose(cleaned[:, 3:5], 0, atol=1e-12)
        np.testing.assert_allclose(cleaned[:, 5], signals[:, 5])  # lone channel untouched
        self.assertAlmostEqual(abs(cleaned[4, 1]), 2 / 3, places=12)
        other_samples = np.delete(np.abs(cleaned[:, 1]), 4)
        np.testing.assert_allclose(other_samples, 0, atol=1e-12)

    def test_comparable_groups_from_geometry(self):
        from preprocessing.week3_time_domain import comparable_channel_groups

        tx, rx = circular_tx_rx_coordinates(8, 0.08)
        self.assertEqual(comparable_channel_groups(tx, rx), [list(range(8))])

        angles = np.deg2rad(np.arange(8) * 45.0)
        ring = 0.08 * np.column_stack((np.cos(angles), np.sin(angles)))
        tx_m = np.vstack([ring[:4], ring[:4]])
        rx_m = np.vstack([np.roll(ring, -1, axis=0)[:4], np.roll(ring, -2, axis=0)[:4]])
        self.assertEqual(comparable_channel_groups(tx_m, rx_m), [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_global_normalize_uses_one_alpha_across_channels(self):
        freqs = np.linspace(1e9, 2e9, 16)
        rng = np.random.default_rng(5)
        s21 = (rng.standard_normal((16, 3)) + 1j * rng.standard_normal((16, 3))) * np.array([1.0, 0.5, 0.1])
        result = process_week3_time_domain(freqs, s21, enable_clutter_removal=False)
        original = result.time_clutter_removed
        alpha = np.max(np.abs(original))
        self.assertAlmostEqual(result.global_scale_alpha, alpha)
        np.testing.assert_allclose(result.time_global_normalized, original / alpha)
        self.assertAlmostEqual(float(np.max(np.abs(result.time_global_normalized))), 1.0)
        peaks_before = np.max(np.abs(original), axis=0)
        peaks_after = np.max(np.abs(result.time_global_normalized), axis=0)
        np.testing.assert_allclose(peaks_after / peaks_after[0], peaks_before / peaks_before[0])

    def test_pipeline_default_normalization_is_global(self):
        from preprocessing.normalization import apply_normalization

        s = np.column_stack([np.full(8, 2.0 + 0j), np.full(8, 0.5 + 0j)])
        out = apply_normalization(s)
        np.testing.assert_allclose(out[:, 0], 1.0)
        np.testing.assert_allclose(out[:, 1], 0.25)
        self.assertEqual(PreprocessingConfig().normalization_method, "global_max")

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
                self.assertEqual(payload["time_filtered"].shape, payload["time_Hamming"].shape)
                self.assertTrue(bool(payload["validation_passed"]))
                with open(os.path.splitext(mat_path)[0] + ".json", encoding="utf-8") as handle:
                    report = json.load(handle)
                self.assertTrue(report["validation_passed"])
                self.assertTrue(report["code_version"])
                self.assertEqual(report["processing_parameters"]["reference_file"], reference)
                self.assertGreaterEqual(len(report["files"]["plots"]), 3)
                for plot in report["files"]["plots"]:
                    self.assertTrue(os.path.isfile(plot))
                checks = {c["check"] for c in report["validation_checks"]}
                self.assertIn("frequency uniformly spaced", checks)
                self.assertIn("time axis uniform with Δt = 1/(NΔf)", checks)
                self.assertIn("S21_Hamming no NaN/Inf", checks)
                self.assertIn("Tx_coordinates", payload)
                self.assertIn("Rx_coordinates", payload)
        finally:
            os.remove(primary)
            os.remove(reference)

    def test_handover_validation_blocks_bad_payloads(self):
        from preprocessing.handover_export import build_handover_payload, validate_handover_payload
        from utils.exceptions import DatasetValidationError

        primary = self._write_touchstone(
            "_test_validate_primary.s2p",
            "\n".join(["# GHz S RI R 50"] + [f"{1.0 + 0.1 * k:.1f} 0 0 1.0 0.5 0 0 0 0" for k in range(8)]),
        )
        try:
            result = run_preprocessing_pipeline(
                load_touchstone_dataset(primary),
                PreprocessingConfig(filter_method="none", enable_week3=True),
            )
            good = build_handover_payload(result)
            self.assertTrue(all(c["passed"] for c in validate_handover_payload(good)))

            def failed(payload):
                return {c["check"] for c in validate_handover_payload(payload) if not c["passed"]}

            bad = dict(good, S21_filtered=good["S21_filtered"].copy())
            bad["S21_filtered"][3, 0] = np.nan
            self.assertIn("S21_filtered no NaN/Inf", failed(bad))

            self.assertIn("S21_Hamming complex", failed(dict(good, S21_Hamming=np.abs(good["S21_Hamming"]))))
            self.assertIn("S21_Hamming dimensions", failed(dict(good, S21_Hamming=good["S21_Hamming"][:-1])))

            freqs = good["frequency_Hz"].copy()
            freqs[4] += 1e6
            self.assertIn("frequency uniformly spaced", failed(dict(good, frequency_Hz=freqs)))

            two_rx = np.zeros((2, 2))
            self.assertIn(
                "Tx/Rx coordinates match channel count", failed(dict(good, Rx_coordinates=two_rx))
            )

            with tempfile.TemporaryDirectory() as tmp:
                result.week3_result.time_hamming[2, 0] = np.inf
                mat_path = os.path.join(tmp, "bad.mat")
                with self.assertRaises(DatasetValidationError):
                    export_module3_handover(result, mat_path)
                self.assertFalse(os.path.exists(mat_path))
        finally:
            os.remove(primary)

    def test_target_and_reference_windowed_identically(self):
        freqs_ghz = np.linspace(1.0, 2.0, 41)
        rng = np.random.default_rng(4)

        def write(name, level, spike_at=None):
            real = level + 0.005 * rng.standard_normal(freqs_ghz.size)
            if spike_at is not None:
                real[spike_at] += 1.0
            lines = ["# GHz S RI R 50"] + [
                f"{f:.4f} 0 0 {r:.6f} 0.1 0 0 0 0" for f, r in zip(freqs_ghz, real)
            ]
            return self._write_touchstone(name, "\n".join(lines))

        primary = write("_test_match_target.s2p", 1.0)
        reference = write("_test_match_reference.s2p", 0.8, spike_at=20)
        try:
            result = run_preprocessing_pipeline(
                load_touchstone_dataset(primary),
                PreprocessingConfig(
                    filter_method="moving_average",
                    filter_kwargs={"window_size": 3},
                    enable_week3=True,
                    reference_dataset=load_touchstone_dataset(reference),
                ),
            )
            week3 = result.week3_result
            self.assertTrue(week3.matched_windowing_performed)
            w = hamming_window(freqs_ghz.size)
            np.testing.assert_allclose(week3.s21_hamming[:, 0], week3.s21_filtered[:, 0] * w)
            ref_h = week3.s21_reference_hamming[:, 0]
            ref_f = ref_h / w
            # Glitch at index 20 was corrected before filtering/windowing.
            self.assertLess(abs(ref_f[20].real - 0.8), 0.05)
            np.testing.assert_allclose(
                week3.s21_reference_subtracted_freq[:, 0], week3.s21_hamming[:, 0] - ref_h
            )

            with tempfile.TemporaryDirectory() as tmp:
                mat_path = os.path.join(tmp, "handoff.mat")
                export_module3_handover(result, mat_path)
                payload = loadmat(mat_path)
                np.testing.assert_allclose(payload["S21_reference_Hamming"][:, 0], ref_h)
        finally:
            os.remove(primary)
            os.remove(reference)


if __name__ == "__main__":
    unittest.main()

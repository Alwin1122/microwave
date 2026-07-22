"""
tests/test_preprocessing.py

Purpose:
    Unit tests for Module 2 (Signal Preprocessing): filtering,
    calibration, normalization, artifact suppression, and the full
    pipeline orchestration + signal quality evaluation.

Input:
    Synthetic complex S-parameter arrays generated in-line (no file I/O
    required for most tests).

Output:
    Standard unittest pass/fail results.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.dataset_info import MicrowaveDataset
from preprocessing.artifact_suppression import (
    apply_artifact_suppression,
    background_subtraction,
    hybrid_artifact_suppression,
    svd_clutter_removal,
)
from preprocessing.calibration import apply_calibration, reference_calibration, self_calibration
from preprocessing.filtering import (
    apply_noise_filter,
    butterworth_lowpass_filter,
    moving_average_filter,
    savitzky_golay_filter,
)
from preprocessing.normalization import (
    apply_normalization,
    max_magnitude_normalization,
    minmax_normalization,
    zscore_normalization,
)
from preprocessing.preprocessing_pipeline import (
    PreprocessingConfig,
    compute_signal_statistics,
    estimate_snr_db,
    run_preprocessing_pipeline,
)
from utils.exceptions import InvalidSParameterError


def _make_synthetic_sparams(n_freq=100, n_traces=6, seed=0):
    rng = np.random.default_rng(seed)
    freqs = np.linspace(1e9, 9e9, n_freq)
    base = 0.5 * np.exp(-((freqs - 5e9) ** 2) / (2 * (2e9) ** 2))
    traces = np.zeros((n_freq, n_traces), dtype=complex)
    for i in range(n_traces):
        mag = base * (1 + 0.05 * i)
        phase = -freqs / 1e9 + 0.2 * i
        noise = rng.normal(scale=0.02, size=n_freq) + 1j * rng.normal(scale=0.02, size=n_freq)
        traces[:, i] = mag * np.exp(1j * phase) + noise
    return freqs, traces


class TestFiltering(unittest.TestCase):
    def setUp(self):
        self.freqs, self.s = _make_synthetic_sparams()

    def test_moving_average_preserves_shape(self):
        out = moving_average_filter(self.s, window_size=5)
        self.assertEqual(out.shape, self.s.shape)

    def test_moving_average_reduces_variance(self):
        out = moving_average_filter(self.s, window_size=9)
        self.assertLess(np.var(np.diff(out.real, axis=0)), np.var(np.diff(self.s.real, axis=0)))

    def test_savgol_preserves_shape(self):
        out = savitzky_golay_filter(self.s, window_length=9, polyorder=3)
        self.assertEqual(out.shape, self.s.shape)

    def test_butterworth_preserves_shape(self):
        out = butterworth_lowpass_filter(self.s, cutoff=0.3, order=4)
        self.assertEqual(out.shape, self.s.shape)

    def test_empty_array_raises(self):
        with self.assertRaises(InvalidSParameterError):
            moving_average_filter(np.empty((0, 4), dtype=complex))

    def test_dispatcher(self):
        out = apply_noise_filter(self.s, method="savgol")
        self.assertEqual(out.shape, self.s.shape)
        with self.assertRaises(ValueError):
            apply_noise_filter(self.s, method="not_a_real_method")


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.freqs, self.s = _make_synthetic_sparams()

    def test_self_calibration_shape(self):
        out = self_calibration(self.s)
        self.assertEqual(out.shape, self.s.shape)

    def test_reference_calibration(self):
        reference = np.mean(self.s, axis=1)
        out = reference_calibration(self.s, reference)
        self.assertEqual(out.shape, self.s.shape)

    def test_reference_length_mismatch_raises(self):
        with self.assertRaises(InvalidSParameterError):
            reference_calibration(self.s, np.ones(10, dtype=complex))

    def test_dispatcher_requires_reference(self):
        with self.assertRaises(ValueError):
            apply_calibration(self.s, method="reference", reference=None)


class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.freqs, self.s = _make_synthetic_sparams()

    def test_minmax_range(self):
        out = minmax_normalization(self.s)
        mag = np.abs(out)
        self.assertTrue(np.all(mag >= -1e-9))
        self.assertTrue(np.all(mag <= 1 + 1e-9))

    def test_max_magnitude_normalization(self):
        out = max_magnitude_normalization(self.s)
        peak = np.max(np.abs(out), axis=0)
        np.testing.assert_allclose(peak, np.ones_like(peak), atol=1e-6)

    def test_zscore_normalization_shape(self):
        out = zscore_normalization(self.s)
        self.assertEqual(out.shape, self.s.shape)

    def test_dispatcher_unknown_method(self):
        with self.assertRaises(ValueError):
            apply_normalization(self.s, method="bogus")


class TestArtifactSuppression(unittest.TestCase):
    def setUp(self):
        self.freqs, self.s = _make_synthetic_sparams(n_traces=8)

    def test_background_subtraction_reduces_common_mode(self):
        out = background_subtraction(self.s)
        # After subtracting the mean trace, the mean across traces should
        # be close to zero.
        residual_mean = np.mean(np.abs(np.mean(out, axis=1)))
        original_mean = np.mean(np.abs(np.mean(self.s, axis=1)))
        self.assertLess(residual_mean, original_mean)

    def test_svd_clutter_removal_shape(self):
        out = svd_clutter_removal(self.s, n_components_removed=1)
        self.assertEqual(out.shape, self.s.shape)

    def test_hybrid_reduces_energy(self):
        out = hybrid_artifact_suppression(self.s)
        self.assertLessEqual(np.mean(np.abs(out) ** 2), np.mean(np.abs(self.s) ** 2) * 1.5)

    def test_dispatcher(self):
        out = apply_artifact_suppression(self.s, method="hybrid")
        self.assertEqual(out.shape, self.s.shape)
        with self.assertRaises(ValueError):
            apply_artifact_suppression(self.s, method="bogus")


class TestSignalQuality(unittest.TestCase):
    def setUp(self):
        self.freqs, self.s = _make_synthetic_sparams()

    def test_snr_is_finite(self):
        snr = estimate_snr_db(self.s)
        self.assertTrue(np.isfinite(snr))

    def test_statistics_keys(self):
        stats = compute_signal_statistics(self.s)
        for key in ("mean", "std", "min", "max", "dynamic_range_db", "snr_db"):
            self.assertIn(key, stats)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.freqs, self.s = _make_synthetic_sparams(n_traces=8)
        self.dataset = MicrowaveDataset(
            file_path="synthetic.mat",
            file_name="synthetic.mat",
            file_type="MATLAB (synthetic)",
            frequencies=self.freqs,
            s_parameters=self.s,
            n_ports=1,
            available_variables=["frequency", "s_parameters"],
        )

    def test_full_pipeline_runs(self):
        config = PreprocessingConfig()
        result = run_preprocessing_pipeline(self.dataset, config)
        self.assertEqual(result.processed_dataset.s_parameters.shape, self.s.shape)
        self.assertIn("Artifact Suppression", result.stage_outputs)
        self.assertTrue(np.isfinite(result.quality_report.snr_after_db))

    def test_progress_callback_invoked(self):
        calls = []

        def cb(pct, msg):
            calls.append((pct, msg))

        run_preprocessing_pipeline(self.dataset, PreprocessingConfig(), progress_callback=cb)
        self.assertGreater(len(calls), 0)
        self.assertEqual(calls[-1][0], 100)

    def test_pipeline_with_all_methods_disabled(self):
        config = PreprocessingConfig(
            filter_method="none",
            calibration_method="none",
            normalization_method="none",
            do_background_subtraction=False,
            artifact_method="none",
        )
        result = run_preprocessing_pipeline(self.dataset, config)
        np.testing.assert_array_equal(result.processed_dataset.s_parameters, self.s)


if __name__ == "__main__":
    unittest.main()

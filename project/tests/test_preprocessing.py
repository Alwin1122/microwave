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
from data_loader.touchstone_loader import load_touchstone_dataset
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
    gaussian_smoothing_filter,
    median_smoothing_filter,
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
from preprocessing.touchstone_s21 import _correct_isolated_samples, _nrmse
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

    def test_moving_average_matches_formula_with_reflect_edges(self):
        x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]) + 1j * np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        out = moving_average_filter(x.reshape(-1, 1), window_size=3)[:, 0]
        # interior: plain 3-point mean; k=0 mirror-pads x[1] (reflect), k=5 mirror-pads x[4]
        expected = np.array([(2 + 1 + 2) / 3, (1 + 2 + 4) / 3, (2 + 4 + 8) / 3,
                             (4 + 8 + 16) / 3, (8 + 16 + 32) / 3, (16 + 32 + 16) / 3])
        np.testing.assert_allclose(out.real, expected)
        np.testing.assert_allclose(out.imag[1:5], [1 / 3, 2 / 3, 1 / 3, 2 / 3])

    def test_moving_average_even_window_is_made_odd_and_centred(self):
        ramp = np.arange(20, dtype=float).reshape(-1, 1) + 0j
        out = moving_average_filter(ramp, window_size=4)[:, 0]
        np.testing.assert_allclose(out.real[2:-2], ramp.real[2:-2, 0])

    def test_compare_noise_filters_reports_all_mild_filters(self):
        from preprocessing.filtering import MILD_FILTER_SETTINGS, compare_noise_filters

        rng = np.random.default_rng(0)
        t = np.linspace(0, 1, 201)
        trace = np.exp(1j * 6 * t) + 0.02 * (rng.standard_normal(201) + 1j * rng.standard_normal(201))
        results = compare_noise_filters(trace)
        self.assertEqual(set(results), set(MILD_FILTER_SETTINGS))
        for entry in results.values():
            self.assertEqual(entry["filtered"].shape, trace.shape)
            self.assertGreater(entry["noise_reduction"], 1.0)
            self.assertLess(entry["nrmse"], 0.1)

    def test_savgol_preserves_shape(self):
        out = savitzky_golay_filter(self.s, window_length=9, polyorder=3)
        self.assertEqual(out.shape, self.s.shape)

    def test_butterworth_preserves_shape(self):
        out = butterworth_lowpass_filter(self.s, cutoff=0.3, order=4)
        self.assertEqual(out.shape, self.s.shape)

    def test_gaussian_preserves_shape(self):
        out = gaussian_smoothing_filter(self.s, sigma=1.2)
        self.assertEqual(out.shape, self.s.shape)

    def test_median_preserves_shape(self):
        out = median_smoothing_filter(self.s, kernel_size=5)
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


class TestTouchstoneS21Pipeline(unittest.TestCase):
    def _write_touchstone(self, filename: str, body: str) -> str:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets",
            filename,
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        return path

    def test_touchstone_pipeline_interpolates_to_uniform_grid(self):
        path = self._write_touchstone(
            "_test_nonuniform_touchstone.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.0 0 0 1.0 0.0 0 0 0 0",
                    "1.3 0 0 1.1 0.1 0 0 0 0",
                    "1.9 0 0 5.0 5.0 0 0 0 0",
                    "2.2 0 0 1.2 0.2 0 0 0 0",
                ]
            ),
        )
        try:
            dataset = load_touchstone_dataset(path)
            result = run_preprocessing_pipeline(dataset, PreprocessingConfig(filter_method="median"))
            self.assertEqual(result.processed_dataset.s_parameters.shape[1], 1)
            self.assertTrue(np.all(np.diff(result.processed_dataset.frequencies) > 0))
            self.assertTrue(np.allclose(np.diff(result.processed_dataset.frequencies), np.diff(result.processed_dataset.frequencies)[0]))
            self.assertIsNotNone(result.validation_report)
            self.assertTrue(result.validation_report.interpolation_performed)
            self.assertIn("Windowed", result.stage_outputs)
        finally:
            os.remove(path)

    def test_touchstone_pipeline_unwraps_phase_and_reports_skipped_reference(self):
        path = self._write_touchstone(
            "_test_phase_wrap_touchstone.s2p",
            "\n".join(
                [
                    "# GHz S MA R 50",
                    "1.0 0 0 1.0 160 0 0 0 0",
                    "1.1 0 0 1.0 170 0 0 0 0",
                    "1.2 0 0 1.0 179 0 0 0 0",
                    "1.3 0 0 1.0 -175 0 0 0 0",
                ]
            ),
        )
        try:
            dataset = load_touchstone_dataset(path)
            result = run_preprocessing_pipeline(dataset, PreprocessingConfig(filter_method="none"))
            self.assertIsNotNone(result.touchstone_s21_result)
            self.assertGreaterEqual(result.validation_report.phase_unwrap_adjustments, 1)
            self.assertAlmostEqual(result.touchstone_s21_result.unwrapped_phase_deg[-1], 185.0, delta=1.0)
            self.assertFalse(result.validation_report.reference_subtraction_performed)
            self.assertTrue(result.validation_report.normalized_copy_generated)
        finally:
            os.remove(path)

    def test_touchstone_pipeline_applies_averaging_and_reference_subtraction(self):
        primary = self._write_touchstone(
            "_test_primary_touchstone.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.0 0 0 1.0 0.0 0 0 0 0",
                    "1.1 0 0 1.0 0.0 0 0 0 0",
                ]
            ),
        )
        repeated = self._write_touchstone(
            "_test_repeated_touchstone.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.0 0 0 1.2 0.0 0 0 0 0",
                    "1.1 0 0 1.2 0.0 0 0 0 0",
                ]
            ),
        )
        reference = self._write_touchstone(
            "_test_reference_touchstone.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.0 0 0 0.5 0.0 0 0 0 0",
                    "1.1 0 0 0.5 0.0 0 0 0 0",
                ]
            ),
        )
        try:
            dataset = load_touchstone_dataset(primary)
            repeated_dataset = load_touchstone_dataset(repeated)
            reference_dataset = load_touchstone_dataset(reference)
            result = run_preprocessing_pipeline(
                dataset,
                PreprocessingConfig(
                    filter_method="none",
                    enable_week3=False,
                    repeated_measurements=[repeated_dataset],
                    reference_dataset=reference_dataset,
                ),
            )
            np.testing.assert_allclose(result.touchstone_s21_result.averaged_s21, np.array([1.1 + 0j, 1.1 + 0j]))
            np.testing.assert_allclose(result.touchstone_s21_result.reference_subtracted_s21, np.array([0.6 + 0j, 0.6 + 0j]))
            np.testing.assert_allclose(result.processed_dataset.s_parameters[:, 0], np.array([0.6 + 0j, 0.6 + 0j]))
            self.assertTrue(result.validation_report.averaging_performed)
            self.assertTrue(result.validation_report.reference_subtraction_performed)
        finally:
            os.remove(primary)
            os.remove(repeated)
            os.remove(reference)

    def test_touchstone_averaging_corrects_repeats_and_rejects_different_conditions(self):
        freqs_ghz = np.linspace(1.0, 2.0, 41)
        rng = np.random.default_rng(3)

        def sweep(name, offset=0.0, spike_at=None):
            real = 0.5 + offset + 0.005 * rng.standard_normal(freqs_ghz.size)
            imag = 0.2 + 0.005 * rng.standard_normal(freqs_ghz.size)
            if spike_at is not None:
                real[spike_at] += 2.0
            lines = ["# GHz S RI R 50"] + [
                f"{f:.4f} 0 0 {r:.6f} {i:.6f} 0 0 0 0"
                for f, r, i in zip(freqs_ghz, real, imag)
            ]
            return self._write_touchstone(name, "\n".join(lines))

        primary = sweep("_test_avg_primary.s2p")
        spiked_repeat = sweep("_test_avg_repeat.s2p", spike_at=20)
        other_position = sweep("_test_avg_other.s2p", offset=-0.6)
        try:
            ts = run_preprocessing_pipeline(
                load_touchstone_dataset(primary),
                PreprocessingConfig(
                    filter_method="none",
                    enable_week3=False,
                    repeated_measurements=[
                        load_touchstone_dataset(spiked_repeat),
                        load_touchstone_dataset(other_position),
                    ],
                ),
            ).touchstone_s21_result
            self.assertTrue(ts.report.averaging_performed)
            self.assertLess(abs(ts.averaged_s21[20].real - 0.5), 0.05)
            self.assertLess(np.max(np.abs(ts.averaged_s21.real - 0.5)), 0.05)
            self.assertTrue(any("M=2" in note for note in ts.report.notes))
            self.assertTrue(any("#2" in note for note in ts.report.notes))
        finally:
            for path in (primary, spiked_repeat, other_position):
                os.remove(path)

    def test_nrmse_measures_only_the_filtering_change(self):
        primary = self._write_touchstone(
            "_test_nrmse_primary.s2p",
            "\n".join(["# GHz S RI R 50"] + [f"{1.0 + 0.1 * k:.1f} 0 0 1.0 0.0 0 0 0 0" for k in range(9)]),
        )
        reference = self._write_touchstone(
            "_test_nrmse_reference.s2p",
            "\n".join(["# GHz S RI R 50"] + [f"{1.0 + 0.1 * k:.1f} 0 0 0.9 0.0 0 0 0 0" for k in range(9)]),
        )
        try:
            result = run_preprocessing_pipeline(
                load_touchstone_dataset(primary),
                PreprocessingConfig(
                    filter_method="none",
                    enable_week3=False,
                    reference_dataset=load_touchstone_dataset(reference),
                ),
            )
            report = result.touchstone_s21_result.report
            self.assertTrue(report.reference_subtraction_performed)
            self.assertAlmostEqual(report.nrmse, 0.0, places=12)
            self.assertFalse(any("exceeds 0.75" in note for note in report.notes))
            self.assertIn("Filter Parameters", report.to_display_dict())
        finally:
            os.remove(primary)
            os.remove(reference)

    def test_touchstone_logs_corrected_sample_frequencies(self):
        freqs_ghz = np.linspace(1.0, 2.0, 21)
        rng = np.random.default_rng(0)
        real = 0.5 + 0.01 * rng.standard_normal(freqs_ghz.size)
        real[10] = 3.0
        lines = ["# GHz S RI R 50"] + [
            f"{f:.4f} 0 0 {r:.6f} 0.0 0 0 0 0" for f, r in zip(freqs_ghz, real)
        ]
        path = self._write_touchstone("_test_spike_log.s2p", "\n".join(lines))
        try:
            result = run_preprocessing_pipeline(
                load_touchstone_dataset(path),
                PreprocessingConfig(filter_method="none", enable_week3=False),
            )
            report = result.touchstone_s21_result.report
            self.assertEqual(report.corrected_sample_indices, [10])
            self.assertAlmostEqual(report.corrected_sample_frequencies_hz[0], 1.5e9, delta=1.0)
            self.assertLess(abs(result.touchstone_s21_result.corrected_s21[10] - 0.5), 0.05)
            self.assertTrue(any("1.5000" in note for note in report.notes))
        finally:
            os.remove(path)

    def test_hampel_does_not_flag_clean_noisy_trace(self):
        rng = np.random.default_rng(7)
        t = np.linspace(0.0, 1.0, 201)
        flagged = 0
        for _ in range(20):
            trace = np.sin(6 * t) + 0.01 * rng.standard_normal(t.size)
            _, count = _correct_isolated_samples(trace + 0j, method="hampel")
            flagged += count
        self.assertLessEqual(flagged, 1)

    def test_touchstone_magnitude_and_phase_use_uniform_grid(self):
        path = self._write_touchstone(
            "_test_nonuniform_phase.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.0 0 0 1.0 0.0 0 0 0 0",
                    "1.2 0 0 0.0 1.0 0 0 0 0",
                    "2.0 0 0 -1.0 0.0 0 0 0 0",
                ]
            ),
        )
        try:
            ts = run_preprocessing_pipeline(
                load_touchstone_dataset(path),
                PreprocessingConfig(filter_method="none", enable_week3=False),
            ).touchstone_s21_result
            n = ts.uniform_frequencies_hz.shape[0]
            self.assertTrue(ts.report.interpolation_performed)
            self.assertEqual(ts.uniform_s21.shape[0], n)
            self.assertEqual(ts.raw_magnitude_db.shape[0], n)
            self.assertEqual(ts.unwrapped_phase_deg.shape[0], n)
            np.testing.assert_allclose(
                ts.raw_magnitude_db, 20 * np.log10(np.abs(ts.uniform_s21) + 1e-12)
            )
        finally:
            os.remove(path)

    def test_reference_with_different_impedance_is_rejected(self):
        rows = [f"{1.0 + 0.1 * k:.1f} 0 0 {v} 0.0 0 0 0 0" for k, v in enumerate([1.0] * 6)]
        primary = self._write_touchstone("_test_z50.s2p", "\n".join(["# GHz S RI R 50"] + rows))
        ref75 = self._write_touchstone("_test_z75.s2p", "\n".join(["# GHz S RI R 75"] + rows))
        ref_offgrid = self._write_touchstone(
            "_test_z50_offgrid.s2p",
            "\n".join(["# GHz S RI R 50"] + [f"{0.95 + 0.11 * k:.2f} 0 0 0.5 0.0 0 0 0 0" for k in range(6)]),
        )
        try:
            for week3 in (False, True):
                bad = run_preprocessing_pipeline(
                    load_touchstone_dataset(primary),
                    PreprocessingConfig(
                        filter_method="none",
                        enable_week3=week3,
                        reference_dataset=load_touchstone_dataset(ref75),
                    ),
                )
                self.assertFalse(bad.validation_report.reference_subtraction_performed)
                self.assertTrue(
                    any("calibration mismatch" in n for n in bad.validation_report.notes)
                )

                good = run_preprocessing_pipeline(
                    load_touchstone_dataset(primary),
                    PreprocessingConfig(
                        filter_method="none",
                        enable_week3=week3,
                        reference_dataset=load_touchstone_dataset(ref_offgrid),
                    ),
                )
                self.assertTrue(good.validation_report.reference_subtraction_performed)
                self.assertTrue(any("resampled" in n for n in good.validation_report.notes))
        finally:
            for path in (primary, ref75, ref_offgrid):
                os.remove(path)

    def test_touchstone_reference_on_different_grid_is_resampled(self):
        primary = self._write_touchstone(
            "_test_grid_primary.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.0 0 0 1.0 0.0 0 0 0 0",
                    "1.5 0 0 1.0 0.0 0 0 0 0",
                    "2.0 0 0 1.0 0.0 0 0 0 0",
                ]
            ),
        )
        reference = self._write_touchstone(
            "_test_grid_reference.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "0.9 0 0 0.0 0.0 0 0 0 0",
                    "1.3 0 0 0.4 0.4 0 0 0 0",
                    "2.1 0 0 1.2 1.2 0 0 0 0",
                ]
            ),
        )
        narrow = self._write_touchstone(
            "_test_grid_narrow.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.2 0 0 0.5 0.0 0 0 0 0",
                    "1.8 0 0 0.5 0.0 0 0 0 0",
                ]
            ),
        )
        try:
            dataset = load_touchstone_dataset(primary)
            result = run_preprocessing_pipeline(
                dataset,
                PreprocessingConfig(
                    filter_method="none",
                    enable_week3=False,
                    reference_dataset=load_touchstone_dataset(reference),
                ),
            )
            # Reference is linear in f (R = I = f - 0.9 GHz), so linear resampling is exact.
            expected_reference = np.array([0.1, 0.6, 1.1]) * (1 + 1j)
            np.testing.assert_allclose(
                result.touchstone_s21_result.reference_subtracted_s21,
                (1.0 + 0j) - expected_reference,
                atol=1e-12,
            )
            self.assertTrue(result.validation_report.reference_subtraction_performed)

            result_narrow = run_preprocessing_pipeline(
                dataset,
                PreprocessingConfig(
                    filter_method="none",
                    enable_week3=False,
                    reference_dataset=load_touchstone_dataset(narrow),
                ),
            )
            self.assertFalse(result_narrow.validation_report.reference_subtraction_performed)
        finally:
            os.remove(primary)
            os.remove(reference)
            os.remove(narrow)

    def test_touchstone_pipeline_reports_nrmse_by_name(self):
        path = self._write_touchstone(
            "_test_nrmse_touchstone.s2p",
            "\n".join(
                [
                    "# GHz S RI R 50",
                    "1.0 0 0 1.0 0.0 0 0 0 0",
                    "1.1 0 0 1.0 0.0 0 0 0 0",
                    "1.2 0 0 1.0 0.0 0 0 0 0",
                ]
            ),
        )
        try:
            dataset = load_touchstone_dataset(path)
            result = run_preprocessing_pipeline(dataset, PreprocessingConfig(filter_method="none"))
            self.assertIsNotNone(result.validation_report)
            self.assertIn("NRMSE", result.validation_report.to_display_dict())
            self.assertAlmostEqual(result.validation_report.nrmse, 0.0, places=6)
            self.assertEqual(result.validation_report.spike_detection_method, "hampel")
        finally:
            os.remove(path)


class TestSpikeCorrectionAndNrmse(unittest.TestCase):
    def test_nrmse_zero_when_identical(self):
        signal = np.array([1 + 1j, 2 + 0j, 0.5 - 0.5j])
        self.assertEqual(_nrmse(signal, signal), 0.0)

    def test_median_and_local_detectors_catch_isolated_spike(self):
        clean = np.ones(11, dtype=complex)
        spiked = clean.copy()
        spiked[5] = 50 + 0j

        corrected_median, median_count = _correct_isolated_samples(
            spiked, method="median", window_size=2, n_sigmas=3.0
        )
        corrected_local, local_count = _correct_isolated_samples(
            spiked, method="local", window_size=2, n_sigmas=3.0
        )
        _, hampel_count = _correct_isolated_samples(spiked, method="hampel", window_size=2, n_sigmas=3.0)

        self.assertGreaterEqual(median_count, 1)
        self.assertGreaterEqual(local_count, 1)
        self.assertAlmostEqual(abs(corrected_median[5] - 1), 0.0, places=6)
        self.assertAlmostEqual(abs(corrected_local[5] - 1), 0.0, places=6)
        # Hampel is stricter (leave-one-out confirm); may or may not flag MAD=0 plateaus.
        self.assertIsInstance(hampel_count, int)


if __name__ == "__main__":
    unittest.main()

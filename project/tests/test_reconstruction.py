"""Unit tests for reconstruction modules."""

from __future__ import annotations

import numpy as np
import unittest

from reconstruction.ifft import frequency_to_time
from reconstruction.das import das_reconstruct
from reconstruction.dmas import dmas_reconstruct
from reconstruction.dmas_d4 import dmas_d4_reconstruct
from reconstruction.reconstruction_manager import (
    ROIRefinement,
    build_circular_antenna_array,
    infer_reconstruction_assessment,
    infer_reconstruction_config,
    reconstruct_all,
    reconstruct_high_resolution_roi,
    select_best_reconstruction,
)
from data_loader.dataset_info import MicrowaveDataset


def _make_synthetic_frequency_data(n_freq=64, n_traces=8):
    freqs = np.linspace(1e9, 8e9, n_freq)
    data = np.exp(-((freqs[:, None] - 4.5e9) ** 2) / (2 * (0.8e9) ** 2))
    data = data * np.exp(1j * np.linspace(0, np.pi, n_traces)[None, :])
    return freqs, data


def _make_point_target_frequency_data(
    freqs: np.ndarray,
    antenna_positions: np.ndarray,
    target_xy: tuple[float, float],
    amplitude: float = 1.0,
    wave_speed: float = 3e8,
) -> np.ndarray:
    target = np.asarray(target_xy, dtype=float)
    distances = np.linalg.norm(antenna_positions - target[None, :], axis=1)
    round_trip = 2.0 * distances / wave_speed
    phase = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * round_trip[None, :])
    envelope = np.exp(-((freqs[:, None] - 4.5e9) ** 2) / (2.0 * (1.0e9) ** 2))
    return amplitude * envelope * phase


class TestIFFT(unittest.TestCase):
    def test_frequency_to_time_shape(self):
        freqs, data = _make_synthetic_frequency_data()
        t = frequency_to_time(data, freqs)
        self.assertEqual(t.shape, data.shape)

    def test_ifft_real_expected(self):
        freqs = np.linspace(1e9, 8e9, 8)
        data = np.ones((8, 2), dtype=complex)
        t = frequency_to_time(data, freqs)
        self.assertEqual(t.shape, (8, 2))
        self.assertTrue(np.isclose(np.abs(t[0, 0]), 1.0))
        self.assertTrue(np.allclose(t[1:, 0], 0.0, atol=1e-8))


class TestBeamforming(unittest.TestCase):
    def setUp(self):
        self.freqs, self.data = _make_synthetic_frequency_data(n_freq=64, n_traces=8)
        self.ant_pos = build_circular_antenna_array(8, radius=0.05)
        self.grid_x = np.linspace(-0.02, 0.02, 16)
        self.grid_y = np.linspace(-0.02, 0.02, 16)
        self.time_data = frequency_to_time(self.data, self.freqs)

    def test_das_reconstruct_shape(self):
        image = das_reconstruct(self.time_data, self.freqs, self.ant_pos, self.grid_x, self.grid_y)
        self.assertEqual(image.shape, (len(self.grid_y), len(self.grid_x)))

    def test_dmas_reconstruct_shape(self):
        image = dmas_reconstruct(self.time_data, self.freqs, self.ant_pos, self.grid_x, self.grid_y)
        self.assertEqual(image.shape, (len(self.grid_y), len(self.grid_x)))

    def test_dmas_d4_reconstruct_shape(self):
        image = dmas_d4_reconstruct(self.time_data, self.freqs, self.ant_pos, self.grid_x, self.grid_y)
        self.assertEqual(image.shape, (len(self.grid_y), len(self.grid_x)))

    def test_reconstruct_all_returns_keys(self):
        images = reconstruct_all(self.data, self.freqs, self.ant_pos, x_span=(-0.02, 0.02), y_span=(-0.02, 0.02), n_x=16, n_y=16)
        self.assertIn("DAS", images)
        self.assertIn("DMAS", images)
        self.assertIn("DMAS-D4", images)

    def test_select_best_reconstruction(self):
        images = {
            "DAS": np.ones((16, 16)),
            "DMAS": np.ones((16, 16)) * 2,
            "DMAS-D4": np.ones((16, 16)) * 0.5,
        }
        selected, best = select_best_reconstruction(images)
        self.assertEqual(selected, "DMAS")
        self.assertEqual(best.shape, (16, 16))

    def test_reconstruct_high_resolution_roi_returns_refinement(self):
        refinement = reconstruct_high_resolution_roi(
            self.data,
            self.freqs,
            "DAS",
            bounding_box=(4, 5, 10, 11),
            coarse_shape=(16, 16),
            antenna_positions=self.ant_pos,
            full_x_span=(-0.02, 0.02),
            full_y_span=(-0.02, 0.02),
            upscale_factor=4,
            min_grid_size=32,
            max_grid_size=64,
        )
        self.assertIsInstance(refinement, ROIRefinement)
        self.assertEqual(refinement.algorithm, "DAS")
        self.assertEqual(refinement.image.shape, (32, 32))
        self.assertLess(refinement.x_span[0], refinement.x_span[1])
        self.assertLess(refinement.y_span[0], refinement.y_span[1])

    def test_das_localizes_known_point_target(self):
        freqs = np.linspace(1e9, 8e9, 256)
        antenna_positions = build_circular_antenna_array(12, radius=0.08)
        target_xy = (0.012, -0.008)
        data = _make_point_target_frequency_data(freqs, antenna_positions, target_xy)
        grid_x = np.linspace(-0.03, 0.03, 61)
        grid_y = np.linspace(-0.03, 0.03, 61)
        image = reconstruct_all(
            data,
            freqs,
            antenna_positions=antenna_positions,
            x_span=(-0.03, 0.03),
            y_span=(-0.03, 0.03),
            n_x=61,
            n_y=61,
        )["DAS"]
        peak_y, peak_x = np.unravel_index(int(np.argmax(image)), image.shape)
        peak_xy = (float(grid_x[peak_x]), float(grid_y[peak_y]))
        error = float(np.linalg.norm(np.asarray(peak_xy) - np.asarray(target_xy)))
        self.assertLess(error, 0.01)


class TestReconstructionConfig(unittest.TestCase):
    def test_infer_reconstruction_config_from_metadata(self):
        dataset = MicrowaveDataset(
            file_path="synthetic.mat",
            file_name="synthetic.mat",
            file_type="MATLAB",
            frequencies=np.linspace(1e9, 9e9, 10),
            s_parameters=np.ones((10, 4), dtype=complex),
            n_ports=4,
            metadata={
                "antenna_radius_m": 0.10,
                "wave_speed_m_per_s": 2.1e8,
                "reconstruction_x_span_m": (-0.06, 0.06),
                "reconstruction_y_span_m": (-0.04, 0.04),
            },
        )
        config = infer_reconstruction_config(dataset)
        self.assertEqual(config.antenna_radius, 0.10)
        self.assertEqual(config.wave_speed, 2.1e8)
        self.assertEqual(config.x_span, (-0.06, 0.06))
        self.assertEqual(config.y_span, (-0.04, 0.04))

    def test_infer_reconstruction_assessment_marks_defaults(self):
        dataset = MicrowaveDataset(
            file_path="synthetic.mat",
            file_name="synthetic.mat",
            file_type="MATLAB",
            frequencies=np.linspace(1e9, 9e9, 10),
            s_parameters=np.ones((10, 4), dtype=complex),
            n_ports=4,
            metadata={},
        )
        assessment = infer_reconstruction_assessment(dataset)
        self.assertEqual(assessment.config.antenna_radius, 0.08)
        self.assertEqual(assessment.config.wave_speed, 3e8)
        self.assertTrue(assessment.warnings)
        self.assertIn("default", " ".join(assessment.warnings).lower())


if __name__ == "__main__":
    unittest.main()

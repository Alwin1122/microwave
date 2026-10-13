"""Tests for the per-step Module 2 GUI figures."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.dataset_info import MicrowaveDataset
from data_loader.touchstone_loader import load_touchstone_dataset
from gui.module2_step_plots import STEP_TITLES, plot_module2_step
from preprocessing.preprocessing_pipeline import PreprocessingConfig, run_preprocessing_pipeline

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "sample_touchstone.s2p")


def _texts(figure: Figure) -> str:
    return " ".join(t.get_text() for ax in figure.axes for t in ax.texts)


class TestModule2StepPlots(unittest.TestCase):
    def test_sixteen_steps(self):
        self.assertEqual(len(STEP_TITLES), 16)

    def test_every_step_draws_for_touchstone(self):
        result = run_preprocessing_pipeline(load_touchstone_dataset(SAMPLE), PreprocessingConfig())
        for step in range(16):
            figure = Figure()
            plot_module2_step(figure, step, result)
            self.assertTrue(figure.axes, STEP_TITLES[step])
            self.assertNotIn("Could not draw", _texts(figure), STEP_TITLES[step])

    def test_multichannel_shows_s2p_message_then_time_domain(self):
        freqs = np.linspace(1e9, 8e9, 41)
        rng = np.random.default_rng(1)
        s_params = (rng.standard_normal((41, 6)) + 1j * rng.standard_normal((41, 6))) * 1e-3
        dataset = MicrowaveDataset(
            file_path="multi.mat", file_name="multi.mat", file_type="synthetic",
            frequencies=freqs, s_parameters=s_params, n_ports=6,
        )
        result = run_preprocessing_pipeline(
            dataset,
            PreprocessingConfig(calibration_method="none", artifact_method="none", do_background_subtraction=False),
        )
        figure = Figure()
        plot_module2_step(figure, 0, result)
        self.assertIn("Steps 1–11", _texts(figure))
        for step in range(11, 16):
            figure = Figure()
            plot_module2_step(figure, step, result)
            self.assertNotIn("Could not draw", _texts(figure), STEP_TITLES[step])

    def test_no_result_message(self):
        figure = Figure()
        plot_module2_step(figure, 4, None)
        self.assertIn("Run Module 2", _texts(figure))


if __name__ == "__main__":
    unittest.main()

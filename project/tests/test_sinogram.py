"""Sinogram window: a symmetric shell dies in the residual, a point echo does not."""

from __future__ import annotations

import unittest

import numpy as np

from classification.features import breast_time_mask
from classification.sinogram import resample_circle, sinogram_pair
from classification.sinogram_net import ADOPT_IF_MEAN_AUC, OFFICIAL_FIRST_SCAN_AUC, circular_shift
from data_loader.bmid_loader import bmid_frequencies_hz


class TestSinogram(unittest.TestCase):
    def setUp(self):
        self.freqs = bmid_frequencies_hz()

    def test_breast_window_is_the_published_interval(self):
        mask = breast_time_mask(self.freqs)
        self.assertGreater(int(mask.sum()), 20)
        self.assertLess(int(mask.sum()), 50)

    def test_resample_keeps_a_peak_when_the_count_matches(self):
        image = np.zeros((3, 8))
        image[:, 2] = 1.0
        out = resample_circle(image, 8)
        self.assertEqual(int(np.argmax(out[0])), 2)

    def test_symmetric_shell_has_no_rotational_channel(self):
        pulse = np.exp(-((self.freqs - 4e9) ** 2) / (2 * (0.8e9) ** 2))
        shell = np.repeat(pulse[:, None], 12, axis=1).astype(complex)
        pair = sinogram_pair(shell, np.zeros_like(shell), self.freqs, n_out=16)
        self.assertEqual(pair.shape[0], 2)
        self.assertLess(float(pair[1].std()), 1e-5)
        self.assertGreater(float(pair[0].std()), 0.0)

    def test_circular_shift_wraps_the_antenna_axis(self):
        batch = np.zeros((1, 2, 4, 8))
        batch[0, 0, 0, 1] = 1.0
        shifted = circular_shift(batch, 1)
        self.assertEqual(float(shifted[0, 0, 0, 2]), 1.0)

    def test_adoption_bar_is_above_the_official_spread(self):
        self.assertGreater(ADOPT_IF_MEAN_AUC, OFFICIAL_FIRST_SCAN_AUC + 0.015)

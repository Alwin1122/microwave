"""Checks for the first-visit scattering observables and the criticality index."""

from __future__ import annotations

import unittest

import numpy as np

from classification.criticality import criticality_class, criticality_index, morphology_type
from classification.literature import PUBLISHED
from classification.scattering_math import MATH_FEATURES, range_bearing_cm, scattering_descriptors
from data_loader.bmid_loader import bmid_frequencies_hz
from quality.tumor_taxonomy import classify_scan


def _rank1(freqs: np.ndarray, antenna: int, n_ant: int, law: np.ndarray, delay_s: float) -> np.ndarray:
    phase = np.exp(-2j * np.pi * freqs * delay_s)
    data = np.zeros((freqs.size, n_ant), dtype=complex)
    data[:, antenna] = law * phase
    return data


class TestScatteringDescriptors(unittest.TestCase):
    def setUp(self):
        self.freqs = bmid_frequencies_hz()

    def test_point_echo_is_rank_one_and_peaks_on_its_antenna(self):
        law = (self.freqs / 1e9) ** 2
        data = _rank1(self.freqs, 5, 24, law, 2.0e-9)
        feats = scattering_descriptors(data, self.freqs)
        self.assertGreater(feats["math_lambda1"], 0.95)
        self.assertLess(feats["math_effective_rank"], 1.2)
        self.assertEqual(int(feats["math_peak_antenna"]), 5)
        self.assertAlmostEqual(feats["math_delay_peak_ns"], 2.0, delta=0.2)
        self.assertTrue(all(np.isfinite(feats[name]) for name in MATH_FEATURES))

    def test_rayleigh_slope_is_near_two_and_flat_target_is_near_zero(self):
        rising = _rank1(self.freqs, 3, 16, (self.freqs / 1e9) ** 2, 2.0e-9)
        flat = _rank1(self.freqs, 3, 16, np.ones(self.freqs.size), 2.0e-9)
        self.assertAlmostEqual(scattering_descriptors(rising, self.freqs)["math_slope_beta"], 2.0, delta=0.15)
        self.assertAlmostEqual(scattering_descriptors(flat, self.freqs)["math_slope_beta"], 0.0, delta=0.15)

    def test_compact_target_outranks_diffuse_clutter(self):
        rng = np.random.default_rng(0)
        compact = _rank1(self.freqs, 4, 24, np.ones(self.freqs.size), 2.0e-9)
        clutter = rng.normal(size=(self.freqs.size, 24)) + 1j * rng.normal(size=(self.freqs.size, 24))
        self.assertGreater(
            scattering_descriptors(compact, self.freqs)["math_lambda1"],
            scattering_descriptors(clutter, self.freqs)["math_lambda1"] + 0.4,
        )

    def test_symmetric_shell_carries_no_residual(self):
        column = np.exp(-2j * np.pi * self.freqs * 2.0e-9)
        shell = np.repeat(column[:, None], 12, axis=1)
        feats = scattering_descriptors(shell, self.freqs)
        self.assertAlmostEqual(feats["math_lambda1"], 0.0, places=6)
        self.assertAlmostEqual(feats["math_angular_R"], 0.0, places=6)

    def test_range_bearing_inverts_the_round_trip(self):
        radius_m = 0.21
        speed = 2.7e8
        delay = 2.0 * (radius_m - 0.03) / speed
        x_cm, y_cm = range_bearing_cm(delay, 0.0, radius_m, speed)
        self.assertAlmostEqual(x_cm, 3.0, delta=0.05)
        self.assertAlmostEqual(y_cm, 0.0, delta=0.05)


class TestCriticality(unittest.TestCase):
    def test_volume_bins_and_spiculated_bump(self):
        self.assertEqual(morphology_type("sphere"), "spherical")
        self.assertEqual(morphology_type("spiculated"), "spiculated")
        self.assertAlmostEqual(criticality_index(2.0, "sphere"), 1.0)
        self.assertEqual(criticality_class(criticality_index(2.0, "sphere")), "low")
        self.assertAlmostEqual(criticality_index(2.0, "spiculated"), 1.5)
        self.assertEqual(criticality_class(criticality_index(2.0, "spiculated")), "moderate")
        self.assertEqual(criticality_class(criticality_index(4.0, "sphere")), "moderate")
        self.assertEqual(criticality_class(criticality_index(6.0, "sphere")), "high")
        self.assertIsNone(criticality_index(None, "sphere"))

    def test_taxonomy_carries_type_and_criticality(self):
        sphere = classify_scan(has_tumor=True, tum_diam_cm=2.0, tum_shape="sphere")
        spic = classify_scan(has_tumor=True, tum_diam_cm=2.0, tum_shape="spiculated")
        healthy = classify_scan(has_tumor=False)
        self.assertEqual(sphere.morphology, "spherical")
        self.assertEqual(sphere.criticality_class, "low")
        self.assertEqual(spic.criticality_class, "moderate")
        self.assertIsNone(healthy.criticality_class)
        self.assertEqual(healthy.short_label(), "healthy")


class TestLiteratureTable(unittest.TestCase):
    def test_every_cited_result_names_its_restriction(self):
        self.assertGreaterEqual(len(PUBLISHED), 4)
        for row in PUBLISHED:
            self.assertIn("doi:", row["citation"].lower())
            self.assertTrue(row["result"])
            self.assertTrue(row["restriction"])
            self.assertNotIn("our auc", row["result"].lower())


if __name__ == "__main__":
    unittest.main()

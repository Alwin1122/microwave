"""Tests for Module 9 characterization and Module 10 confidence."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quality.characterization import characterize_tumor_region
from quality.confidence import assess_reconstruction_confidence
from roi.roi_detector import detect_roi


class TestCharacterizationConfidence(unittest.TestCase):
    def test_characterize_compact_hotspot(self):
        image = np.zeros((64, 64), dtype=float)
        image[20:26, 40:46] = 5.0
        roi = detect_roi(image, threshold_ratio=0.6, min_area=4, margin=1, sigma=0.4)
        char = characterize_tumor_region(
            image,
            roi,
            x_span=(-0.06, 0.06),
            y_span=(-0.06, 0.06),
        )
        self.assertGreater(char.equivalent_diameter_cm, 0.0)
        self.assertGreater(char.area_cm2, 0.0)
        self.assertGreater(char.local_scr, 0.0)
        self.assertGreaterEqual(char.compactness, 0.0)
        self.assertLessEqual(char.compactness, 1.0 + 1e-6)
        payload = char.to_dict()
        self.assertIn("fwhm_cm", payload)

    def test_confidence_higher_for_strong_candidate_near_gt(self):
        strong = assess_reconstruction_confidence(
            selected_quality={"scr": 12.0, "contrast": 0.8, "ccr": 3.0, "score": 1.2},
            all_quality_metrics={
                "DAS": {"score": 0.7},
                "DMAS": {"score": 0.9},
                "DMAS-D4": {"score": 1.2},
            },
            selected_beamformer="DMAS-D4",
            tumor_candidate={"confidence": 0.85, "is_tumor_candidate": True},
            characterization={
                "compactness": 0.8,
                "equivalent_diameter_cm": 2.0,
                "fwhm_cm": 1.8,
                "area_cm2": 3.0,
            },
            tumor_gt_distance_m=0.01,
        )
        weak = assess_reconstruction_confidence(
            selected_quality={"scr": 1.2, "contrast": 0.1, "ccr": 0.2, "score": 0.3},
            all_quality_metrics={
                "DAS": {"score": 0.35},
                "DMAS": {"score": 0.32},
                "DMAS-D4": {"score": 0.3},
            },
            selected_beamformer="DMAS-D4",
            tumor_candidate={"confidence": 0.15, "is_tumor_candidate": False},
            characterization={
                "compactness": 0.15,
                "equivalent_diameter_cm": 8.0,
                "fwhm_cm": 7.0,
                "area_cm2": 25.0,
            },
            tumor_gt_distance_m=0.06,
        )
        self.assertGreater(strong.overall, weak.overall)
        self.assertEqual(strong.label, "high")
        self.assertIn(weak.label, {"low", "medium"})
        self.assertTrue(strong.reasons)


if __name__ == "__main__":
    unittest.main()

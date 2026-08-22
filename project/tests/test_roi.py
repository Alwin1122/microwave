"""Tests for ROI localization."""

from __future__ import annotations

import unittest

import numpy as np

from roi.roi_detector import detect_roi


class TestROIDetector(unittest.TestCase):
    def test_detect_roi_finds_bright_region(self):
        image = np.zeros((32, 32), dtype=float)
        image[10:15, 20:25] = 5.0
        result = detect_roi(image, threshold_ratio=0.6, min_area=4, margin=1, sigma=0.5)
        x0, y0, x1, y1 = result.bounding_box
        self.assertLessEqual(x0, 20)
        self.assertLessEqual(y0, 10)
        self.assertGreaterEqual(x1, 25)
        self.assertGreaterEqual(y1, 15)
        self.assertGreater(result.area, 0)

    def test_detect_roi_fallback_peak(self):
        image = np.zeros((16, 16), dtype=float)
        image[7, 9] = 1.0
        result = detect_roi(image, threshold_ratio=0.95, min_area=50, margin=2, sigma=0.1)
        self.assertTrue(result.mask.any())
        self.assertAlmostEqual(result.centroid[0], 9.0, delta=2.0)
        self.assertAlmostEqual(result.centroid[1], 7.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()

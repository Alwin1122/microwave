"""Tests for ROI localization."""

from __future__ import annotations

import unittest

import numpy as np

from roi.roi_detector import detect_roi, roi_centroid_meters


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

    def test_prefer_off_center_avoids_origin_blob(self):
        image = np.zeros((64, 64), dtype=float)
        image[28:36, 28:36] = 3.0  # bright center clutter
        image[12:18, 44:50] = 2.5  # weaker off-center target
        peak = detect_roi(image, threshold_ratio=0.5, min_area=4, margin=1, sigma=0.5)
        off = detect_roi(
            image,
            threshold_ratio=0.5,
            min_area=4,
            margin=1,
            sigma=0.5,
            prefer_off_center=True,
            off_center_weight=3.0,
        )
        self.assertLess(abs(peak.centroid[0] - 31.5), 6.0)
        self.assertGreater(abs(off.centroid[0] - 31.5), 8.0)

    def test_tight_peak_snaps_to_max_inside_blob(self):
        image = np.zeros((48, 48), dtype=float)
        image[10:20, 20:35] = 2.0
        image[12, 33] = 6.0  # peak inside the same blob
        tight = detect_roi(image, threshold_ratio=0.5, min_area=4, margin=2, sigma=0.2, tight_peak=True)
        self.assertAlmostEqual(tight.centroid[0], 33.0, delta=1.5)
        self.assertAlmostEqual(tight.centroid[1], 12.0, delta=1.5)
        self.assertEqual(tight.mode, "tight")

    def test_roi_centroid_meters(self):
        image = np.zeros((11, 11), dtype=float)
        image[8, 2] = 1.0
        roi = detect_roi(image, threshold_ratio=0.5, min_area=1, margin=0, sigma=0.1)
        x_m, y_m = roi_centroid_meters(roi, image.shape, (-0.05, 0.05), (-0.05, 0.05))
        self.assertAlmostEqual(x_m, -0.03, places=2)
        self.assertAlmostEqual(y_m, 0.03, places=2)


if __name__ == "__main__":
    unittest.main()

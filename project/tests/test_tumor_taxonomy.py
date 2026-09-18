"""Tests for the tumor taxonomy base (labels only, not a classifier)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.bmid_loader import BmidScanInfo
from quality.tumor_taxonomy import (
    classify_scan,
    taxonomy_from_metadata,
    taxonomy_from_scan,
)
from automation.resolver import _choose_bmid_scans


def _scan(
    index: int,
    *,
    has_tumor: bool,
    diam: float | None = None,
    x: float | None = None,
    y: float | None = None,
    birads: int | None = None,
    shape: str | None = None,
) -> BmidScanInfo:
    return BmidScanInfo(
        index,
        index + 1,
        "A",
        has_tumor,
        diam,
        x,
        y,
        birads,
        18.0,
        f"s{index}",
        shape,
    )


class TestTumorTaxonomy(unittest.TestCase):
    def test_healthy_and_size_severity_buckets(self):
        healthy = classify_scan(has_tumor=False)
        self.assertEqual(healthy.presence, "healthy")
        self.assertEqual(healthy.short_label(), "healthy")

        small = classify_scan(
            has_tumor=True, tum_diam_cm=1.5, tum_x_cm=2.0, tum_y_cm=2.0, birads=2
        )
        self.assertEqual(small.size_class, "small")
        self.assertEqual(small.severity_class, "low")
        self.assertEqual(small.location_quadrant, "upper-right")

        medium = classify_scan(has_tumor=True, tum_diam_cm=3.0, birads=3)
        self.assertEqual(medium.size_class, "medium")
        self.assertEqual(medium.severity_class, "moderate")

        large = classify_scan(
            has_tumor=True,
            tum_diam_cm=6.0,
            tum_x_cm=-2.0,
            tum_y_cm=-1.5,
            birads=1,
            tum_shape="sphere",
        )
        self.assertEqual(large.size_class, "large")
        self.assertEqual(large.severity_class, "high")
        self.assertEqual(large.shape, "sphere")
        self.assertEqual(large.location_quadrant, "lower-left")

    def test_from_metadata_and_scan(self):
        tax = taxonomy_from_metadata(
            {
                "bmid_has_tumor": True,
                "tumor_diameter_m": 0.04,
                "tumor_x_m": 0.02,
                "tumor_y_m": -0.01,
                "bmid_birads": 3,
                "bmid_tum_shape": "sphere",
            }
        )
        self.assertEqual(tax.size_class, "medium")
        self.assertEqual(tax.shape, "sphere")
        scan = _scan(0, has_tumor=True, diam=2.0, x=1.0, y=1.0, birads=1, shape="sphere")
        self.assertEqual(taxonomy_from_scan(scan).size_class, "small")

    def test_diverse_strategy_covers_healthy_and_sizes(self):
        scans = [
            _scan(0, has_tumor=False),
            _scan(1, has_tumor=True, diam=1.5, x=2.0, y=2.0, birads=1, shape="sphere"),
            _scan(2, has_tumor=True, diam=3.0, x=-2.0, y=2.0, birads=3, shape="sphere"),
            _scan(3, has_tumor=True, diam=6.0, x=-2.0, y=-2.0, birads=4, shape=""),
            _scan(4, has_tumor=True, diam=1.5, x=2.0, y=2.0, birads=1, shape="sphere"),
        ]
        chosen = _choose_bmid_scans(scans, bmid_strategy="diverse", max_scans=8)
        labels = {taxonomy_from_scan(s).presence for s in chosen}
        sizes = {taxonomy_from_scan(s).size_class for s in chosen if s.has_tumor}
        self.assertIn("healthy", labels)
        self.assertTrue({"small", "medium", "large"}.issubset(sizes))
        self.assertLessEqual(len(chosen), 8)


if __name__ == "__main__":
    unittest.main()

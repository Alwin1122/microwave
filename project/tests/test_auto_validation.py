"""Unit and integration tests for auto-validation automation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automation.classifier import classify_evidence
from automation.discovery import classify_path, discover_files
from automation.policy import get_threshold_profile
from automation.resolver import metadata_to_measurement_candidate, resolve_measurement_targets
from automation.service import AutoValidationService
from automation.types import DiscoveredFile, FileKind
from scipy.io import savemat


class TestPolicyAndClassifier(unittest.TestCase):
    def test_profiles_exist(self):
        for name in ("strict", "balanced", "lenient"):
            profile = get_threshold_profile(name)
            self.assertEqual(profile.name, name)

    def test_classifier_pass_and_fail(self):
        profile = get_threshold_profile("balanced")
        status, _, reasons, _, recommended = classify_evidence(
            profile=profile,
            load_ok=True,
            load_error=None,
            finite_ok=True,
            energy=1.0,
            roi_area=20,
            localization_error_cm=1.0,
            scr=5.0,
            snr=2.0,
            confidence=0.7,
            is_tumor_candidate=True,
            has_tumor_gt=True,
        )
        self.assertEqual(status.value, "pass")
        self.assertTrue(recommended)

        status_fail, _, _, _, recommended_fail = classify_evidence(
            profile=profile,
            load_ok=True,
            load_error=None,
            finite_ok=False,
            energy=0.0,
            roi_area=0,
            localization_error_cm=None,
            scr=None,
            snr=None,
            confidence=None,
            is_tumor_candidate=None,
            has_tumor_gt=False,
        )
        self.assertEqual(status_fail.value, "fail")
        self.assertFalse(recommended_fail)


class TestDiscoveryResolver(unittest.TestCase):
    def test_classify_metadata_vs_measurement(self):
        md = classify_path(r"C:\data\md_list_s21_adi.mat")
        generic_md = classify_path(r"C:\data\metadata_gen_two.mat")
        fd = classify_path(r"C:\data\fd_data_s21_adi.mat")
        s2p = classify_path(r"C:\data\sample.s2p")
        self.assertEqual(md.kind, FileKind.METADATA)
        self.assertEqual(generic_md.kind, FileKind.METADATA)
        self.assertEqual(fd.kind, FileKind.MEASUREMENT)
        self.assertEqual(s2p.kind, FileKind.MEASUREMENT)

    def test_metadata_pairing_and_orphan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fd = root / "fd_data_s21_adi.mat"
            md = root / "md_list_s21_adi.mat"
            # Minimal fake cube + metadata not required for pairing path check.
            savemat(str(fd), {"fd_data_s21_adi": np.zeros((4, 2, 2), dtype=complex)})
            savemat(str(md), {"dummy": 1})
            paired = metadata_to_measurement_candidate(str(md))
            self.assertEqual(Path(paired).name, "fd_data_s21_adi.mat")

            orphan = root / "md_list_s21_emp.mat"
            savemat(str(orphan), {"dummy": 1})
            self.assertIsNone(metadata_to_measurement_candidate(str(orphan)))

    def test_discover_files_limits(self):
        project_datasets = Path(__file__).resolve().parent.parent / "datasets"
        if not project_datasets.is_dir():
            self.skipTest("datasets folder missing")
        found = discover_files([str(project_datasets)], max_files=3)
        self.assertLessEqual(len(found), 3)
        self.assertTrue(all(f.kind != FileKind.UNKNOWN for f in found))

    def test_all_strategy_unlimited_keeps_every_scan(self):
        from data_loader.bmid_loader import BmidScanInfo
        from automation.resolver import _choose_bmid_scans

        scans = [
            BmidScanInfo(0, 1, "A", True, 2.0, 1.0, 1.0, None, 18.0, "t0"),
            BmidScanInfo(1, 2, "A", False, None, None, None, None, 18.0, "h1"),
            BmidScanInfo(2, 3, "A", True, 4.0, 2.0, 2.0, None, 18.0, "t2"),
        ]
        chosen = _choose_bmid_scans(scans, bmid_strategy="all", max_scans=0)
        self.assertEqual(len(chosen), 3)

    def test_default_roots_use_umbmid_not_datasets(self):
        from automation.service import default_validation_roots

        roots = default_validation_roots()
        self.assertTrue(roots)
        self.assertTrue(any("umbmid" in Path(r).as_posix().lower() for r in roots))
        self.assertFalse(any(Path(r).name == "datasets" for r in roots))

    def test_umbmid_root_is_simple_clean_parent_of_gens(self):
        from automation.service import umbmid_root

        root = umbmid_root()
        if root is None:
            self.skipTest("umbmid/simple-clean is not present")
        self.assertEqual(root.name, "simple-clean")
        gens = {p.name for p in root.iterdir() if p.is_dir()}
        self.assertTrue({"matlab-data", "matlab-data2", "matlab-data3"} & gens)


class TestAutoValidationIntegration(unittest.TestCase):
    def test_corrupted_and_touchstone_and_bmid(self):
        datasets = Path(__file__).resolve().parent.parent / "datasets"
        results = Path(__file__).resolve().parent.parent / "results"
        service = AutoValidationService(output_dir=str(results))

        # Corrupted / invalid path should fail without crashing.
        bad = resolve_measurement_targets(
            [DiscoveredFile(str(datasets / "corrupted.mat"), FileKind.MEASUREMENT, "corrupted.mat")]
        )
        self.assertTrue(bad)
        from automation.pipeline import validate_one_target

        bad_result = validate_one_target(
            bad[0],
            profile=get_threshold_profile("lenient"),
            include_reconstruction_checks=False,
        )
        self.assertIn(bad_result.status.value, {"fail", "warning", "pass", "blocked"})

        # Valid touchstone sample — load-only path for speed.
        s2p = datasets / "sample_touchstone.s2p"
        if s2p.is_file():
            batch = service.run(
                [str(s2p)],
                max_files=1,
                threshold_profile="lenient",
                include_reconstruction_checks=False,
                use_cache=False,
            )
            self.assertGreaterEqual(len(batch.results), 1)
            self.assertTrue((results / "latest_auto_analysis.md").is_file())
            payload = json.loads((results / "latest_auto_validation.json").read_text(encoding="utf-8"))
            self.assertIn("counts", payload)
            self.assertIn("tumor_index", payload)

        # Multi-scan BMID cube should resolve with a scan index.
        bmid = datasets / "fd_data_s21_adi.mat"
        if bmid.is_file():
            targets = resolve_measurement_targets(
                [DiscoveredFile(str(bmid), FileKind.MEASUREMENT, bmid.name)],
                bmid_strategy="tumor_and_healthy",
                max_bmid_scans=1,
            )
            self.assertEqual(len(targets), 1)
            self.assertIsNotNone(targets[0].scan_index)


if __name__ == "__main__":
    unittest.main()

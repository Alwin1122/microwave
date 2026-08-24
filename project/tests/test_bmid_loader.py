"""Unit tests for UM-BMID scan loading helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.bmid_loader import (
    ScanSelectionRequiredError,
    companion_metadata_path,
    is_bmid_fd_filename,
    list_bmid_scans,
    load_bmid_scan,
)
from data_loader.loader import load_dataset
from data_loader.matlab_loader import load_matlab_dataset

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def _write_fake_bmid(folder: str) -> tuple[str, str]:
    n_scans, n_freq, n_ant = 3, 1001, 8
    fd = (np.random.randn(n_scans, n_freq, n_ant) + 1j * np.random.randn(n_scans, n_freq, n_ant)) * 1e-3
    fd_path = os.path.join(folder, "fd_data_s21_adi.mat")
    savemat(fd_path, {"fd_data_s21": fd})

    md = np.empty((3,), dtype=object)
    for i in range(3):
        md[i] = {
            "n_expt": i + 1,
            "id": i + 1,
            "phant_id": "A2F1",
            "tum_diam": 2.0 if i == 0 else np.nan,
            "tum_shape": "sphere" if i == 0 else "",
            "tum_x": 1.5 if i == 0 else np.nan,
            "tum_y": -1.0 if i == 0 else np.nan,
            "tum_z": -6.5 if i == 0 else np.nan,
            "birads": 1,
            "adi_ref_id": 3,
            "emp_ref_id": 16,
            "date": "20210801",
            "n_session": 1,
            "ant_rad": 18.0,
            "ant_z": -6.5,
            "fib_ang": 0.0,
            "adi_x": 0.0,
            "adi_y": 0.0,
            "fib_ref_id": 2,
            "fib_x": 0.0,
            "fib_y": 0.0,
            "tum_in_fib": 0,
        }
    md_path = os.path.join(folder, "md_list_s21_adi.mat")
    # scipy savemat needs struct array for nested fields; use simple dict of object array via dtype
    # Easiest path: save as cell-like object array of dicts is unreliable; build mat_struct via recarray.
    dtype = [
        ("n_expt", "O"),
        ("id", "O"),
        ("phant_id", "O"),
        ("tum_diam", "O"),
        ("tum_shape", "O"),
        ("tum_x", "O"),
        ("tum_y", "O"),
        ("tum_z", "O"),
        ("birads", "O"),
        ("adi_ref_id", "O"),
        ("emp_ref_id", "O"),
        ("date", "O"),
        ("n_session", "O"),
        ("ant_rad", "O"),
        ("ant_z", "O"),
        ("fib_ang", "O"),
        ("adi_x", "O"),
        ("adi_y", "O"),
        ("fib_ref_id", "O"),
        ("fib_x", "O"),
        ("fib_y", "O"),
        ("tum_in_fib", "O"),
    ]
    arr = np.zeros((1, 3), dtype=dtype)
    for i, item in enumerate(md):
        for key, value in item.items():
            arr[0, i][key] = value
    savemat(md_path, {"md_s21": arr})
    return fd_path, md_path


class TestBmidLoader(unittest.TestCase):
    def test_filename_detection(self):
        self.assertTrue(is_bmid_fd_filename("fd_data_s21_adi.mat"))
        self.assertTrue(is_bmid_fd_filename(r"C:\\data\\fd_data_s11_emp.mat"))
        self.assertFalse(is_bmid_fd_filename("sample_matlab.mat"))

    def test_synthetic_scan_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            fd_path, md_path = _write_fake_bmid(tmp)
            self.assertEqual(companion_metadata_path(fd_path), md_path)
            scans = list_bmid_scans(fd_path)
            self.assertEqual(len(scans), 3)
            self.assertTrue(scans[0].has_tumor)
            self.assertFalse(scans[1].has_tumor)

            with self.assertRaises(ScanSelectionRequiredError):
                load_matlab_dataset(fd_path)

            ds = load_dataset(fd_path, scan_index=0)
            self.assertEqual(ds.s_parameters.shape, (1001, 8))
            self.assertEqual(ds.n_frequencies, 1001)
            self.assertAlmostEqual(ds.freq_range_hz[0], 1e9, delta=1.0)
            self.assertAlmostEqual(ds.freq_range_hz[1], 8e9, delta=1.0)
            self.assertAlmostEqual(ds.metadata["antenna_radius_m"], 0.18, places=6)
            self.assertTrue(ds.metadata["bmid_has_tumor"])
            self.assertAlmostEqual(ds.metadata["tumor_x_m"], 0.015, places=6)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(DATASETS_DIR, "fd_data_s21_adi.mat"))
        and os.path.isfile(os.path.join(DATASETS_DIR, "md_list_s21_adi.mat")),
        "Real UM-BMID files not present",
    )
    def test_real_bmid_tumor_and_healthy(self):
        fd_path = os.path.join(DATASETS_DIR, "fd_data_s21_adi.mat")
        scans = list_bmid_scans(fd_path)
        self.assertEqual(len(scans), 200)
        tumor = next(s for s in scans if s.has_tumor)
        healthy = next(s for s in scans if not s.has_tumor)

        tumor_ds = load_bmid_scan(fd_path, tumor.index)
        healthy_ds = load_bmid_scan(fd_path, healthy.index)
        self.assertEqual(tumor_ds.s_parameters.shape, (1001, 72))
        self.assertEqual(healthy_ds.s_parameters.shape, (1001, 72))
        self.assertTrue(tumor_ds.metadata["bmid_has_tumor"])
        self.assertFalse(healthy_ds.metadata["bmid_has_tumor"])
        self.assertAlmostEqual(tumor_ds.metadata["antenna_radius_m"], 0.18, places=6)


if __name__ == "__main__":
    unittest.main()

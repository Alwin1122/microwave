"""Tests for session report export."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.dataset_info import MicrowaveDataset
from preprocessing.preprocessing_pipeline import (
    PreprocessingConfig,
    PreprocessingResult,
    SignalQualityReport,
)
from reconstruction.reconstruction_manager import ReconstructionConfig, ROIRefinement
from roi.roi_detector import ROIResult
from utils.session_report import (
    ReconstructionSnapshot,
    SessionReportContext,
    build_session_report,
    write_session_report,
)


class TestSessionReport(unittest.TestCase):
    def _dataset(self) -> MicrowaveDataset:
        freqs = np.linspace(1e9, 8e9, 21)
        s_params = (np.random.randn(21, 4) + 1j * np.random.randn(21, 4)) * 1e-3
        return MicrowaveDataset(
            file_path="datasets/fd_data_s21_adi.mat",
            file_name="fd_data_s21_adi.mat",
            file_type="MATLAB (UM-BMID)",
            frequencies=freqs,
            s_parameters=s_params,
            n_ports=4,
            metadata={
                "dataset_family": "UM-BMID",
                "bmid_has_tumor": True,
                "tumor_x_m": 0.0225,
                "tumor_y_m": 0.0225,
                "antenna_radius_m": 0.18,
            },
        )

    def test_build_and_write_report(self):
        dataset = self._dataset()
        preprocessing = PreprocessingResult(
            original_dataset=dataset,
            processed_dataset=dataset,
            stage_outputs={"Raw": dataset.s_parameters, "Filtered": dataset.s_parameters},
            quality_report=SignalQualityReport(
                snr_before_db=10.0,
                snr_after_db=12.0,
                dynamic_range_before_db=20.0,
                dynamic_range_after_db=22.0,
                mean_magnitude_before=0.01,
                mean_magnitude_after=0.009,
                std_magnitude_before=0.001,
                std_magnitude_after=0.0008,
                artifact_reduction_ratio=0.1,
            ),
            config=PreprocessingConfig(),
        )
        snap = ReconstructionSnapshot(
            selected_beamformer="DAS",
            source_label="Processed Dataset",
            config=ReconstructionConfig(antenna_radius=0.18, x_span=(-0.06, 0.06), y_span=(-0.06, 0.06)),
            quality_metrics={"DAS": {"snr": 1.0, "scr": 2.0, "contrast": 0.5, "time": 0.01, "score": 0.9}},
            roi=ROIResult(
                mask=np.zeros((8, 8), dtype=bool),
                bounding_box=(2, 2, 5, 5),
                centroid=(3.0, 3.0),
                area=9,
                threshold=0.7,
                score=1.2,
            ),
            refinement=ROIRefinement(
                algorithm="DAS",
                image=np.ones((16, 16)),
                x_span=(-0.02, 0.02),
                y_span=(-0.02, 0.02),
                grid_shape=(16, 16),
            ),
            tumor_xy_m=(0.0225, 0.0225),
            roi_centroid_m=(0.01, 0.01),
            tumor_gt_distance_m=0.0177,
            image_peak=1.0,
            image_mean=0.5,
        )
        markdown, payload = build_session_report(
            SessionReportContext(dataset=dataset, preprocessing=preprocessing, reconstruction=snap)
        )
        self.assertIn("Session Report", markdown)
        self.assertIn("Data quality checks", markdown)
        self.assertIn("Suggested next steps", markdown)
        self.assertIn("UM-BMID", markdown)

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_session_report(
                SessionReportContext(dataset=dataset, preprocessing=preprocessing, reconstruction=snap),
                output_dir=tmp,
                basename="unit_report",
            )
            self.assertTrue(os.path.isfile(paths["markdown"]))
            self.assertTrue(os.path.isfile(paths["json"]))
            with open(paths["json"], encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.assertIn("checks", loaded)
            self.assertIn("next_steps", loaded)


if __name__ == "__main__":
    unittest.main()

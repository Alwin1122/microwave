"""Generate a validation-rich session report from sample data.

This script runs preprocessing and reconstruction without GUI interaction,
then writes `results/latest_session_report.md` and `.json` with the
validation plots added by `utils.session_report.write_session_report`.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.loader import load_dataset
from preprocessing.preprocessing_pipeline import PreprocessingConfig, run_preprocessing_pipeline
from quality.beamformer_selector import select_best_beamformer
from reconstruction.reconstruction_manager import (
    ReconstructionConfig,
    reconstruct_all,
    reconstruct_high_resolution_roi,
)
from roi.roi_detector import detect_roi, roi_centroid_meters
from utils.session_report import ReconstructionSnapshot, SessionReportContext, write_session_report


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(root, "datasets", "sample_touchstone.s2p")
    results_dir = os.path.abspath(os.path.join(root, "..", "results"))

    dataset = load_dataset(dataset_path)
    prep = run_preprocessing_pipeline(
        dataset,
        PreprocessingConfig(
            filter_method="savgol",
            filter_kwargs={"window_length": 7, "polyorder": 3},
            enable_week3=True,
            enable_group_clutter_removal=True,
            enable_global_normalize=True,
        ),
    )

    processed = prep.processed_dataset
    recon_config = ReconstructionConfig()
    images, timings = reconstruct_all(
        processed.s_parameters,
        processed.frequencies,
        config=recon_config,
        return_timings=True,
    )
    selected, quality = select_best_beamformer(images, timings, mode="prefer_dmas_d4")
    roi = detect_roi(images[selected], prefer_off_center=False, tight_peak=False)
    refinement = reconstruct_high_resolution_roi(
        processed.s_parameters,
        processed.frequencies,
        selected,
        roi.bounding_box,
        images[selected].shape,
        min_grid_size=96,
    )
    centroid_m = roi_centroid_meters(
        roi,
        images[selected].shape,
        (-0.05, 0.05),
        (-0.05, 0.05),
    )

    snap = ReconstructionSnapshot(
        selected_beamformer=selected,
        source_label="Processed Dataset",
        config=recon_config,
        quality_metrics=quality,
        roi=roi,
        refinement=refinement,
        tumor_xy_m=None,
        roi_centroid_m=centroid_m,
        tumor_gt_distance_m=None,
        image_peak=float(np.max(images[selected])),
        image_mean=float(np.mean(images[selected])),
        selection_mode="prefer_dmas_d4",
        prefer_off_center_roi=False,
        selected_image=np.asarray(images[selected]).copy(),
        beamformer_images={name: np.asarray(image).copy() for name, image in images.items()},
    )

    ctx = SessionReportContext(dataset=dataset, preprocessing=prep, reconstruction=snap)
    paths = write_session_report(ctx, output_dir=results_dir, basename="latest_session_report")
    print(paths["markdown"])
    print(paths["json"])


if __name__ == "__main__":
    main()

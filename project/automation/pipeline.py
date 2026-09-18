"""Per-dataset load → preprocess → reconstruct → ROI evidence collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from automation.classifier import classify_evidence
from automation.figures import write_case_figures
from automation.policy import ThresholdProfile
from automation.types import DatasetValidationResult, DatasetVerdict, ResolvedTarget
from data_loader.loader import load_dataset
from preprocessing.preprocessing_pipeline import PreprocessingConfig, run_preprocessing_pipeline
from quality.beamformer_selector import select_best_beamformer
from quality.characterization import characterize_tumor_region
from quality.confidence import assess_reconstruction_confidence
from reconstruction.reconstruction_manager import (
    ReconstructionConfig,
    infer_reconstruction_config,
    reconstruct_all,
    reconstruct_high_resolution_roi,
)
from roi.roi_detector import detect_rois, evaluate_rois
from utils.exceptions import MicrowaveFrameworkError

ProgressCb = Callable[[str], None]


def _is_bmid(dataset) -> bool:
    meta = dataset.metadata or {}
    return str(meta.get("dataset_family", "")).upper() == "UM-BMID"


def _preprocess_config_for(dataset) -> PreprocessingConfig:
    if _is_bmid(dataset):
        return PreprocessingConfig(
            filter_method="none",
            calibration_method="none",
            normalization_method="none",
            artifact_method="none",
            do_background_subtraction=False,
            enable_week3=False,
            enable_group_clutter_removal=False,
            enable_global_normalize=False,
        )
    return PreprocessingConfig(
        filter_method="savgol",
        filter_kwargs={"window_length": 7, "polyorder": 3},
        enable_week3=True,
        enable_group_clutter_removal=True,
        enable_global_normalize=True,
    )


def _recon_config_for(dataset) -> ReconstructionConfig:
    config = infer_reconstruction_config(dataset)
    # Keep automation runs fast and comparable across files.
    config.n_x = 64
    config.n_y = 64
    if _is_bmid(dataset):
        config.use_bmid_phase_delay_radius = False
        config.x_span = (-0.06, 0.06)
        config.y_span = (-0.06, 0.06)
    return config


def validate_one_target(
    target: ResolvedTarget,
    *,
    profile: ThresholdProfile,
    include_reconstruction_checks: bool = True,
    progress: ProgressCb | None = None,
    figure_dir: str | None = None,
) -> DatasetValidationResult:
    def log(msg: str) -> None:
        if progress:
            progress(msg)

    details: dict[str, Any] = {"source_note": target.source_note}

    try:
        log(f"Loading {target.measurement_path} (scan={target.scan_index})")
        dataset = load_dataset(target.measurement_path, scan_index=target.scan_index)
    except MicrowaveFrameworkError as exc:
        status, metrics, reasons, remediation, recommended = classify_evidence(
            profile=profile,
            load_ok=False,
            load_error=str(exc),
            finite_ok=None,
            energy=None,
            roi_area=None,
            localization_error_cm=None,
            scr=None,
            snr=None,
            confidence=None,
            is_tumor_candidate=None,
            has_tumor_gt=False,
        )
        return DatasetValidationResult(
            target=target,
            status=status,
            reasons=reasons,
            remediation=remediation,
            metrics=metrics,
            recommended_upload=recommended,
            details={"error": str(exc)},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return DatasetValidationResult(
            target=target,
            status=DatasetVerdict.FAIL,
            reasons=[f"Unexpected load error: {exc}"],
            remediation=["Inspect the file format and stack trace in logs."],
            recommended_upload=False,
            details={"error": str(exc)},
        )

    meta = dataset.metadata or {}
    has_tumor_gt = bool(meta.get("bmid_has_tumor"))
    tumor_xy = None
    if meta.get("tumor_x_m") is not None and meta.get("tumor_y_m") is not None:
        tumor_xy = (float(meta["tumor_x_m"]), float(meta["tumor_y_m"]))
    details["dataset_family"] = meta.get("dataset_family")
    details["file_name"] = dataset.file_name
    details["scan_label"] = target.scan_label
    details["has_tumor_gt"] = has_tumor_gt
    details["tumor_gt_x_cm"] = None if tumor_xy is None else tumor_xy[0] * 100.0
    details["tumor_gt_y_cm"] = None if tumor_xy is None else tumor_xy[1] * 100.0
    from quality.tumor_taxonomy import taxonomy_from_metadata

    details["tumor_taxonomy"] = taxonomy_from_metadata(meta).to_dict()

    if not include_reconstruction_checks:
        s_params = np.asarray(dataset.s_parameters)
        finite_ok = bool(np.all(np.isfinite(s_params.real)) and np.all(np.isfinite(s_params.imag)))
        energy = float(np.mean(np.abs(s_params) ** 2)) if s_params.size else 0.0
        status, metrics, reasons, remediation, recommended = classify_evidence(
            profile=profile,
            load_ok=True,
            load_error=None,
            finite_ok=finite_ok,
            energy=energy,
            roi_area=profile.min_roi_area,
            localization_error_cm=None,
            scr=None,
            snr=None,
            confidence=None,
            is_tumor_candidate=None,
            has_tumor_gt=has_tumor_gt,
        )
        details["mode"] = "load_only"
        return DatasetValidationResult(
            target=target,
            status=status,
            reasons=reasons,
            remediation=remediation,
            metrics=metrics,
            recommended_upload=recommended,
            details=details,
        )

    try:
        log("Preprocessing…")
        prep = run_preprocessing_pipeline(dataset, _preprocess_config_for(dataset))
        processed = prep.processed_dataset
        pre_cfg = prep.config
        pre_steps = list(prep.stage_outputs.keys())
        pre_summary = {
            "steps": pre_steps,
            "filter_method": pre_cfg.filter_method,
            "calibration_method": pre_cfg.calibration_method,
            "normalization_method": pre_cfg.normalization_method,
            "artifact_method": pre_cfg.artifact_method,
            "background_subtraction": bool(pre_cfg.do_background_subtraction),
            "week3_enabled": bool(pre_cfg.enable_week3),
            "snr_before_db": float(prep.quality_report.snr_before_db),
            "snr_after_db": float(prep.quality_report.snr_after_db),
            "dynamic_range_before_db": float(prep.quality_report.dynamic_range_before_db),
            "dynamic_range_after_db": float(prep.quality_report.dynamic_range_after_db),
        }
        config = _recon_config_for(processed)
        log("Reconstructing DAS/DMAS/DMAS-D4…")
        images, timings = reconstruct_all(
            processed.s_parameters,
            processed.frequencies,
            config=config,
            return_timings=True,
        )
        mode = "tumor_gt" if tumor_xy is not None else "quality"
        selected, quality = select_best_beamformer(
            images,
            timings,
            mode=mode,
            x_span=config.x_span,
            y_span=config.y_span,
            tumor_xy_m=tumor_xy,
            prefer_off_center=False,
            tight_peak=False,
        )
        image = np.asarray(images[selected], dtype=float)
        log("Detecting ROI spots and scoring…")
        rois = detect_rois(
            image,
            prefer_off_center=False,
            tight_peak=False,
            x_span=config.x_span,
            y_span=config.y_span,
            prior_xy_m=tumor_xy,
            prior_weight=0.0,
            max_rois=4,
            min_score_ratio=0.28,
        )
        evaluated = evaluate_rois(
            image,
            rois,
            config.x_span,
            config.y_span,
            tumor_xy_m=tumor_xy,
        )
        roi = evaluated.localization_roi
        refinement = reconstruct_high_resolution_roi(
            processed.s_parameters,
            processed.frequencies,
            selected,
            roi.bounding_box,
            image.shape,
            config=config,
            min_grid_size=96,
        )
        centroid_m = evaluated.localization_centroid_m
        gt_dist_cm = evaluated.gt_distance_cm
        candidate = evaluated.combined
        char = characterize_tumor_region(
            image, roi, x_span=config.x_span, y_span=config.y_span, centroid_m=centroid_m
        )
        cand_payload = {
            "is_tumor_candidate": candidate.is_tumor_candidate,
            "confidence": candidate.confidence,
            "suspicion_score": candidate.suspicion_score,
            "reason": candidate.reason,
            "n_rois": len(evaluated.rois),
            "spots": evaluated.spot_summaries,
        }
        conf = assess_reconstruction_confidence(
            selected_quality=quality.get(selected),
            all_quality_metrics=quality,
            selected_beamformer=selected,
            tumor_candidate=cand_payload,
            characterization=char.to_dict(),
            tumor_gt_distance_m=None if gt_dist_cm is None else gt_dist_cm / 100.0,
        )
        selected_metrics = quality.get(selected, {})
        finite_ok = bool(np.all(np.isfinite(image)))
        energy = float(np.mean(image**2))
        recon_summary = {
            "beamformer": selected,
            "image_peak": float(np.max(image)),
            "image_mean": float(np.mean(image)),
            "image_std": float(np.std(image)),
            "image_energy": energy,
            "roi_area_px": int(roi.area),
            "n_rois": len(evaluated.rois),
            "centroid_x_cm": float(centroid_m[0] * 100.0),
            "centroid_y_cm": float(centroid_m[1] * 100.0),
        }
        status, metrics, reasons, remediation, recommended = classify_evidence(
            profile=profile,
            load_ok=True,
            load_error=None,
            finite_ok=finite_ok,
            energy=energy,
            roi_area=int(roi.area),
            localization_error_cm=gt_dist_cm,
            scr=float(selected_metrics.get("scr")) if selected_metrics.get("scr") is not None else None,
            snr=float(selected_metrics.get("snr")) if selected_metrics.get("snr") is not None else None,
            confidence=float(conf.overall),
            is_tumor_candidate=bool(candidate.is_tumor_candidate),
            has_tumor_gt=has_tumor_gt,
        )
        figure_paths: dict[str, str] = {}
        if figure_dir:
            scan_tag = "na" if target.scan_index is None else str(target.scan_index)
            stem = f"{Path(target.measurement_path).stem}_scan{scan_tag}"
            try:
                figure_paths = write_case_figures(
                    figure_dir,
                    stem,
                    images=images,
                    selected=selected,
                    roi_boxes=[item.bounding_box for item in evaluated.rois],
                    x_span=config.x_span,
                    y_span=config.y_span,
                    refined_image=refinement.image,
                )
            except Exception:
                figure_paths = {}
        details.update(
            {
                "selected_beamformer": selected,
                "roi_area": int(roi.area),
                "n_rois": len(evaluated.rois),
                "roi_spots": evaluated.spot_summaries,
                "localization_error_cm": gt_dist_cm,
                "tumor_candidate": cand_payload,
                "characterization": char.to_dict(),
                "confidence": conf.to_dict(),
                "preprocessing": pre_summary,
                "reconstruction": recon_summary,
                "refinement_algorithm": refinement.algorithm,
                "quality": {k: dict(v) for k, v in quality.items()},
                "figures": figure_paths,
            }
        )
        return DatasetValidationResult(
            target=target,
            status=status,
            reasons=reasons,
            remediation=remediation,
            metrics=metrics,
            recommended_upload=recommended,
            details=details,
        )
    except MicrowaveFrameworkError as exc:
        return DatasetValidationResult(
            target=target,
            status=DatasetVerdict.FAIL,
            reasons=[f"Pipeline failed: {exc}"],
            remediation=["Inspect preprocessing/reconstruction settings for this file type."],
            recommended_upload=False,
            details={"error": str(exc), **details},
        )
    except Exception as exc:  # pragma: no cover
        return DatasetValidationResult(
            target=target,
            status=DatasetVerdict.FAIL,
            reasons=[f"Unexpected pipeline error: {exc}"],
            remediation=["Check logs and re-run with lenient profile for diagnosis."],
            recommended_upload=False,
            details={"error": str(exc), **details},
        )

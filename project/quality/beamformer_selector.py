"""Automatic selection of the best beamformer based on image quality."""

from __future__ import annotations

import numpy as np

from quality.metrics import compute_quality_metrics
from roi.roi_detector import detect_roi, roi_centroid_meters


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    return float((value - min_val) / (max_val - min_val))


def select_best_beamformer(
    images: dict[str, np.ndarray],
    timings: dict[str, float],
    weights: dict[str, float] | None = None,
    mode: str = "quality",
    x_span: tuple[float, float] | None = None,
    y_span: tuple[float, float] | None = None,
    tumor_xy_m: tuple[float, float] | None = None,
    prefer_off_center: bool = True,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Select the best beamformer using quality metrics and optional GT distance.

    Modes:
        quality: weighted SNR / SCR / contrast / time (default).
        prefer_dmas_d4: force DMAS-D4 when present; still report quality scores.
        tumor_gt: pick the image whose ROI is closest to tumor ground truth.
    """
    weights = weights or {
        "snr": 0.35,
        "scr": 0.35,
        "contrast": 0.2,
        "time": 0.1,
    }

    metrics = {name: compute_quality_metrics(image) for name, image in images.items()}
    snr_vals = [m["snr"] for m in metrics.values()]
    scr_vals = [m["scr"] for m in metrics.values()]
    contrast_vals = [m["contrast"] for m in metrics.values()]
    time_vals = [timings.get(name, 0.0) for name in images.keys()]

    score_map: dict[str, float] = {}
    for name, m in metrics.items():
        snr_norm = _normalize(m["snr"], min(snr_vals), max(snr_vals))
        scr_norm = _normalize(m["scr"], min(scr_vals), max(scr_vals))
        contrast_norm = _normalize(m["contrast"], min(contrast_vals), max(contrast_vals))
        time_norm = 1.0 - _normalize(timings.get(name, 0.0), min(time_vals), max(time_vals))
        score_map[name] = (
            weights["snr"] * snr_norm
            + weights["scr"] * scr_norm
            + weights["contrast"] * contrast_norm
            + weights["time"] * time_norm
        )
        metrics[name]["score"] = float(score_map[name])
        metrics[name]["time"] = float(timings.get(name, 0.0))

    mode = (mode or "quality").strip().lower()
    if mode in {"prefer_dmas_d4", "dmas_d4", "dmas-d4"} and "DMAS-D4" in images:
        selected = "DMAS-D4"
        metrics[selected]["selection_mode"] = "prefer_dmas_d4"
        return selected, metrics

    force_map = {
        "force_das": "DAS",
        "das": "DAS",
        "force_dmas": "DMAS",
        "dmas": "DMAS",
        "force_dmas_d4": "DMAS-D4",
        "force_dmas-d4": "DMAS-D4",
    }
    if mode in force_map and force_map[mode] in images:
        selected = force_map[mode]
        metrics[selected]["selection_mode"] = mode
        return selected, metrics

    if mode in {"tumor_gt", "gt", "closest_to_tumor"} and tumor_xy_m is not None and x_span and y_span:
        best_name = None
        best_dist = float("inf")
        for name, image in images.items():
            roi = detect_roi(
                image,
                prefer_off_center=prefer_off_center,
                x_span=x_span,
                y_span=y_span,
                prior_xy_m=None,
            )
            centroid_m = roi_centroid_meters(roi, image.shape, x_span, y_span)
            dist = float(np.hypot(centroid_m[0] - tumor_xy_m[0], centroid_m[1] - tumor_xy_m[1]))
            metrics[name]["tumor_gt_distance_m"] = dist
            if dist < best_dist:
                best_dist = dist
                best_name = name
        if best_name is not None:
            metrics[best_name]["selection_mode"] = "tumor_gt"
            return best_name, metrics

    selected = max(score_map, key=score_map.get)
    metrics[selected]["selection_mode"] = "quality"
    return selected, metrics

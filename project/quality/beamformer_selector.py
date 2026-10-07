"""Automatic selection of the best beamformer based on image quality."""

from __future__ import annotations

import numpy as np

from quality.metrics import compute_quality_metrics
from roi.roi_detector import detect_rois, roi_centroid_meters


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
    tight_peak: bool = False,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Select the best beamformer using quality metrics and optional GT distance.

    Modes:
        quality: weighted SNR / SCR / contrast / time (default).
        prefer_dmas_d4: force DMAS-D4 when present; still report quality scores.
        tumor_gt: lab-only. Pick the image whose ROI is closest to a known
        label. Do not use this as the default detector — it peeks at the answer.
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
        metrics[name]["selected"] = 0.0
        metrics[name]["selection_reason"] = ""

    def _finish(selected: str, mode_name: str, reason: str):
        for name in metrics:
            metrics[name]["selection_mode"] = mode_name
            metrics[name]["selected"] = 1.0 if name == selected else 0.0
            metrics[name]["selection_reason"] = reason if name == selected else "not selected"
        return selected, metrics

    mode = (mode or "quality").strip().lower()
    if mode in {"prefer_dmas_d4", "dmas_d4", "dmas-d4"} and "DMAS-D4" in images:
        winner = score_map and max(score_map, key=score_map.get)
        extra = (
            f" Quality winner would be {winner} ({score_map[winner]:.3f})."
            if winner and winner != "DMAS-D4"
            else ""
        )
        return _finish(
            "DMAS-D4",
            "prefer_dmas_d4",
            "Prefer DMAS-D4 mode: DMAS-D4 is chosen even if another score is higher."
            + extra,
        )

    force_map = {
        "force_das": "DAS",
        "das": "DAS",
        "force_dmas": "DMAS",
        "dmas": "DMAS",
        "force_dmas_d4": "DMAS-D4",
        "force_dmas-d4": "DMAS-D4",
    }
    if mode in force_map and force_map[mode] in images:
        chosen = force_map[mode]
        return _finish(
            chosen,
            mode,
            f"Forced by user setting ({mode}). Scores are reported but not used to pick.",
        )

    if mode in {"tumor_gt", "gt", "closest_to_tumor"} and tumor_xy_m is not None and x_span and y_span:
        best_name = None
        best_dist = float("inf")
        for name, image in images.items():
            rois = detect_rois(
                image,
                prefer_off_center=prefer_off_center,
                tight_peak=tight_peak,
                x_span=x_span,
                y_span=y_span,
                prior_xy_m=tumor_xy_m,
                prior_weight=0.0,
                max_rois=4,
            )
            dists = []
            for roi in rois:
                centroid_m = roi_centroid_meters(roi, image.shape, x_span, y_span)
                dists.append(
                    float(np.hypot(centroid_m[0] - tumor_xy_m[0], centroid_m[1] - tumor_xy_m[1]))
                )
            dist = min(dists) if dists else float("inf")
            metrics[name]["tumor_gt_distance_m"] = dist
            if dist < best_dist:
                best_dist = dist
                best_name = name
        if best_name is not None:
            return _finish(
                best_name,
                "tumor_gt",
                f"Closest ROI to tumor GT ({best_dist * 100:.2f} cm). "
                "Quality scores are shown for comparison, not used as the pick rule.",
            )

    selected = max(score_map, key=score_map.get)
    return _finish(
        selected,
        "quality",
        (
            f"Highest weighted score {score_map[selected]:.3f} "
            f"(SNR {weights['snr']:.2f}, SCR {weights['scr']:.2f}, "
            f"contrast {weights['contrast']:.2f}, time {weights['time']:.2f}; "
            "each metric min-max normalized across DAS/DMAS/DMAS-D4)."
        ),
    )

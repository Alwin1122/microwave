"""Automatic selection of the best beamformer based on image quality."""

from __future__ import annotations

import numpy as np

from quality.metrics import compute_quality_metrics


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    return float((value - min_val) / (max_val - min_val))


def select_best_beamformer(
    images: dict[str, np.ndarray],
    timings: dict[str, float],
    weights: dict[str, float] | None = None,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Select the best beamformer using normalized quality metrics and timing."""
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
    for idx, (name, m) in enumerate(metrics.items()):
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

    selected = max(score_map, key=score_map.get)
    return selected, metrics

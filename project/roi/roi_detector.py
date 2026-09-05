"""ROI localization utilities for reconstructed microwave images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_opening, find_objects, gaussian_filter, label


@dataclass
class ROIResult:
    """Detected ROI summary."""

    mask: np.ndarray
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    threshold: float
    score: float
    mode: str = "peak"


_EPS = 1e-12


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    span = max(max_val - min_val, _EPS)
    return (image - min_val) / span


def _pixel_to_meters(
    cx: float,
    cy: float,
    shape: tuple[int, int],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
) -> tuple[float, float]:
    ny, nx = shape
    if nx <= 1:
        x_m = float(x_span[0])
    else:
        x_m = float(np.interp(cx, [0, nx - 1], [x_span[0], x_span[1]]))
    if ny <= 1:
        y_m = float(y_span[0])
    else:
        y_m = float(np.interp(cy, [0, ny - 1], [y_span[0], y_span[1]]))
    return x_m, y_m


def detect_roi(
    image: np.ndarray,
    threshold_ratio: float = 0.7,
    min_area: int = 6,
    margin: int = 3,
    sigma: float = 1.0,
    prefer_off_center: bool = False,
    off_center_weight: float = 1.5,
    tight_peak: bool = False,
    x_span: tuple[float, float] | None = None,
    y_span: tuple[float, float] | None = None,
    prior_xy_m: tuple[float, float] | None = None,
    prior_weight: float = 2.0,
) -> ROIResult:
    """Detect the most suspicious ROI in a reconstruction image.

    Args:
        image: 2D reconstruction image.
        threshold_ratio: Threshold on normalized intensity in [0, 1].
        min_area: Minimum connected-component area to accept.
        margin: Extra pixels to add around the selected bounding box.
        sigma: Gaussian smoothing parameter.
        prefer_off_center: Boost blobs away from the image center.
        off_center_weight: Strength of the off-center boost (>= 0).
        tight_peak: If True, localize to the intensity peak inside the chosen
            region and shrink the box (avoids large-blob centroid bias).
        x_span / y_span: Physical extents in meters for prior scoring.
        prior_xy_m: Optional prior location (e.g. tumor GT) in meters.
        prior_weight: Strength of the prior proximity boost.

    Returns:
        ROIResult for the most suspicious region.
    """
    if image.ndim != 2:
        raise ValueError("ROI detection expects a 2D reconstruction image.")

    mode = "tight" if tight_peak else ("off_center" if prefer_off_center else "peak")
    # Tight peak keeps the same blob selection as peak mode, then snaps the
    # centroid to the intensity maximum inside that blob and shrinks the box.
    effective_threshold = threshold_ratio
    if tight_peak:
        margin = min(margin, 2)
        sigma = min(sigma, 1.0)

    normalized = _normalize_image(image)
    smoothed = gaussian_filter(normalized, sigma=sigma)
    threshold = float(np.clip(effective_threshold, 0.0, 1.0))
    binary = smoothed >= threshold
    binary = binary_opening(binary, structure=np.ones((3, 3), dtype=bool))

    labeled, num = label(binary)
    if num == 0:
        peak_index = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
        y, x = peak_index
        half = 2 if tight_peak else margin
        y0 = max(0, y - half)
        y1 = min(smoothed.shape[0], y + half + 1)
        x0 = max(0, x - half)
        x1 = min(smoothed.shape[1], x + half + 1)
        mask = np.zeros_like(binary, dtype=bool)
        mask[y0:y1, x0:x1] = True
        area = int(np.sum(mask))
        return ROIResult(
            mask=mask,
            bounding_box=(x0, y0, x1, y1),
            centroid=(float(x), float(y)),
            area=area,
            threshold=threshold,
            score=float(smoothed[peak_index]),
            mode=mode,
        )

    objects = find_objects(labeled)
    best_score = -1.0
    best_mask: np.ndarray | None = None
    best_box: tuple[int, int, int, int] | None = None
    best_centroid = (0.0, 0.0)
    best_area = 0
    best_component_id = -1

    ny, nx = smoothed.shape
    center_x = (nx - 1) / 2.0
    center_y = (ny - 1) / 2.0
    max_r = float(np.hypot(center_x, center_y)) + _EPS
    half_diag_m = None
    if x_span is not None and y_span is not None:
        half_diag_m = float(
            np.hypot(
                0.5 * abs(x_span[1] - x_span[0]),
                0.5 * abs(y_span[1] - y_span[0]),
            )
        ) + _EPS

    for component_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        component_mask = labeled[slc] == component_id
        area = int(np.sum(component_mask))
        if area < min_area:
            continue
        component_values = smoothed[slc][component_mask]
        peak_val = float(np.max(component_values))
        # Component ranking stays brightness×extent; tight only changes localization.
        score = float(np.mean(component_values) * peak_val * area)

        ys, xs = np.where(labeled == component_id)
        weights = smoothed[ys, xs]
        wsum = float(np.sum(weights)) + _EPS
        cx = float(np.sum(xs * weights) / wsum)
        cy = float(np.sum(ys * weights) / wsum)
        if tight_peak:
            local = np.zeros_like(smoothed)
            local[ys, xs] = smoothed[ys, xs]
            py, px = np.unravel_index(int(np.argmax(local)), local.shape)
            cx, cy = float(px), float(py)

        if prefer_off_center:
            r_norm = float(np.hypot(cx - center_x, cy - center_y) / max_r)
            center_penalty = 0.2 + 0.8 * r_norm
            score *= center_penalty * (1.0 + float(off_center_weight) * r_norm)

        if prior_xy_m is not None and x_span is not None and y_span is not None and half_diag_m is not None:
            x_m, y_m = _pixel_to_meters(cx, cy, smoothed.shape, x_span, y_span)
            dist = float(np.hypot(x_m - prior_xy_m[0], y_m - prior_xy_m[1]))
            proximity = 1.0 - min(dist / half_diag_m, 1.0)
            score *= 1.0 + float(prior_weight) * proximity

        if score <= best_score:
            continue

        if tight_peak:
            half = max(2, margin)
            y0 = max(0, int(round(cy)) - half)
            y1 = min(smoothed.shape[0], int(round(cy)) + half + 1)
            x0 = max(0, int(round(cx)) - half)
            x1 = min(smoothed.shape[1], int(round(cx)) + half + 1)
            full_mask = np.zeros_like(binary, dtype=bool)
            full_mask[y0:y1, x0:x1] = True
            best_area = int(np.sum(full_mask))
        else:
            y_slice, x_slice = slc
            y0 = max(0, y_slice.start - margin)
            y1 = min(smoothed.shape[0], y_slice.stop + margin)
            x0 = max(0, x_slice.start - margin)
            x1 = min(smoothed.shape[1], x_slice.stop + margin)
            full_mask = np.zeros_like(binary, dtype=bool)
            full_mask[y0:y1, x0:x1] = True
            best_area = area

        best_score = score
        best_mask = full_mask
        best_box = (x0, y0, x1, y1)
        best_centroid = (cx, cy)
        best_component_id = component_id

    if best_mask is None or best_box is None:
        return detect_roi(
            image,
            threshold_ratio=min(threshold + 0.05, 0.95),
            min_area=1,
            margin=margin,
            sigma=sigma,
            prefer_off_center=prefer_off_center,
            off_center_weight=off_center_weight,
            tight_peak=tight_peak,
            x_span=x_span,
            y_span=y_span,
            prior_xy_m=prior_xy_m,
            prior_weight=prior_weight,
        )

    del best_component_id
    return ROIResult(
        mask=best_mask,
        bounding_box=best_box,
        centroid=best_centroid,
        area=best_area,
        threshold=threshold,
        score=best_score,
        mode=mode,
    )


def roi_centroid_meters(
    roi: ROIResult,
    image_shape: tuple[int, int],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
) -> tuple[float, float]:
    """Convert an ROI pixel centroid to meters using the reconstruction FOV."""
    return _pixel_to_meters(roi.centroid[0], roi.centroid[1], image_shape, x_span, y_span)


def tumor_likelihood_features(
    image: np.ndarray,
    roi: ROIResult,
) -> dict[str, float]:
    """Features that tend to separate healthy center-clutter from tumor foci.

    On BMID peak-ROI runs, healthy scans often show a large near-origin blob
    (high area, low r_norm). Tumor foci are usually smaller and more off-center.
    """
    ny, nx = image.shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    r_norm = float(np.hypot(roi.centroid[0] - cx, roi.centroid[1] - cy) / (np.hypot(cx, cy) + _EPS))
    peak = float(np.max(image))
    mean = float(np.mean(image))
    area = float(roi.area)
    # Higher => more tumor-like: off-center and not a FOV-filling clutter sheet.
    suspicion = float(r_norm * (1.0 / (1.0 + area / 800.0)) * (peak / (mean + _EPS)))
    return {
        "r_norm": r_norm,
        "roi_area": area,
        "peak_over_mean": peak / (mean + _EPS),
        "compactness": float(peak / (area + 1.0)),
        "suspicion": suspicion,
    }

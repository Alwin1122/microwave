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


_EPS = 1e-12


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    span = max(max_val - min_val, _EPS)
    return (image - min_val) / span


def detect_roi(
    image: np.ndarray,
    threshold_ratio: float = 0.7,
    min_area: int = 6,
    margin: int = 3,
    sigma: float = 1.0,
) -> ROIResult:
    """Detect the most suspicious ROI in a reconstruction image.

    Args:
        image: 2D reconstruction image.
        threshold_ratio: Threshold on normalized intensity in [0, 1].
        min_area: Minimum connected-component area to accept.
        margin: Extra pixels to add around the selected bounding box.
        sigma: Gaussian smoothing parameter.

    Returns:
        ROIResult for the most suspicious region.
    """
    if image.ndim != 2:
        raise ValueError("ROI detection expects a 2D reconstruction image.")

    normalized = _normalize_image(image)
    smoothed = gaussian_filter(normalized, sigma=sigma)
    threshold = float(np.clip(threshold_ratio, 0.0, 1.0))
    binary = smoothed >= threshold
    binary = binary_opening(binary, structure=np.ones((3, 3), dtype=bool))

    labeled, num = label(binary)
    if num == 0:
        peak_index = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
        y, x = peak_index
        y0 = max(0, y - margin)
        y1 = min(smoothed.shape[0], y + margin + 1)
        x0 = max(0, x - margin)
        x1 = min(smoothed.shape[1], x + margin + 1)
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
        )

    objects = find_objects(labeled)
    best_score = -1.0
    best_mask: np.ndarray | None = None
    best_box: tuple[int, int, int, int] | None = None
    best_centroid = (0.0, 0.0)
    best_area = 0

    for component_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        component_mask = labeled[slc] == component_id
        area = int(np.sum(component_mask))
        if area < min_area:
            continue
        component_values = smoothed[slc][component_mask]
        score = float(np.mean(component_values) * np.max(component_values) * area)
        if score <= best_score:
            continue

        y_slice, x_slice = slc
        y0 = max(0, y_slice.start - margin)
        y1 = min(smoothed.shape[0], y_slice.stop + margin)
        x0 = max(0, x_slice.start - margin)
        x1 = min(smoothed.shape[1], x_slice.stop + margin)
        full_mask = np.zeros_like(binary, dtype=bool)
        full_mask[y0:y1, x0:x1] = True

        ys, xs = np.where(labeled == component_id)
        best_score = score
        best_mask = full_mask
        best_box = (x0, y0, x1, y1)
        best_centroid = (float(np.mean(xs)), float(np.mean(ys)))
        best_area = area

    if best_mask is None or best_box is None:
        return detect_roi(image, threshold_ratio=min(threshold + 0.05, 0.95), min_area=1, margin=margin, sigma=sigma)

    return ROIResult(
        mask=best_mask,
        bounding_box=best_box,
        centroid=best_centroid,
        area=best_area,
        threshold=threshold,
        score=best_score,
    )

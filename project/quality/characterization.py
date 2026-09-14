"""Module 9 — quantitative tumor / ROI characterization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from quality.metrics import compute_image_fwhm_pixels, compute_image_scr
from roi.roi_detector import ROIResult

_EPS = 1e-12


@dataclass
class TumorCharacterization:
    """Physical and intensity descriptors for a localized suspicious region."""

    centroid_x_cm: float
    centroid_y_cm: float
    radial_offset_cm: float
    equivalent_diameter_cm: float
    bbox_width_cm: float
    bbox_height_cm: float
    aspect_ratio: float
    area_cm2: float
    peak_intensity: float
    mean_intensity: float
    peak_to_mean: float
    local_scr: float
    fwhm_cm: float
    compactness: float
    eccentricity_proxy: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _pixel_scales_cm(
    image_shape: tuple[int, int],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
) -> tuple[float, float]:
    ny, nx = image_shape
    px_cm_x = abs(float(x_span[1] - x_span[0])) * 100.0 / max(nx - 1, 1)
    px_cm_y = abs(float(y_span[1] - y_span[0])) * 100.0 / max(ny - 1, 1)
    return px_cm_x, px_cm_y


def characterize_tumor_region(
    image: np.ndarray,
    roi: ROIResult,
    *,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    centroid_m: tuple[float, float] | None = None,
) -> TumorCharacterization:
    """
    Derive quantitative descriptors from the selected reconstruction and ROI.

    Uses physical FOV spans so size metrics are reported in centimeters.
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("image must be a non-empty 2-D array")

    ny, nx = arr.shape
    px_cm_x, px_cm_y = _pixel_scales_cm(arr.shape, x_span, y_span)
    px_cm = 0.5 * (px_cm_x + px_cm_y)

    if centroid_m is None:
        cx_m = float(x_span[0] + (roi.centroid[0] / max(nx - 1, 1)) * (x_span[1] - x_span[0]))
        cy_m = float(y_span[0] + (roi.centroid[1] / max(ny - 1, 1)) * (y_span[1] - y_span[0]))
    else:
        cx_m, cy_m = float(centroid_m[0]), float(centroid_m[1])

    x0, y0, x1, y1 = roi.bounding_box
    bbox_w_px = max(1, int(x1 - x0))
    bbox_h_px = max(1, int(y1 - y0))
    bbox_w_cm = float(bbox_w_px * px_cm_x)
    bbox_h_cm = float(bbox_h_px * px_cm_y)
    aspect = float(max(bbox_w_cm, bbox_h_cm) / (min(bbox_w_cm, bbox_h_cm) + _EPS))

    area_px = float(max(roi.area, 1))
    area_cm2 = float(area_px * px_cm_x * px_cm_y)
    eq_diameter_cm = float(2.0 * np.sqrt(area_cm2 / np.pi))

    mask = np.asarray(roi.mask, dtype=bool)
    if mask.shape != arr.shape or not mask.any():
        roi_vals = arr[y0:y1, x0:x1].ravel()
    else:
        roi_vals = arr[mask]

    if roi_vals.size == 0:
        roi_vals = np.asarray([float(np.max(arr))], dtype=float)

    peak = float(np.max(roi_vals))
    mean = float(np.mean(roi_vals))
    peak_to_mean = float(peak / (mean + _EPS))

    local_scr = float(compute_image_scr(arr[y0:y1, x0:x1])) if (y1 > y0 and x1 > x0) else 0.0
    fwhm_cm = float(compute_image_fwhm_pixels(arr) * px_cm)
    compactness = float(area_px / (bbox_w_px * bbox_h_px + _EPS))
    eccentricity = float(np.clip((aspect - 1.0) / (aspect + 1.0), 0.0, 1.0))
    radial_offset_cm = float(np.hypot(cx_m, cy_m) * 100.0)

    return TumorCharacterization(
        centroid_x_cm=float(cx_m * 100.0),
        centroid_y_cm=float(cy_m * 100.0),
        radial_offset_cm=radial_offset_cm,
        equivalent_diameter_cm=eq_diameter_cm,
        bbox_width_cm=bbox_w_cm,
        bbox_height_cm=bbox_h_cm,
        aspect_ratio=aspect,
        area_cm2=area_cm2,
        peak_intensity=peak,
        mean_intensity=mean,
        peak_to_mean=peak_to_mean,
        local_scr=local_scr,
        fwhm_cm=fwhm_cm,
        compactness=compactness,
        eccentricity_proxy=eccentricity,
    )

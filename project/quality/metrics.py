"""Quality metrics for reconstructed images."""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


def compute_image_snr(image: np.ndarray) -> float:
    """Estimate an image SNR from mean and standard deviation."""
    mean_val = np.mean(image)
    std_val = np.std(image)
    return float(mean_val / (std_val + _EPS))


def compute_image_scr(image: np.ndarray) -> float:
    """Estimate a crude signal-to-clutter ratio for a reconstruction."""
    peak = np.max(image)
    clutter = np.mean(np.abs(image - np.mean(image)))
    return float(peak / (clutter + _EPS))


def compute_image_ccr(image: np.ndarray) -> float:
    """Compute a contrast-to-clutter ratio using high-intensity vs background sets."""
    flat = np.asarray(image, dtype=float).ravel()
    if flat.size < 4:
        return 0.0

    signal_threshold = float(np.percentile(flat, 95))
    signal = flat[flat >= signal_threshold]
    clutter = flat[flat < signal_threshold]
    if signal.size == 0 or clutter.size == 0:
        return 0.0

    return float((np.mean(signal) - np.mean(clutter)) / (np.std(clutter) + _EPS))


def _contiguous_width(mask: np.ndarray, center: int) -> int:
    left = center
    right = center
    while left - 1 >= 0 and mask[left - 1]:
        left -= 1
    while right + 1 < mask.shape[0] and mask[right + 1]:
        right += 1
    return int(right - left + 1)


def compute_image_fwhm_pixels(image: np.ndarray) -> float:
    """
    Estimate FWHM in pixels by thresholding at half height along row/column
    crossing the global peak and averaging widths.
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0

    peak_idx = np.unravel_index(int(np.argmax(arr)), arr.shape)
    y_peak, x_peak = int(peak_idx[0]), int(peak_idx[1])
    peak = float(arr[y_peak, x_peak])
    base = float(np.min(arr))
    half_level = base + 0.5 * (peak - base)

    row_mask = arr[y_peak, :] >= half_level
    col_mask = arr[:, x_peak] >= half_level
    if not row_mask[x_peak] or not col_mask[y_peak]:
        return 0.0

    width_x = _contiguous_width(row_mask, x_peak)
    width_y = _contiguous_width(col_mask, y_peak)
    return float(0.5 * (width_x + width_y))


def compute_image_contrast(image: np.ndarray) -> float:
    """Compute a contrast metric for the reconstructed image."""
    max_val = np.max(image)
    min_val = np.min(image)
    return float((max_val - min_val) / (max_val + min_val + _EPS))


def compute_quality_metrics(image: np.ndarray) -> dict[str, float]:
    """Return all image quality metrics for a reconstructed image."""
    return {
        "snr": compute_image_snr(image),
        "scr": compute_image_scr(image),
        "ccr": compute_image_ccr(image),
        "contrast": compute_image_contrast(image),
        "fwhm_px": compute_image_fwhm_pixels(image),
        "peak": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
    }

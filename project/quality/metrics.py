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
        "contrast": compute_image_contrast(image),
        "peak": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
    }

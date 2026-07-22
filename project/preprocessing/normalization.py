"""
preprocessing/normalization.py

Purpose:
    Normalize calibrated S-parameter data onto a consistent numerical
    scale so that downstream image reconstruction algorithms (and any
    machine-learning-based detection stage built later) are not biased by
    differing measurement amplitudes across datasets/antennas.

Input:
    Complex S-parameters, shape (n_freq, ...).

Output:
    Normalized complex S-parameters, same shape as the input.

Description:
    Normalization is applied to the magnitude of each trace while
    preserving phase, since phase carries the propagation-delay
    information required for later time-domain conversion and image
    reconstruction.
    Three strategies are provided:
      - Min-max normalization: magnitude rescaled to [0, 1] per trace.
      - Z-score normalization: magnitude rescaled to zero mean, unit
        variance per trace.
      - Max-magnitude normalization: each trace divided by its own peak
        magnitude (classic radar/UWB convention, preserves relative shape
        across frequency).
"""

from __future__ import annotations

import numpy as np

from utils.exceptions import InvalidSParameterError
from utils.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-12


def _magnitude_phase(s_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.abs(s_params), np.angle(s_params)


def _recombine(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return magnitude * np.exp(1j * phase)


def minmax_normalization(s_params: np.ndarray) -> np.ndarray:
    """
    Purpose:
        Rescale each trace's magnitude to the [0, 1] range while
        preserving phase.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
    Output:
        np.ndarray: normalized S-parameters, same shape.
    """
    if s_params.size == 0:
        raise InvalidSParameterError("Cannot normalize an empty S-parameter array.")

    magnitude, phase = _magnitude_phase(s_params)
    min_val = np.min(magnitude, axis=0, keepdims=True)
    max_val = np.max(magnitude, axis=0, keepdims=True)
    span = np.where((max_val - min_val) < _EPS, 1.0, max_val - min_val)

    logger.info("Applying min-max normalization to S-parameter magnitude")
    normalized_mag = (magnitude - min_val) / span
    return _recombine(normalized_mag, phase)


def zscore_normalization(s_params: np.ndarray) -> np.ndarray:
    """
    Purpose:
        Rescale each trace's magnitude to zero mean and unit standard
        deviation while preserving phase.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
    Output:
        np.ndarray: normalized S-parameters, same shape.
    """
    if s_params.size == 0:
        raise InvalidSParameterError("Cannot normalize an empty S-parameter array.")

    magnitude, phase = _magnitude_phase(s_params)
    mean = np.mean(magnitude, axis=0, keepdims=True)
    std = np.std(magnitude, axis=0, keepdims=True)
    std_safe = np.where(std < _EPS, 1.0, std)

    logger.info("Applying z-score normalization to S-parameter magnitude")
    normalized_mag = (magnitude - mean) / std_safe
    # Shift back to non-negative magnitude domain for physical plausibility
    normalized_mag = normalized_mag - np.min(normalized_mag, axis=0, keepdims=True)
    return _recombine(normalized_mag, phase)


def max_magnitude_normalization(s_params: np.ndarray) -> np.ndarray:
    """
    Purpose:
        Divide each trace by its own peak magnitude (classic UWB/radar
        normalization convention), preserving relative frequency-domain
        shape and phase.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
    Output:
        np.ndarray: normalized S-parameters, same shape.
    """
    if s_params.size == 0:
        raise InvalidSParameterError("Cannot normalize an empty S-parameter array.")

    magnitude = np.abs(s_params)
    peak = np.max(magnitude, axis=0, keepdims=True)
    peak_safe = np.where(peak < _EPS, 1.0, peak)

    logger.info("Applying max-magnitude normalization")
    return s_params / peak_safe


def apply_normalization(s_params: np.ndarray, method: str = "max") -> np.ndarray:
    """
    Purpose:
        Dispatch to the requested normalization strategy for the
        preprocessing pipeline.
    Input:
        s_params (np.ndarray): complex S-parameters.
        method (str): 'minmax', 'zscore', 'max', or 'none'.
    Output:
        np.ndarray: normalized S-parameters.
    """
    method = method.lower()
    if method in ("none", "off", "disabled"):
        return s_params.copy()
    if method in ("minmax", "min-max", "min_max"):
        return minmax_normalization(s_params)
    if method in ("zscore", "z-score", "z_score"):
        return zscore_normalization(s_params)
    if method in ("max", "max_magnitude", "max-magnitude"):
        return max_magnitude_normalization(s_params)
    raise ValueError(f"Unknown normalization method: '{method}'")

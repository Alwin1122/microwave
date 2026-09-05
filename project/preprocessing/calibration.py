"""
preprocessing/calibration.py

Purpose:
    Calibrate raw S-parameter measurements to remove systematic errors
    introduced by the measurement setup (cable loss, connector mismatch,
    antenna coupling offsets), preparing data for reliable image
    reconstruction.

Input:
    Raw complex S-parameters and, optionally, a reference/calibration
    measurement (e.g. an empty-chamber or open/short/load measurement).

Output:
    Calibrated complex S-parameters of the same shape as the input.

Description:
    Two calibration strategies are provided:
      - Reference-based calibration: divide the measured response by a
        known reference/calibration sweep (classic vector-network-analyzer
        style normalization: S_cal = S_meas / S_ref).
      - Self-calibration (when no reference is available): estimate and
        remove a systematic per-frequency offset using the mean response
        across all traces/ports, which suppresses common-mode instrumen-
        tation artifacts shared by every channel.
"""

from __future__ import annotations

import numpy as np

from utils.exceptions import InvalidSParameterError
from utils.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-12


def reference_calibration(s_params: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Purpose:
        Normalize measured S-parameters against a known reference sweep
        (e.g. a calibration/empty-chamber measurement) taken on the same
        frequency grid.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
        reference (np.ndarray): complex reference sweep, broadcastable to
            s_params' shape (typically shape (n_freq,) or (n_freq, ...)
            matching s_params).
    Output:
        np.ndarray: calibrated S-parameters, same shape as s_params.
    """
    if reference.shape[0] != s_params.shape[0]:
        raise InvalidSParameterError(
            "Reference calibration sweep length does not match the number "
            "of frequency points in the dataset."
        )

    ref = reference
    if ref.ndim == 1:
        # broadcast reference across all trailing (port/trace) dimensions
        ref = ref.reshape((ref.shape[0],) + (1,) * (s_params.ndim - 1))

    logger.info("Applying reference-based calibration (S_meas / S_ref)")
    denom = np.where(np.abs(ref) < _EPS, _EPS, ref)
    return s_params / denom


def self_calibration(s_params: np.ndarray) -> np.ndarray:
    """
    Purpose:
        Remove common-mode systematic offsets shared across every trace
        when no separate reference/calibration measurement is available,
        by subtracting the per-frequency mean across traces and restoring
        the original signal's overall scale.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
    Output:
        np.ndarray: calibrated S-parameters, same shape as s_params.
    """
    if s_params.ndim < 2 or s_params.shape[1] < 2:
        logger.warning(
            "Self-calibration needs multiple traces/ports; skipping (single-trace data)."
        )
        return s_params.copy()

    axes_to_average = tuple(range(1, s_params.ndim))
    common_mode = np.mean(s_params, axis=axes_to_average, keepdims=True)

    logger.info("Applying self-calibration (common-mode offset removal)")
    calibrated = s_params - common_mode
    # Restore overall signal energy so calibration does not zero out data
    original_energy = np.mean(np.abs(s_params) ** 2)
    new_energy = np.mean(np.abs(calibrated) ** 2)
    if new_energy > _EPS:
        scale = np.sqrt(original_energy / new_energy)
        calibrated = calibrated * scale
    return calibrated


def apply_calibration(
    s_params: np.ndarray, method: str = "self", reference: np.ndarray | None = None
) -> np.ndarray:
    """
    Purpose:
        Dispatch to the requested calibration strategy for the
        preprocessing pipeline.
    Input:
        s_params (np.ndarray): complex S-parameters.
        method (str): 'reference', 'self', or 'none'.
        reference (np.ndarray | None): required when method == 'reference'.
    Output:
        np.ndarray: calibrated S-parameters.
    """
    method = method.lower()
    if method in ("none", "off", "disabled"):
        return s_params.copy()
    if method == "reference":
        if reference is None:
            raise ValueError("Reference calibration requires a `reference` array.")
        return reference_calibration(s_params, reference)
    if method == "self":
        return self_calibration(s_params)
    raise ValueError(f"Unknown calibration method: '{method}'")

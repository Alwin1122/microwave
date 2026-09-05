"""
preprocessing/artifact_suppression.py

Purpose:
    Remove strong, unwanted reflections (skin/antenna coupling artifacts,
    static clutter) from calibrated, normalized S-parameter data so that
    the much weaker signals scattered by internal targets (e.g. a tumor
    response in microwave breast imaging) become visible for later image
    reconstruction.

Input:
    Complex, calibrated & normalized S-parameters, typically shaped
    (n_freq, n_traces) where each trace corresponds to one antenna
    position/rotation, or (n_freq, n_ports, n_ports).

Output:
    Artifact-suppressed complex S-parameters, same shape as the input.

Description:
    Implements three techniques and a "hybrid" combination of them, in
    line with the confocal microwave imaging literature on skin-artifact
    removal for antenna-array measurement systems:

      1. Background subtraction: subtract a static background/reference
         sweep (e.g. an empty-scenario measurement) when available.

      2. Rotation / average-trace subtraction: when no separate background
         measurement exists, the mean response across all
         antenna/rotation traces is treated as the common "skin" response
         and subtracted from every trace (classic technique for circular
         antenna-array UWB systems).

      3. SVD-based (Singular Value Decomposition) adaptive clutter
         removal: the dominant singular components of the trace matrix
         (which correspond to the strong, spatially-correlated skin
         reflection) are removed, leaving the weaker, less-correlated
         target scattering.

      4. Hybrid artifact suppression: sequentially applies rotation
         subtraction followed by SVD-based clutter removal. Combining a
         simple subtraction stage with an adaptive/statistical stage
         suppresses both the strong common-mode artifact and residual
         correlated clutter that a single technique leaves behind, which
         is the "hybrid" approach referenced in the project brief.
"""

from __future__ import annotations

import numpy as np

from utils.exceptions import InvalidSParameterError
from utils.logger import get_logger

logger = get_logger(__name__)


def _flatten_traces(s_params: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Flatten any (n_freq, ...) array into (n_freq, n_traces) for
    matrix-based operations, remembering the original shape to restore it."""
    original_shape = s_params.shape
    flat = s_params.reshape(original_shape[0], -1)
    return flat, original_shape


def background_subtraction(
    s_params: np.ndarray, background: np.ndarray | None = None
) -> np.ndarray:
    """
    Purpose:
        Subtract a static background/reference response from the
        measured data. If no explicit background is supplied, the mean
        across all traces is used as an estimate of the common background
        (rotation subtraction).
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
        background (np.ndarray | None): optional explicit background
            sweep, broadcastable to s_params' shape.
    Output:
        np.ndarray: background-subtracted S-parameters, same shape.
    """
    if s_params.size == 0:
        raise InvalidSParameterError("Cannot process an empty S-parameter array.")

    if background is not None:
        if background.shape[0] != s_params.shape[0]:
            raise InvalidSParameterError(
                "Background reference length does not match number of frequency points."
            )
        bg = background
        if bg.ndim == 1:
            bg = bg.reshape((bg.shape[0],) + (1,) * (s_params.ndim - 1))
        logger.info("Applying background subtraction using explicit reference sweep")
        return s_params - bg

    flat, shape = _flatten_traces(s_params)
    if flat.shape[1] < 2:
        logger.warning(
            "Only one trace available; cannot estimate background via averaging. "
            "Returning data unchanged."
        )
        return s_params.copy()

    logger.info(
        "Applying rotation/average-trace subtraction "
        f"(estimating background from {flat.shape[1]} traces)"
    )
    mean_trace = np.mean(flat, axis=1, keepdims=True)
    result = flat - mean_trace
    return result.reshape(shape)


def svd_clutter_removal(
    s_params: np.ndarray, n_components_removed: int = 1
) -> np.ndarray:
    """
    Purpose:
        Remove the dominant singular components of the trace matrix,
        which typically correspond to the strong, spatially-correlated
        skin/coupling artifact, using Singular Value Decomposition.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
        n_components_removed (int): number of leading singular components
            to remove (default 1, i.e. remove only the strongest
            component).
    Output:
        np.ndarray: clutter-suppressed S-parameters, same shape.
    """
    if s_params.size == 0:
        raise InvalidSParameterError("Cannot process an empty S-parameter array.")

    flat, shape = _flatten_traces(s_params)
    n_freq, n_traces = flat.shape

    if n_traces < 2:
        logger.warning(
            "SVD clutter removal needs multiple traces; skipping (single-trace data)."
        )
        return s_params.copy()

    k = min(n_components_removed, min(n_freq, n_traces) - 1)
    k = max(k, 0)
    if k == 0:
        return s_params.copy()

    logger.info(f"Applying SVD-based clutter removal (removing top {k} component(s))")
    U, S, Vh = np.linalg.svd(flat, full_matrices=False)
    S_reduced = S.copy()
    S_reduced[:k] = 0.0
    cleaned = (U * S_reduced) @ Vh
    return cleaned.reshape(shape)


def hybrid_artifact_suppression(
    s_params: np.ndarray,
    background: np.ndarray | None = None,
    n_components_removed: int = 1,
) -> np.ndarray:
    """
    Purpose:
        Apply the hybrid artifact-suppression strategy: rotation/
        background subtraction followed by SVD-based adaptive clutter
        removal, combining a deterministic subtraction stage with a
        statistical/adaptive stage for stronger artifact rejection than
        either technique alone.
    Input:
        s_params (np.ndarray): complex, calibrated, normalized
            S-parameters, shape (n_freq, ...).
        background (np.ndarray | None): optional explicit background
            sweep passed through to background_subtraction().
        n_components_removed (int): number of leading SVD components to
            remove in the second stage.
    Output:
        np.ndarray: artifact-suppressed S-parameters, same shape as input.
    """
    logger.info("Running hybrid artifact suppression (subtraction + SVD)")
    stage1 = background_subtraction(s_params, background=background)
    stage2 = svd_clutter_removal(stage1, n_components_removed=n_components_removed)
    return stage2


def apply_artifact_suppression(
    s_params: np.ndarray,
    method: str = "hybrid",
    background: np.ndarray | None = None,
    n_components_removed: int = 1,
) -> np.ndarray:
    """
    Purpose:
        Dispatch to the requested artifact-suppression strategy for the
        preprocessing pipeline.
    Input:
        s_params (np.ndarray): complex S-parameters.
        method (str): 'background', 'svd', 'hybrid', or 'none'.
        background (np.ndarray | None): optional explicit background sweep.
        n_components_removed (int): SVD components to remove (svd/hybrid).
    Output:
        np.ndarray: artifact-suppressed S-parameters.
    """
    method = method.lower()
    if method in ("none", "off", "disabled"):
        return s_params.copy()
    if method == "background":
        return background_subtraction(s_params, background=background)
    if method == "svd":
        return svd_clutter_removal(s_params, n_components_removed=n_components_removed)
    if method == "hybrid":
        return hybrid_artifact_suppression(
            s_params, background=background, n_components_removed=n_components_removed
        )
    raise ValueError(f"Unknown artifact suppression method: '{method}'")

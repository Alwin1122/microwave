"""Delay-Multiply-and-Sum (DMAS) beamforming reconstruction."""

from __future__ import annotations

import numpy as np

from reconstruction.das import (
    coherence_factor,
    gather_delayed_samples,
)

# Textbook DMAS uses signed square-root (pair_exponent=0.5) and no extra
# coherence weight. UM-BMID scans benefit from a slightly higher pairing
# exponent (stronger compact-scatterer emphasis) plus a mild CF weight.
DEFAULT_DMAS_PAIR_EXPONENT = 0.55
DEFAULT_DMAS_COHERENCE_GAMMA = 0.6
DEFAULT_DMAS_EPS = 1e-9


def _signed_power(values: np.ndarray, exponent: float, eps: float) -> np.ndarray:
    mag = np.abs(values) + float(eps)
    return np.real(np.sign(values) * np.power(mag, float(exponent)))


def dmas_reconstruct(
    time_signals: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    wave_speed: float = 3e8,
    pair_exponent: float = DEFAULT_DMAS_PAIR_EXPONENT,
    coherence_gamma: float = DEFAULT_DMAS_COHERENCE_GAMMA,
    eps: float = DEFAULT_DMAS_EPS,
) -> np.ndarray:
    """Reconstruct an image using Delay-Multiply-and-Sum beamforming.

    Defaults are BMID-tuned (pair_exponent=0.55, coherence_gamma=0.6).
    Set pair_exponent=0.5 and coherence_gamma=0 for the textbook form.
    """
    if time_signals.ndim != 2:
        raise ValueError("time_signals must be 2D (time, traces).")

    n_time, n_traces = time_signals.shape
    if antenna_positions.shape != (n_traces, 2):
        raise ValueError("antenna_positions must have shape (n_traces, 2).")

    points = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)
    delayed = gather_delayed_samples(
        time_signals, frequencies, antenna_positions, points, wave_speed=wave_speed
    )
    image = np.zeros(points.shape[0], dtype=float)
    for i in range(n_traces):
        sample_i = delayed[i]
        for j in range(i + 1, n_traces):
            prod = sample_i * np.conj(delayed[j])
            image += _signed_power(prod, pair_exponent, eps)

    gamma = float(coherence_gamma)
    if gamma > 0.0:
        image = image * np.power(coherence_factor(delayed), gamma)
    return image.reshape((len(grid_y), len(grid_x)))

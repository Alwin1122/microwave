"""DMAS-D4 beamforming reconstruction."""

from __future__ import annotations

import numpy as np

from reconstruction.dmas import (
    DEFAULT_DMAS_COHERENCE_GAMMA,
    DEFAULT_DMAS_EPS,
    DEFAULT_DMAS_PAIR_EXPONENT,
    dmas_reconstruct,
)

# Textbook D4 uses exponent 1/4. On the UM-BMID cohort sweep
# (results/beamformer_constant_sweep.pdf) exponents from 0.10 to 0.80
# landed within 0.01 of each other. 0.25 is that textbook value and
# matches the peak. 0.00 scatters the spot. 0.85 and above raise
# healthy false positives. The earlier 0.55 default was not better.
DEFAULT_DMAS_D4_EXPONENT = 0.25


def dmas_d4_reconstruct(
    time_signals: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    wave_speed: float = 3e8,
    pair_exponent: float = DEFAULT_DMAS_PAIR_EXPONENT,
    coherence_gamma: float = DEFAULT_DMAS_COHERENCE_GAMMA,
    eps: float = DEFAULT_DMAS_EPS,
    d4_exponent: float = DEFAULT_DMAS_D4_EXPONENT,
) -> np.ndarray:
    """Reconstruct using DMAS followed by a contrast-preserving D4 map.

    ``d4_exponent=0.25`` is both the textbook fourth root and the cohort
    average. See results/beamformer_constant_sweep.pdf.
    """
    intermediate = dmas_reconstruct(
        time_signals,
        frequencies,
        antenna_positions,
        grid_x,
        grid_y,
        wave_speed=wave_speed,
        pair_exponent=pair_exponent,
        coherence_gamma=coherence_gamma,
        eps=eps,
    )
    mag = np.abs(intermediate)
    return np.sign(intermediate) * np.power(mag, float(d4_exponent))

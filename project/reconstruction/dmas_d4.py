"""DMAS-D4 beamforming reconstruction."""

from __future__ import annotations

import numpy as np

from reconstruction.dmas import (
    DEFAULT_DMAS_COHERENCE_GAMMA,
    DEFAULT_DMAS_EPS,
    DEFAULT_DMAS_PAIR_EXPONENT,
    dmas_reconstruct,
)

# Textbook-style D4 mapping uses exponent 1/4, which over-compresses
# UM-BMID adipose-referenced images (almost all pixels look equally bright).
# 0.55 keeps nonlinear emphasis while preserving usable contrast for ROI.
DEFAULT_DMAS_D4_EXPONENT = 0.55


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

    ``d4_exponent=0.25`` is the original compressive mapping. The project
    default (0.55) is tuned so compact tumors stay visible against clutter.
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

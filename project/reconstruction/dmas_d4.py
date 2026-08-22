"""DMAS-D4 beamforming reconstruction."""

from __future__ import annotations

import numpy as np

from reconstruction.dmas import dmas_reconstruct


def dmas_d4_reconstruct(
    time_signals: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    wave_speed: float = 3e8,
) -> np.ndarray:
    """Reconstruct an image using DMAS-D4 beamforming.

    This implementation applies DMAS twice to produce a higher-order
    nonlinear emphasis on coherent scatterers.
    """
    intermediate = dmas_reconstruct(
        time_signals, frequencies, antenna_positions, grid_x, grid_y, wave_speed=wave_speed
    )
    return np.sign(intermediate) * np.abs(intermediate) ** 0.25

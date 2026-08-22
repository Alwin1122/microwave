"""Delay-Multiply-and-Sum (DMAS) beamforming reconstruction."""

from __future__ import annotations

import numpy as np

from reconstruction.das import _compute_time_axis, _compute_travel_times


def dmas_reconstruct(
    time_signals: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    wave_speed: float = 3e8,
) -> np.ndarray:
    """Reconstruct an image using Delay-Multiply-and-Sum beamforming."""
    if time_signals.ndim != 2:
        raise ValueError("time_signals must be 2D (time, traces).")

    n_time, n_traces = time_signals.shape
    if antenna_positions.shape != (n_traces, 2):
        raise ValueError("antenna_positions must have shape (n_traces, 2).")

    points = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)
    travel_times = _compute_travel_times(antenna_positions, points, wave_speed=wave_speed)
    time_axis = _compute_time_axis(frequencies, n_time)

    image = np.zeros(points.shape[0], dtype=float)
    for i in range(n_traces):
        signal_i = time_signals[:, i]
        idx_i = np.round(travel_times[i] / (time_axis[1] - time_axis[0])).astype(int)
        valid_i = (idx_i >= 0) & (idx_i < n_time)
        for j in range(i + 1, n_traces):
            signal_j = time_signals[:, j]
            idx_j = np.round(travel_times[j] / (time_axis[1] - time_axis[0])).astype(int)
            valid_j = (idx_j >= 0) & (idx_j < n_time)
            valid = valid_i & valid_j
            prod = signal_i[idx_i[valid]] * np.conj(signal_j[idx_j[valid]])
            image[valid] += np.real(np.sign(prod) * np.sqrt(np.abs(prod) + 1e-12))

    image = image.reshape((len(grid_y), len(grid_x)))
    return image

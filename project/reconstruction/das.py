"""Delay-and-Sum (DAS) beamforming reconstruction."""

from __future__ import annotations

import numpy as np

from reconstruction.ifft import frequency_to_time

_C = 3e8  # speed of light in m/s


def _compute_time_axis(frequencies: np.ndarray, n_samples: int) -> np.ndarray:
    df = frequencies[1] - frequencies[0]
    period = 1.0 / (df * n_samples)
    return np.arange(n_samples) * period


def _compute_travel_times(antenna_positions: np.ndarray, points: np.ndarray, wave_speed: float = _C) -> np.ndarray:
    distances = np.linalg.norm(antenna_positions[:, None, :] - points[None, :, :], axis=-1)
    return 2.0 * distances / wave_speed


def das_reconstruct(
    time_signals: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    wave_speed: float = _C,
) -> np.ndarray:
    """Reconstruct an image using Delay-and-Sum beamforming.

    Args:
        time_signals: Time-domain signals, shape (n_time, n_traces).
        frequencies: Frequency axis used to compute time resolution.
        antenna_positions: Positions of each antenna, shape (n_traces, 2).
        grid_x: 1D x-axis locations.
        grid_y: 1D y-axis locations.

    Returns:
        2D reconstructed image of shape (len(grid_y), len(grid_x)).
    """
    if time_signals.ndim != 2:
        raise ValueError("time_signals must be 2D (time, traces).")

    n_time, n_traces = time_signals.shape
    if antenna_positions.shape != (n_traces, 2):
        raise ValueError("antenna_positions must have shape (n_traces, 2).")

    points = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)
    travel_times = _compute_travel_times(antenna_positions, points, wave_speed=wave_speed)
    time_axis = _compute_time_axis(frequencies, n_time)

    image = np.zeros(points.shape[0], dtype=float)
    for ant_idx in range(n_traces):
        signal = time_signals[:, ant_idx]
        sample_indices = np.round(travel_times[ant_idx] / (time_axis[1] - time_axis[0])).astype(int)
        valid = (sample_indices >= 0) & (sample_indices < n_time)
        image[valid] += np.abs(signal[sample_indices[valid]])

    image = image.reshape((len(grid_y), len(grid_x)))
    return image


def das_from_frequency(
    s_parameters: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    zero_padding: int = 0,
    wave_speed: float = _C,
) -> np.ndarray:
    time_signals = frequency_to_time(s_parameters, frequencies, zero_padding=zero_padding)
    return das_reconstruct(time_signals, frequencies, antenna_positions, grid_x, grid_y, wave_speed=wave_speed)

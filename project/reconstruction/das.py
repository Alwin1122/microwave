"""Delay-and-Sum (DAS) beamforming reconstruction."""

from __future__ import annotations

import numpy as np

from reconstruction.ifft import frequency_to_time

_C = 3e8  # speed of light in m/s
_EPS = 1e-12

# Textbook DAS uses coherence_gamma=0 (plain delay-and-sum of envelopes).
# UM-BMID adipose-referenced scans have strong ring clutter, so a mild
# coherence-factor weight (0.75) suppresses incoherent background without
# erasing compact tumor foci.
DEFAULT_DAS_COHERENCE_GAMMA = 0.75


def _compute_time_axis(frequencies: np.ndarray, n_samples: int) -> np.ndarray:
    df = frequencies[1] - frequencies[0]
    period = 1.0 / (df * n_samples)
    return np.arange(n_samples) * period


def _compute_travel_times(antenna_positions: np.ndarray, points: np.ndarray, wave_speed: float = _C) -> np.ndarray:
    distances = np.linalg.norm(antenna_positions[:, None, :] - points[None, :, :], axis=-1)
    return 2.0 * distances / wave_speed


def gather_delayed_samples(
    time_signals: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    points: np.ndarray,
    wave_speed: float = _C,
) -> np.ndarray:
    """Sample each antenna trace at the two-way delay of every image point.

    Returns a complex array of shape (n_traces, n_points).
    """
    n_time, n_traces = time_signals.shape
    n_points = points.shape[0]
    travel_times = _compute_travel_times(antenna_positions, points, wave_speed=wave_speed)
    time_axis = _compute_time_axis(frequencies, n_time)
    dt = float(time_axis[1] - time_axis[0]) if n_time > 1 else 1.0
    delayed = np.zeros((n_traces, n_points), dtype=np.complex128)
    for ant_idx in range(n_traces):
        idx = np.round(travel_times[ant_idx] / dt).astype(int)
        valid = (idx >= 0) & (idx < n_time)
        delayed[ant_idx, valid] = time_signals[idx[valid], ant_idx]
    return delayed


def coherence_factor(delayed_samples: np.ndarray) -> np.ndarray:
    """Hollmann-style coherence factor in [0, 1] for each image point."""
    n_traces = max(delayed_samples.shape[0], 1)
    coherent = np.abs(np.sum(delayed_samples, axis=0)) ** 2
    incoherent = n_traces * np.sum(np.abs(delayed_samples) ** 2, axis=0)
    return coherent / (incoherent + _EPS)


def das_reconstruct(
    time_signals: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    wave_speed: float = _C,
    coherence_gamma: float = DEFAULT_DAS_COHERENCE_GAMMA,
) -> np.ndarray:
    """Reconstruct an image using Delay-and-Sum beamforming.

    ``coherence_gamma=0`` is textbook DAS. The project default (0.75) applies
    a coherence-factor weight that is better matched to UM-BMID clutter.
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
    image = np.sum(np.abs(delayed), axis=0)
    gamma = float(coherence_gamma)
    if gamma > 0.0:
        image = image * np.power(coherence_factor(delayed), gamma)
    return image.reshape((len(grid_y), len(grid_x)))


def das_from_frequency(
    s_parameters: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    zero_padding: int = 0,
    wave_speed: float = _C,
    coherence_gamma: float = DEFAULT_DAS_COHERENCE_GAMMA,
) -> np.ndarray:
    time_signals = frequency_to_time(s_parameters, frequencies, zero_padding=zero_padding)
    return das_reconstruct(
        time_signals,
        frequencies,
        antenna_positions,
        grid_x,
        grid_y,
        wave_speed=wave_speed,
        coherence_gamma=coherence_gamma,
    )

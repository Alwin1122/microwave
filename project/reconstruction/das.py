"""Delay-and-Sum (DAS) beamforming reconstruction."""

from __future__ import annotations

import numpy as np

from reconstruction.ifft import frequency_to_time

_C = 3e8  # speed of light in m/s
_EPS = 1e-12

# Textbook DAS uses coherence_gamma=0 (plain delay-and-sum of envelopes).
# On a 12-scan UM-BMID cohort (see results/beamformer_constant_sweep.pdf)
# the average-best gamma is 0.55. gamma=0 leaves the antenna ring in charge.
# 0.75 is within 0.01 of the peak. 0.80 and above starts false positives
# on healthy scans.
DEFAULT_DAS_COHERENCE_GAMMA = 0.55


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

    ``coherence_gamma=0`` is textbook DAS. The project default (0.55) is the
    cohort average from results/beamformer_constant_sweep.pdf.
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


def das_coherent_from_frequency(
    s_parameters: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    wave_speed: float = _C,
    pad_factor: int = 4,
) -> np.ndarray:
    """Phase-coherent monostatic DAS: |sum over antennas of the echo at each pixel's delay|.

    The IFFT of the band-limited sweep is a baseband signal, so each sample is
    re-modulated by exp(j*2*pi*f0*tau) before the coherent sum. ``pad_factor``
    refines the delay grid (dt = 1 / (pad_factor * N * df)).
    """
    s = np.asarray(s_parameters, dtype=complex)
    freqs = np.asarray(frequencies, dtype=float)
    n_freq, n_traces = s.shape
    if antenna_positions.shape != (n_traces, 2):
        raise ValueError("antenna_positions must have shape (n_traces, 2).")
    n_time = int(n_freq * max(1, pad_factor))
    df = float(freqs[1] - freqs[0])
    dt = 1.0 / (n_time * df)
    baseband = np.fft.ifft(s, n=n_time, axis=0)

    points = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)
    tau = _compute_travel_times(antenna_positions, points, wave_speed=wave_speed)
    idx = np.rint(tau / dt).astype(int)
    valid = (idx >= 0) & (idx < n_time)
    samples = baseband[np.clip(idx, 0, n_time - 1), np.arange(n_traces)[:, None]]
    samples = np.where(valid, samples * np.exp(2j * np.pi * freqs[0] * tau), 0.0)
    image = np.abs(samples.sum(axis=0))
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

"""Frequency-to-time conversion utilities."""

from __future__ import annotations

import numpy as np


def frequency_to_time(
    s_parameters: np.ndarray,
    frequencies: np.ndarray,
    axis: int = 0,
    zero_padding: int = 0,
) -> np.ndarray:
    """Convert frequency-domain S-parameters to time-domain signals.

    The IFFT is applied along the selected frequency axis. Optional zero
    padding improves time resolution and avoids aliasing.

    Args:
        s_parameters: Complex frequency-domain S-parameters.
        frequencies: 1D frequency axis in Hz.
        axis: Axis corresponding to frequency samples.
        zero_padding: Number of zero-frequency samples to pad at the end.

    Returns:
        Complex time-domain signals with the same shape except for the
        frequency axis length, which may be extended by zero_padding.
    """
    if s_parameters.ndim == 0:
        raise ValueError("S-parameter array must be at least 1D.")

    if frequencies.ndim != 1:
        raise ValueError("Frequencies must be a 1D array.")

    n_freq = frequencies.shape[0]
    if s_parameters.shape[axis] != n_freq:
        raise ValueError(
            "Frequency axis length of s_parameters does not match frequencies."
        )

    if n_freq < 2:
        raise ValueError("At least two frequency samples are required for IFFT.")

    freq_spacing = np.diff(frequencies)
    if not np.allclose(freq_spacing, freq_spacing[0], rtol=1e-3, atol=0.0):
        raise ValueError("Frequencies must be uniformly spaced for simple IFFT.")

    n_out = n_freq + zero_padding
    time_domain = np.fft.ifft(s_parameters, n=n_out, axis=axis)
    return time_domain

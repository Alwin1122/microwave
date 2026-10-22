"""Breast-window sinogram used by the published UM-BMID convolutional network.

The hand-built features collapse each time sample across antennas. This keeps
the time-by-antenna picture those features throw away: log magnitude inside
the 0.71–5.71 ns breast window, once for the empty-chamber-calibrated scan
and once after the rotational mean is removed. Antennas are resampled onto a
fixed circle so the three generations share one input size.
"""

from __future__ import annotations

import numpy as np

from classification.features import BREAST_T_END_S, BREAST_T_START_S, breast_time_mask, calibrate_empty
from reconstruction.bmid_geometry import remove_rotational_mean
from reconstruction.ifft import frequency_to_time

N_ANTENNAS = 32
_LOG_FLOOR = 1e-12


def resample_circle(image: np.ndarray, n_out: int = N_ANTENNAS) -> np.ndarray:
    """Linearly resample the antenna axis onto ``n_out`` angles, wrapping around."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"expected a time-by-antenna image, got shape {image.shape}")
    n_in = image.shape[1]
    if n_in < 2:
        raise ValueError("need at least two antennas to resample the circle")
    if n_out == n_in:
        return np.array(image, dtype=float, copy=True)
    dst = np.linspace(0.0, n_in, n_out, endpoint=False)
    extended = np.concatenate([image, image[:, :1]], axis=1)
    ext_x = np.arange(n_in + 1, dtype=float)
    out = np.empty((image.shape[0], n_out), dtype=float)
    for row in range(image.shape[0]):
        out[row] = np.interp(dst, ext_x, extended[row])
    return out


def _log_magnitude(scan: np.ndarray, frequencies: np.ndarray, mask: np.ndarray) -> np.ndarray:
    timed = np.abs(frequency_to_time(scan, frequencies))
    return np.log10(timed[mask] + _LOG_FLOOR)


def sinogram_pair(
    raw: np.ndarray,
    reference: np.ndarray,
    frequencies: np.ndarray,
    n_out: int = N_ANTENNAS,
) -> np.ndarray:
    """Two channels, shape (2, n_time, n_out): calibrated, then rotational residual.

    Each channel is then standardised on its own scan, so a louder sitting
    does not become the label.
    """
    cal = calibrate_empty(raw, reference)
    mask = breast_time_mask(frequencies)
    if int(mask.sum()) < 8:
        raise ValueError(f"breast window has only {int(mask.sum())} samples")
    channels = np.stack(
        [
            resample_circle(_log_magnitude(cal, frequencies, mask), n_out),
            resample_circle(_log_magnitude(remove_rotational_mean(cal), frequencies, mask), n_out),
        ],
        axis=0,
    )
    return _per_scan_standardize(channels).astype(np.float32)


def _per_scan_standardize(channels: np.ndarray) -> np.ndarray:
    out = np.empty_like(channels, dtype=float)
    for index in range(channels.shape[0]):
        values = channels[index]
        scale = float(values.std())
        # A vanished residual (a shell that is the same from every antenna) must
        # stay zero. Dividing by a tiny std would turn numerical dust into a picture.
        out[index] = 0.0 if scale < 1e-6 else (values - float(values.mean())) / scale
    return out


def breast_window_ns() -> tuple[float, float]:
    return (BREAST_T_START_S * 1e9, BREAST_T_END_S * 1e9)

"""
preprocessing/filtering.py

Purpose:
    Provide noise-filtering operations applied to raw S-parameter data as
    the first stage of the Module 2 preprocessing pipeline.

Input:
    Complex S-parameter arrays of shape (n_freq, ...) as produced by
    Module 1 loaders.

Output:
    Filtered S-parameter arrays of the same shape.

Description:
    Three complementary filters are implemented, all operating along the
    frequency axis (axis 0) independently for every trace/port pair:
      - Moving-average filter: simple low-pass smoothing.
      - Savitzky-Golay filter: polynomial smoothing that preserves peak
        shape better than a moving average, useful for resonance-rich
        microwave responses.
      - Butterworth low-pass filter: frequency-domain style smoothing
        with a sharper roll-off, applied to the (already frequency-domain)
        S-parameter trace as if it were a signal indexed by sample number,
        which is a standard way to denoise measurement sweeps.
    Filtering is applied separately to the real and imaginary parts (or
    magnitude and phase) to preserve the complex signal correctly.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, medfilt, savgol_filter

from utils.exceptions import InvalidSParameterError
from utils.logger import get_logger

logger = get_logger(__name__)


def _apply_along_freq(s_params: np.ndarray, func) -> np.ndarray:
    """Apply a real-valued 1D filter function along axis 0 to real and
    imaginary parts independently, preserving the complex dtype/shape."""
    original_shape = s_params.shape
    flat = s_params.reshape(original_shape[0], -1)

    real_filtered = np.apply_along_axis(func, 0, flat.real)
    imag_filtered = np.apply_along_axis(func, 0, flat.imag)

    result = (real_filtered + 1j * imag_filtered).reshape(original_shape)
    return result


def moving_average_filter(s_params: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Purpose:
        Smooth S-parameter traces with a simple centered moving average,
        reducing high-frequency measurement noise.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
        window_size (int): number of samples averaged (must be >= 1, odd
            values are recommended for a symmetric window).
    Output:
        np.ndarray: filtered S-parameters, same shape as input.
    """
    if s_params.shape[0] == 0:
        raise InvalidSParameterError("Cannot filter an empty S-parameter array.")
    window_size = max(1, min(window_size, s_params.shape[0]))
    if window_size == 1:
        return s_params.copy()

    kernel = np.ones(window_size) / window_size

    def _conv(x: np.ndarray) -> np.ndarray:
        return np.convolve(x, kernel, mode="same")

    logger.info(f"Applying moving-average filter (window={window_size})")
    return _apply_along_freq(s_params, _conv)


def savitzky_golay_filter(
    s_params: np.ndarray, window_length: int = 7, polyorder: int = 3
) -> np.ndarray:
    """
    Purpose:
        Smooth S-parameter traces using a Savitzky-Golay polynomial
        filter, which preserves resonance peak shapes better than a plain
        moving average.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
        window_length (int): odd integer, length of the filter window.
        polyorder (int): order of the polynomial used to fit samples;
            must be less than window_length.
    Output:
        np.ndarray: filtered S-parameters, same shape as input.
    """
    n_freq = s_params.shape[0]
    if n_freq == 0:
        raise InvalidSParameterError("Cannot filter an empty S-parameter array.")

    window_length = min(window_length, n_freq if n_freq % 2 == 1 else n_freq - 1)
    window_length = max(window_length, polyorder + 1 + ((polyorder + 1) % 2 == 0))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, n_freq if n_freq % 2 == 1 else n_freq - 1)

    if window_length <= polyorder or window_length < 3:
        logger.warning(
            "Not enough frequency points for Savitzky-Golay filtering; skipping."
        )
        return s_params.copy()

    def _sg(x: np.ndarray) -> np.ndarray:
        return savgol_filter(x, window_length=window_length, polyorder=polyorder)

    logger.info(
        f"Applying Savitzky-Golay filter (window={window_length}, order={polyorder})"
    )
    return _apply_along_freq(s_params, _sg)


def butterworth_lowpass_filter(
    s_params: np.ndarray, cutoff: float = 0.3, order: int = 4
) -> np.ndarray:
    """
    Purpose:
        Apply a zero-phase Butterworth low-pass filter along the frequency
        axis to remove rapid sample-to-sample noise while preserving the
        overall trend of the response.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
        cutoff (float): normalized cutoff frequency in (0, 1), where 1.0
            is the Nyquist frequency of the sample index axis.
        order (int): filter order (higher = sharper roll-off).
    Output:
        np.ndarray: filtered S-parameters, same shape as input.
    """
    n_freq = s_params.shape[0]
    if n_freq == 0:
        raise InvalidSParameterError("Cannot filter an empty S-parameter array.")
    if n_freq < 3 * order:
        logger.warning(
            "Not enough frequency points for stable Butterworth filtering; skipping."
        )
        return s_params.copy()

    cutoff = float(np.clip(cutoff, 1e-3, 0.999))
    b, a = butter(order, cutoff, btype="low")

    def _filt(x: np.ndarray) -> np.ndarray:
        return filtfilt(b, a, x)

    logger.info(
        f"Applying Butterworth low-pass filter (cutoff={cutoff}, order={order})"
    )
    return _apply_along_freq(s_params, _filt)


def gaussian_smoothing_filter(s_params: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Purpose:
        Apply mild Gaussian smoothing along the frequency axis, often a
        good compromise between denoising and preserving weak response
        variations.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
        sigma (float): standard deviation of the Gaussian kernel in
            samples.
    Output:
        np.ndarray: filtered S-parameters, same shape as input.
    """
    if s_params.shape[0] == 0:
        raise InvalidSParameterError("Cannot filter an empty S-parameter array.")

    sigma = max(float(sigma), 0.0)
    if sigma == 0.0:
        return s_params.copy()

    logger.info(f"Applying Gaussian smoothing filter (sigma={sigma})")
    return _apply_along_freq(
        s_params, lambda x: gaussian_filter1d(x, sigma=sigma, mode="nearest")
    )


def median_smoothing_filter(s_params: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Purpose:
        Apply a median filter along the frequency axis to suppress
        isolated spikes while preserving broader structure.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
        kernel_size (int): odd median-filter length in samples.
    Output:
        np.ndarray: filtered S-parameters, same shape as input.
    """
    if s_params.shape[0] == 0:
        raise InvalidSParameterError("Cannot filter an empty S-parameter array.")

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(
        kernel_size,
        s_params.shape[0] if s_params.shape[0] % 2 == 1 else s_params.shape[0] - 1,
    )
    if kernel_size <= 1:
        return s_params.copy()

    logger.info(f"Applying median smoothing filter (kernel_size={kernel_size})")
    return _apply_along_freq(s_params, lambda x: medfilt(x, kernel_size=kernel_size))


def apply_noise_filter(
    s_params: np.ndarray, method: str = "savgol", **kwargs
) -> np.ndarray:
    """
    Purpose:
        Convenience dispatcher used by the preprocessing pipeline to apply
        the selected noise-filtering method by name.
    Input:
        s_params (np.ndarray): complex S-parameters.
        method (str): one of 'moving_average', 'gaussian', 'median',
            'savgol', 'butterworth', 'none'.
        **kwargs: forwarded to the specific filter function.
    Output:
        np.ndarray: filtered S-parameters.
    """
    method = method.lower()
    if method in ("none", "off", "disabled"):
        return s_params.copy()
    if method in ("moving_average", "moving-average", "ma"):
        return moving_average_filter(s_params, **kwargs)
    if method in ("gaussian", "gaussian_smoothing"):
        return gaussian_smoothing_filter(s_params, **kwargs)
    if method in ("median", "median_filter"):
        return median_smoothing_filter(s_params, **kwargs)
    if method in ("savgol", "savitzky_golay", "savitzky-golay"):
        return savitzky_golay_filter(s_params, **kwargs)
    if method in ("butterworth", "butter", "lowpass"):
        return butterworth_lowpass_filter(s_params, **kwargs)
    raise ValueError(f"Unknown filtering method: '{method}'")

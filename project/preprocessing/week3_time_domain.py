"""
Module 2 Week 3 — Hamming, IFFT, matched reference subtract, clutter, global normalize.

Implements steps 10–15 of Signal Preprocessing_21082026.pdf for complex S21
on a uniform frequency grid (single- or multi-channel).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reconstruction.ifft import frequency_to_time


@dataclass
class Week3TimeDomainResult:
    frequencies_hz: np.ndarray
    delta_f_hz: float
    time_s: np.ndarray
    delta_t_s: float
    bandwidth_hz: float
    period_s: float
    s21_filtered: np.ndarray
    s21_hamming: np.ndarray
    s21_reference_hamming: np.ndarray | None
    s21_reference_subtracted_freq: np.ndarray | None
    time_filtered: np.ndarray
    time_hamming: np.ndarray
    time_reference_hamming: np.ndarray | None
    time_reference_subtracted: np.ndarray | None
    time_clutter_removed: np.ndarray | None
    time_global_normalized: np.ndarray | None
    global_scale_alpha: float | None
    matched_windowing_performed: bool
    time_reference_subtraction_performed: bool
    clutter_removal_performed: bool
    notes: list[str] = field(default_factory=list)

    def to_display_dict(self) -> dict:
        return {
            "Week3 Δf (Hz)": f"{self.delta_f_hz:.6g}",
            "Week3 Δt (s)": f"{self.delta_t_s:.6g}",
            "Week3 Bandwidth (Hz)": f"{self.bandwidth_hz:.6g}",
            "Week3 Time Samples": str(self.time_s.shape[0]),
            "Matched Hamming Window": "Yes"
            if self.matched_windowing_performed
            else "No",
            "Time-Domain Ref Subtract": "Yes"
            if self.time_reference_subtraction_performed
            else "No",
            "Group Clutter Removal": "Yes" if self.clutter_removal_performed else "No",
            "Global Normalize α": f"{self.global_scale_alpha:.6g}"
            if self.global_scale_alpha
            else "N/A",
            "Week3 Notes": "; ".join(self.notes) if self.notes else "None",
        }


def hamming_window(n: int) -> np.ndarray:
    """Hamming weights w[k] = 0.54 - 0.46 cos(2πk/(N-1)), length N."""
    if n <= 0:
        return np.asarray([], dtype=float)
    if n == 1:
        return np.asarray([1.0], dtype=float)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def apply_hamming(s21: np.ndarray, axis: int = 0) -> np.ndarray:
    """Multiply frequency-domain traces by a Hamming window along `axis`."""
    n = s21.shape[axis]
    weights = hamming_window(n)
    shape = [1] * s21.ndim
    shape[axis] = n
    return s21 * weights.reshape(shape)


def build_time_axis(
    frequencies_hz: np.ndarray, n_time: int | None = None
) -> tuple[np.ndarray, float, float, float]:
    """
    Build t[n] = n Δt with Δt = 1/(N Δf), T_period = 1/Δf, B = (N-1)Δf.

    Returns (time_s, delta_t_s, bandwidth_hz, period_s).
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    if frequencies_hz.ndim != 1 or frequencies_hz.size < 2:
        raise ValueError("Need a 1D frequency axis with at least two samples.")

    delta_f = float(np.mean(np.diff(frequencies_hz)))
    if delta_f <= 0:
        raise ValueError("Frequency spacing must be positive.")

    n = int(n_time if n_time is not None else frequencies_hz.shape[0])
    delta_t = 1.0 / (n * delta_f)
    time_s = np.arange(n, dtype=float) * delta_t
    bandwidth = float(frequencies_hz[-1] - frequencies_hz[0])
    period = 1.0 / delta_f
    return time_s, delta_t, bandwidth, period


def ifft_to_time(
    s21_freq: np.ndarray,
    frequencies_hz: np.ndarray,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """IFFT along the frequency axis and return (time_signal, time_s, Δt, B, T)."""
    time_signal = frequency_to_time(s21_freq, frequencies_hz, axis=axis, zero_padding=0)
    time_s, delta_t, bandwidth, period = build_time_axis(
        frequencies_hz, n_time=time_signal.shape[axis]
    )
    return time_signal, time_s, delta_t, bandwidth, period


def group_mean_clutter_removal(
    time_signals: np.ndarray, axis: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """
    μ_G[n] = mean_p s_p[n];  s_clean = s - μ_G  (Hammouch / Blanco-Angulo style).

    `time_signals` shape (n_time, n_channels) when axis=-1.
    """
    if time_signals.ndim == 1:
        return time_signals.copy(), time_signals.copy()

    mu = np.mean(time_signals, axis=axis, keepdims=True)
    return time_signals - mu, np.squeeze(mu, axis=axis)


def global_normalize(
    time_signals: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """One α = max_{p,n} |s|; return s/α (preserves relative amplitudes across channels)."""
    peak = float(np.max(np.abs(time_signals))) if time_signals.size else 0.0
    if peak <= 0.0:
        return None, None
    return time_signals / peak, peak


def circular_tx_rx_coordinates(
    n_channels: int,
    antenna_radius_m: float,
    angle_offset_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Place Tx=Rx antennas on a circle (monostatic / same-port approximation).

    Returns Tx and Rx arrays of shape (n_channels, 2) in meters (x, y).
    """
    if n_channels <= 0:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty

    angles = np.deg2rad(angle_offset_deg + np.arange(n_channels) * (360.0 / n_channels))
    coords = np.column_stack(
        (antenna_radius_m * np.cos(angles), antenna_radius_m * np.sin(angles))
    )
    return coords.copy(), coords.copy()


def process_week3_time_domain(
    frequencies_hz: np.ndarray,
    s21_filtered: np.ndarray,
    *,
    reference_filtered: np.ndarray | None = None,
    enable_clutter_removal: bool = True,
    enable_global_normalize: bool = True,
) -> Week3TimeDomainResult:
    """
    Week 3 path on filtered complex S21 (freq × channels or 1D).

    Order: Hamming (matched) → IFFT → optional time ref subtract → optional
    group-mean clutter → optional global normalize.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    s21_filtered = np.asarray(s21_filtered, dtype=complex)
    notes: list[str] = []

    if s21_filtered.ndim == 1:
        s21_filtered = s21_filtered.reshape(-1, 1)
    if s21_filtered.shape[0] != frequencies_hz.shape[0]:
        raise ValueError("S21 frequency length must match frequencies_hz.")

    n_channels = s21_filtered.shape[1]
    window = hamming_window(s21_filtered.shape[0]).reshape(-1, 1)
    s21_hamming = s21_filtered * window

    s21_reference_hamming = None
    matched_windowing = False
    if reference_filtered is not None:
        reference_filtered = np.asarray(reference_filtered, dtype=complex)
        if reference_filtered.ndim == 1:
            reference_filtered = reference_filtered.reshape(-1, 1)
        if reference_filtered.shape != s21_filtered.shape:
            notes.append(
                "Matched Hamming/reference skipped: reference shape did not match target."
            )
            reference_filtered = None
        else:
            s21_reference_hamming = reference_filtered * window
            matched_windowing = True

    time_filtered, time_s, delta_t, bandwidth, period = ifft_to_time(
        s21_filtered, frequencies_hz, axis=0
    )
    time_hamming, _, _, _, _ = ifft_to_time(s21_hamming, frequencies_hz, axis=0)

    time_reference_hamming = None
    s21_diff_freq = None
    time_diff = None
    time_ref_done = False
    if (
        matched_windowing
        and s21_reference_hamming is not None
        and reference_filtered is not None
    ):
        s21_diff_freq = s21_hamming - s21_reference_hamming
        time_reference_hamming, _, _, _, _ = ifft_to_time(
            s21_reference_hamming, frequencies_hz, axis=0
        )
        time_diff = time_hamming - time_reference_hamming
        time_ref_done = True
    else:
        notes.append(
            "Time-domain reference subtraction skipped (no matching reference)."
        )

    working_time = time_diff if time_diff is not None else time_hamming

    time_clutter = None
    clutter_done = False
    if enable_clutter_removal and n_channels >= 2:
        time_clutter, _ = group_mean_clutter_removal(working_time, axis=1)
        clutter_done = True
    else:
        time_clutter = working_time.copy()
        if enable_clutter_removal and n_channels < 2:
            notes.append(
                "Group-mean clutter removal skipped (need ≥2 comparable channels)."
            )

    time_norm = None
    alpha = None
    if enable_global_normalize:
        time_norm, alpha = global_normalize(time_clutter)
        if time_norm is None:
            notes.append("Global normalization skipped (zero peak magnitude).")

    delta_f = float(np.mean(np.diff(frequencies_hz)))

    return Week3TimeDomainResult(
        frequencies_hz=frequencies_hz.copy(),
        delta_f_hz=delta_f,
        time_s=time_s,
        delta_t_s=delta_t,
        bandwidth_hz=bandwidth,
        period_s=period,
        s21_filtered=s21_filtered,
        s21_hamming=s21_hamming,
        s21_reference_hamming=s21_reference_hamming,
        s21_reference_subtracted_freq=s21_diff_freq,
        time_filtered=time_filtered,
        time_hamming=time_hamming,
        time_reference_hamming=time_reference_hamming,
        time_reference_subtracted=time_diff,
        time_clutter_removed=time_clutter,
        time_global_normalized=time_norm,
        global_scale_alpha=alpha,
        matched_windowing_performed=matched_windowing,
        time_reference_subtraction_performed=time_ref_done,
        clutter_removal_performed=clutter_done,
        notes=notes,
    )

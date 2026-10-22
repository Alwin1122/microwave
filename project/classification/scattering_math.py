"""Physics observables for a first visit, when no earlier healthy scan exists.

The empty-chamber scan is subtracted first. What remains is still dominated by
the skin and fat shell, which look almost the same from every antenna. That
common part is removed by subtracting the antenna mean (the same step used
before imaging). The leftovers are treated as a compact-scatterer problem.

Three descriptions are used:

1. Scattering-operator spectrum. The breast-window time signals form a matrix
   B (time by antenna). Its singular values σ_i are the strengths of the
   orthogonal scattering patterns (the monostatic analogue of the
   decomposition of the time-reversal operator). With λ_i = σ_i² / Σ σ_j²,

       λ_1              energy in the strongest pattern
       R_eff = 1 / Σ λ_i²    effective number of patterns

   A compact inclusion raises λ_1 and lowers R_eff. Extended fibroglandular
   tissue spreads the energy across many patterns.

2. Frequency slope. On the residual, let p(f) be the mean magnitude across
   antennas. Ordinary least squares fits

       log10 p(f) = α + β log10(f / 1 GHz)

   over 2–6 GHz. Under the Born / Rayleigh approximation the scattered field
   of a small sphere grows as f², so β is near 2. Larger inclusions move β
   down toward the optical regime. The scale-free slope does not depend on
   the instrument gain, which differs between dataset generations.

3. Time compactness and antenna concentration. The energy-weighted delay of
   the breast-window power has skewness and excess kurtosis; a point echo is
   more leptokurtic than distributed clutter. The angular resultant length

       R = |Σ_k p_k exp(i θ_k)| / Σ_k p_k

   is high when one antenna direction carries the residual.

A closed-form range follows from the round-trip delay of the brightest
antenna, t_peak:

       d = R_ant - c t_peak / 2

with the antenna angle of that channel. This is an identification estimate,
not a trained localiser.
"""

from __future__ import annotations

import numpy as np

from classification.features import BREAST_T_END_S, BREAST_T_START_S, breast_time_mask
from reconstruction.bmid_geometry import remove_rotational_mean
from reconstruction.ifft import frequency_to_time

_EPS = 1e-20
SLOPE_BAND_HZ = (2.0e9, 6.0e9)
F_REF_HZ = 1.0e9

# Classifier inputs. Peak antenna index and lab-frame angle are identification
# outputs only: they are not scale-free tumour evidence.
MATH_FEATURES = (
    "math_lambda1",
    "math_lambda2",
    "math_effective_rank",
    "math_spec_entropy",
    "math_slope_beta",
    "math_slope_r2",
    "math_born_log",
    "math_kurtosis",
    "math_skewness",
    "math_delay_mean_ns",
    "math_delay_std_ns",
    "math_delay_peak_ns",
    "math_angular_R",
    "math_sparsity",
    "math_band_ratio_log",
)


def _zeros() -> dict[str, float]:
    return {name: 0.0 for name in MATH_FEATURES}


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and r² of y on x. Returns (0, 0) when x has no spread."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or float(np.std(x)) < 1e-12:
        return 0.0, 0.0
    x0 = x - x.mean()
    y0 = y - y.mean()
    var = float(np.dot(x0, x0))
    if var <= 0:
        return 0.0, 0.0
    slope = float(np.dot(x0, y0) / var)
    pred = slope * x0
    ss_tot = float(np.dot(y0, y0))
    r2 = 0.0 if ss_tot <= 0 else float(1.0 - np.sum((y0 - pred) ** 2) / ss_tot)
    return slope, float(np.clip(r2, 0.0, 1.0))


def _moments(t: np.ndarray, energy: np.ndarray) -> tuple[float, float, float, float]:
    """Energy-weighted mean, std, skewness and excess kurtosis of t."""
    t = np.asarray(t, dtype=float)
    e = np.clip(np.asarray(energy, dtype=float), 0.0, None)
    total = float(e.sum())
    if total <= 0 or t.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    p = e / total
    mean = float(np.dot(p, t))
    var = float(np.dot(p, (t - mean) ** 2))
    if var <= 1e-30:
        return mean, 0.0, 0.0, 0.0
    z = (t - mean) / np.sqrt(var)
    skew = float(np.dot(p, z**3))
    kurt = float(np.dot(p, z**4) - 3.0)
    return mean, float(np.sqrt(var)), skew, kurt


def scattering_descriptors(calibrated: np.ndarray, frequencies: np.ndarray) -> dict[str, float]:
    """Scale-free descriptors of one empty-chamber-calibrated sinogram.

    ``calibrated`` has shape (frequency, antenna). The rotational mean is
    removed inside this function.
    """
    cal = np.asarray(calibrated, dtype=complex)
    freqs = np.asarray(frequencies, dtype=float)
    if cal.ndim != 2 or cal.shape[0] != freqs.size or freqs.size < 8:
        raise ValueError("calibrated must be (frequency, antenna) and match frequencies")
    feats = _zeros()
    feats["math_peak_antenna"] = 0.0
    residual = remove_rotational_mean(cal)
    power = np.abs(residual) ** 2
    if float(power.sum()) <= 0:
        return feats

    time = frequency_to_time(residual, freqs)
    in_breast = breast_time_mask(freqs)
    gated = time[in_breast]
    df = float(np.mean(np.diff(freqs)))
    t_all = np.arange(freqs.size, dtype=float) / (freqs.size * df)
    t_breast = t_all[in_breast]
    if gated.size == 0 or float(np.sum(np.abs(gated) ** 2)) <= 0:
        return feats

    singular = np.linalg.svd(gated, compute_uv=False)
    singular = np.clip(singular.real, 0.0, None)
    mode_power = singular**2
    total_mode = float(mode_power.sum()) + _EPS
    lam = mode_power / total_mode
    feats["math_lambda1"] = float(lam[0])
    feats["math_lambda2"] = float(lam[1]) if lam.size > 1 else 0.0
    feats["math_effective_rank"] = float(1.0 / (np.sum(lam**2) + _EPS))
    positive = lam[lam > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    feats["math_spec_entropy"] = float(entropy / np.log(max(lam.size, 2)))

    band = (freqs >= SLOPE_BAND_HZ[0]) & (freqs <= SLOPE_BAND_HZ[1])
    magnitude = np.mean(np.abs(residual), axis=1)
    if int(band.sum()) >= 3 and float(magnitude[band].max()) > 0:
        slope, r2 = _ols_slope(np.log10(freqs[band] / F_REF_HZ), np.log10(magnitude[band] + _EPS))
        feats["math_slope_beta"] = slope
        feats["math_slope_r2"] = r2
        born = magnitude[band] / (freqs[band] / F_REF_HZ) ** 2
        feats["math_born_log"] = float(np.log10(np.median(born) + _EPS))

    energy_t = np.sum(np.abs(gated) ** 2, axis=1)
    mean_t, std_t, skew, kurt = _moments(t_breast, energy_t)
    feats["math_delay_mean_ns"] = mean_t * 1e9
    feats["math_delay_std_ns"] = std_t * 1e9
    feats["math_skewness"] = skew
    feats["math_kurtosis"] = kurt
    bright = int(np.argmax(np.sum(np.abs(gated) ** 2, axis=0)))
    peak_sample = int(np.argmax(np.abs(gated[:, bright]) ** 2))
    feats["math_peak_antenna"] = float(bright)
    feats["math_delay_peak_ns"] = float(t_breast[peak_sample] * 1e9)

    per_antenna = power.sum(axis=0)
    n_ant = per_antenna.size
    theta = np.linspace(0.0, np.deg2rad(355.0), n_ant, endpoint=True)
    resultant = np.abs(np.sum(per_antenna * np.exp(1j * theta)))
    feats["math_angular_R"] = float(resultant / (per_antenna.sum() + _EPS))
    order = np.sort(per_antenna)
    brightest = max(1, n_ant // 8)
    feats["math_sparsity"] = float(order[-brightest:].sum() / (per_antenna.sum() + _EPS))

    low = (freqs >= 1.0e9) & (freqs < 4.0e9)
    high = (freqs >= 4.0e9) & (freqs <= 8.0e9)
    low_p = float(power[low].sum()) if np.any(low) else 0.0
    high_p = float(power[high].sum()) if np.any(high) else 0.0
    feats["math_band_ratio_log"] = float(np.log10((high_p + _EPS) / (low_p + _EPS)))
    return feats


def peak_antenna(calibrated: np.ndarray, frequencies: np.ndarray) -> tuple[int, float]:
    """Antenna index and breast-window peak delay (seconds) of the residual."""
    cal = np.asarray(calibrated, dtype=complex)
    freqs = np.asarray(frequencies, dtype=float)
    residual = remove_rotational_mean(cal)
    time = frequency_to_time(residual, freqs)
    in_breast = breast_time_mask(freqs)
    gated = time[in_breast]
    df = float(np.mean(np.diff(freqs)))
    t_breast = np.arange(freqs.size, dtype=float)[in_breast] / (freqs.size * df)
    if gated.size == 0:
        return 0, float(BREAST_T_START_S)
    antenna = int(np.argmax(np.sum(np.abs(gated) ** 2, axis=0)))
    sample = int(np.argmax(np.abs(gated[:, antenna]) ** 2))
    delay = float(t_breast[sample]) if t_breast.size else float(BREAST_T_START_S)
    delay = float(np.clip(delay, BREAST_T_START_S, BREAST_T_END_S))
    return antenna, delay


def range_bearing_cm(
    delay_s: float,
    angle_rad: float,
    antenna_radius_m: float,
    wave_speed: float,
) -> tuple[float, float]:
    """Tumour (x, y) in centimetres from a monostatic round-trip delay.

    d = R_ant - c t / 2, then (x, y) = d (cos θ, sin θ). d is clipped to the
    antenna disc. The angle must already be in the lab frame used by the
    ground-truth coordinates.
    """
    if antenna_radius_m <= 0 or wave_speed <= 0:
        raise ValueError("antenna radius and wave speed must be positive")
    radial_m = float(antenna_radius_m - wave_speed * float(delay_s) / 2.0)
    radial_m = float(np.clip(radial_m, 0.0, antenna_radius_m))
    x_cm = float(radial_m * np.cos(angle_rad) * 100.0)
    y_cm = float(radial_m * np.sin(angle_rad) * 100.0)
    return x_cm, y_cm

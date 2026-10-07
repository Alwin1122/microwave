"""Per-scan feature vectors for tumor detection and characterization.

Signal features come from the empty-chamber-subtracted S-parameters. Image
features come from the blind top-ranked ROI of a reconstruction; no label or
ground-truth position is used anywhere in this module.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from quality.characterization import characterize_tumor_region
from quality.metrics import compute_image_ccr, compute_image_fwhm_pixels
from reconstruction.ifft import frequency_to_time
from roi.roi_detector import classify_tumor_candidate, detect_rois, tumor_likelihood_features

_EPS = 1e-20

BAND_EDGES_HZ = tuple(float(edge) * 1e9 for edge in range(1, 9))
TIME_BIN_NS = 1.0
N_TIME_BINS = 10
N_ANGULAR_HARMONICS = 4


def _log10(value: float) -> float:
    return float(np.log10(max(float(value), _EPS)))


def calibrate_empty(s_parameters: np.ndarray, empty_reference: np.ndarray) -> np.ndarray:
    """Subtract the empty-chamber scan recorded for the same session."""
    scan = np.asarray(s_parameters, dtype=complex)
    ref = np.asarray(empty_reference, dtype=complex)
    if scan.shape != ref.shape:
        raise ValueError(f"reference shape {ref.shape} does not match scan shape {scan.shape}")
    return scan - ref


def signal_features(
    raw: np.ndarray,
    calibrated: np.ndarray,
    frequencies: np.ndarray,
    prefix: str = "sig",
) -> dict[str, float]:
    """Band, time-window and angular-asymmetry descriptors of the calibrated signal."""
    raw = np.asarray(raw, dtype=complex)
    cal = np.asarray(calibrated, dtype=complex)
    freqs = np.asarray(frequencies, dtype=float)
    power = np.abs(cal) ** 2
    feats: dict[str, float] = {}

    feats["sig_cal_to_raw_log"] = _log10(power.sum() / (np.sum(np.abs(raw) ** 2) + _EPS))

    rotational = cal - cal.mean(axis=1, keepdims=True)
    rot_power = np.abs(rotational) ** 2
    feats["sig_rot_residual"] = float(rot_power.sum() / (power.sum() + _EPS))

    for band, (lo, hi) in enumerate(zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]), start=1):
        sel = (freqs >= lo) & (freqs < hi if band < len(BAND_EDGES_HZ) - 1 else freqs <= hi)
        per_antenna = power[sel].mean(axis=0)
        mean = float(per_antenna.mean())
        feats[f"sig_band{band}_log"] = _log10(mean)
        feats[f"sig_band{band}_cv"] = float(per_antenna.std() / (mean + _EPS))
        feats[f"sig_band{band}_rot"] = float(rot_power[sel].sum() / (power[sel].sum() + _EPS))

    time_signals = np.abs(frequency_to_time(cal, freqs)) ** 2
    dt_ns = 1e9 / (float(np.mean(np.diff(freqs))) * freqs.size)
    per_bin = max(1, int(round(TIME_BIN_NS / dt_ns)))
    total_time = float(time_signals.sum()) + _EPS
    for k in range(N_TIME_BINS):
        window = time_signals[k * per_bin : (k + 1) * per_bin]
        feats[f"sig_t{k}_log"] = _log10(window.mean())
        feats[f"sig_t{k}_frac"] = float(window.sum() / total_time)

    per_antenna_total = power.sum(axis=0)
    spectrum = np.abs(np.fft.rfft(per_antenna_total))
    for k in range(1, N_ANGULAR_HARMONICS + 1):
        feats[f"sig_ang_h{k}"] = float(spectrum[k] / (spectrum[0] + _EPS)) if k < spectrum.size else 0.0
    feats["sig_ant_cv"] = float(per_antenna_total.std() / (per_antenna_total.mean() + _EPS))
    feats["sig_ant_max_over_median"] = float(per_antenna_total.max() / (np.median(per_antenna_total) + _EPS))
    if prefix != "sig":
        feats = {prefix + name[3:]: value for name, value in feats.items()}
    return feats


def _entropy(arr: np.ndarray) -> float:
    flat = np.clip(np.asarray(arr, dtype=float).ravel(), 0.0, None)
    total = flat.sum()
    if total <= 0:
        return 0.0
    p = flat / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(flat.size))


def image_features(
    image: np.ndarray,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    prefix: str,
) -> dict[str, float]:
    """Global contrast plus blind top-ROI shape, position and tumor-likelihood cues."""
    arr = np.asarray(image, dtype=float)
    mean = float(np.mean(arr)) + _EPS
    px_cm = 100.0 * abs(x_span[1] - x_span[0]) / max(arr.shape[1] - 1, 1)
    feats = {
        "peak_over_mean": float(np.max(arr) / mean),
        "std_over_mean": float(np.std(arr) / mean),
        "ccr": compute_image_ccr(arr),
        "fwhm_cm": float(compute_image_fwhm_pixels(arr) * px_cm),
        "entropy": _entropy(arr),
        "top1pct_over_median": float(np.percentile(arr, 99) / (np.median(arr) + _EPS)),
    }
    smooth = gaussian_filter(arr, 1.0)
    iy, ix = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    feats["peak_x_cm"] = float(100.0 * np.linspace(x_span[0], x_span[1], arr.shape[1])[ix])
    feats["peak_y_cm"] = float(100.0 * np.linspace(y_span[0], y_span[1], arr.shape[0])[iy])
    feats["peak_r_cm"] = float(np.hypot(feats["peak_x_cm"], feats["peak_y_cm"]))

    rois = detect_rois(arr, x_span=x_span, y_span=y_span, max_rois=4, min_score_ratio=0.28)
    primary = rois[0]
    likelihood = tumor_likelihood_features(arr, primary)
    candidate = classify_tumor_candidate(arr, primary)
    shape = characterize_tumor_region(arr, primary, x_span=x_span, y_span=y_span)
    blind = [classify_tumor_candidate(arr, roi) for roi in rois]

    feats.update(
        {
            "x_cm": shape.centroid_x_cm,
            "y_cm": shape.centroid_y_cm,
            "r_cm": shape.radial_offset_cm,
            "eq_diam_cm": shape.equivalent_diameter_cm,
            "area_cm2": shape.area_cm2,
            "aspect": shape.aspect_ratio,
            "roi_peak_to_mean": shape.peak_to_mean,
            "roi_scr": shape.local_scr,
            "compactness": shape.compactness,
            "eccentricity": shape.eccentricity_proxy,
            "local_contrast": likelihood["local_contrast"],
            "ring_excess": likelihood["ring_excess"],
            "size_score": likelihood["size_score"],
            "suspicion": likelihood["suspicion"],
            "rule_confidence": candidate.confidence,
            "rule_flag": 1.0 if candidate.is_tumor_candidate else 0.0,
            "rule_any_flag": 1.0 if any(item.is_tumor_candidate for item in blind) else 0.0,
            "rule_max_confidence": float(max(item.confidence for item in blind)),
            "n_rois": float(len(rois)),
            "second_score_ratio": float(rois[1].score / (rois[0].score + _EPS)) if len(rois) > 1 else 0.0,
        }
    )
    return {f"{prefix}_{key}": float(value) for key, value in feats.items()}

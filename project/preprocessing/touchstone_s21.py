from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from data_loader.dataset_info import MicrowaveDataset
from data_loader.touchstone_loader import TouchstoneHeaderInfo, load_touchstone_s21_trace
from preprocessing.filtering import apply_noise_filter
from utils.exceptions import InvalidFrequencyError, InvalidSParameterError


@dataclass
class TouchstoneS21ValidationReport:
    filter_used: str
    corrected_samples: int
    averaging_performed: bool
    reference_subtraction_performed: bool
    interpolation_performed: bool
    normalized_copy_generated: bool
    phase_unwrap_adjustments: int
    distortion_ratio: float
    notes: list[str] = field(default_factory=list)

    def to_display_dict(self) -> dict:
        notes = "; ".join(self.notes) if self.notes else "None"
        return {
            "Module 2 Filter": self.filter_used,
            "Corrected Samples": str(self.corrected_samples),
            "Averaging Performed": "Yes" if self.averaging_performed else "No",
            "Reference Subtraction": "Yes" if self.reference_subtraction_performed else "No",
            "Interpolation Performed": "Yes" if self.interpolation_performed else "No",
            "Normalized Copy Generated": "Yes" if self.normalized_copy_generated else "No",
            "Phase Unwrap Adjustments": str(self.phase_unwrap_adjustments),
            "Distortion Ratio": f"{self.distortion_ratio:.3f}",
            "Validation Notes": notes,
        }


@dataclass
class TouchstoneS21ProcessingResult:
    header: TouchstoneHeaderInfo
    raw_frequencies_hz: np.ndarray
    raw_s21: np.ndarray
    uniform_frequencies_hz: np.ndarray
    raw_magnitude_db: np.ndarray
    wrapped_phase_rad: np.ndarray
    wrapped_phase_deg: np.ndarray
    unwrapped_phase_rad: np.ndarray
    unwrapped_phase_deg: np.ndarray
    corrected_s21: np.ndarray
    averaged_s21: np.ndarray | None
    reference_subtracted_s21: np.ndarray | None
    filtered_s21: np.ndarray
    windowed_s21: np.ndarray
    normalized_s21: np.ndarray | None
    delta_f_hz: np.ndarray
    report: TouchstoneS21ValidationReport


def _validate_frequency_and_s21_lengths(frequencies_hz: np.ndarray, s21: np.ndarray) -> None:
    if frequencies_hz.ndim != 1:
        raise InvalidFrequencyError("Frequency vector must be one-dimensional.")
    if s21.ndim != 1:
        raise InvalidSParameterError("S21 trace must be one-dimensional.")
    if frequencies_hz.shape[0] != s21.shape[0]:
        raise InvalidSParameterError(
            "Frequency vector and S21 trace must contain the same number of samples."
        )
    if frequencies_hz.size == 0:
        raise InvalidFrequencyError("Frequency vector is empty.")


def _validate_numeric_content(frequencies_hz: np.ndarray, s21: np.ndarray) -> None:
    if not np.issubdtype(frequencies_hz.dtype, np.number):
        raise InvalidFrequencyError("Frequency vector contains non-numeric samples.")
    if np.any(~np.isfinite(frequencies_hz)):
        raise InvalidFrequencyError("Frequency vector contains missing, NaN, or infinite values.")
    if np.any(frequencies_hz < 0):
        raise InvalidFrequencyError("Frequency vector contains negative values.")
    if np.any(~np.isfinite(s21.real)) or np.any(~np.isfinite(s21.imag)):
        raise InvalidSParameterError("S21 contains missing, NaN, or infinite values.")


def _sort_and_validate_unique_frequencies(
    frequencies_hz: np.ndarray,
    s21: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(frequencies_hz)
    frequencies_sorted = np.asarray(frequencies_hz[order], dtype=float)
    s21_sorted = np.asarray(s21[order], dtype=complex)

    if np.any(np.diff(frequencies_sorted) <= 0):
        raise InvalidFrequencyError("Frequency vector contains duplicate frequency points.")

    return frequencies_sorted, s21_sorted


def _is_uniform_spacing(delta_f_hz: np.ndarray) -> bool:
    if delta_f_hz.size <= 1:
        return True
    return bool(np.allclose(delta_f_hz, delta_f_hz[0], rtol=1e-5, atol=max(abs(delta_f_hz[0]) * 1e-8, 1e-9)))


def _unwrap_phase_degrees(phase_deg: np.ndarray) -> tuple[np.ndarray, int]:
    if phase_deg.size == 0:
        return phase_deg.copy(), 0

    unwrapped = phase_deg.astype(float).copy()
    adjustments = 0
    for index in range(1, unwrapped.shape[0]):
        difference = unwrapped[index] - unwrapped[index - 1]
        if difference > 180.0:
            unwrapped[index:] -= 360.0
            adjustments += 1
        elif difference < -180.0:
            unwrapped[index:] += 360.0
            adjustments += 1
    return unwrapped, adjustments


def _interpolate_complex_trace(
    frequencies_hz: np.ndarray,
    s21: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    delta_f = np.diff(frequencies_hz)
    if _is_uniform_spacing(delta_f):
        return frequencies_hz.copy(), s21.copy(), False

    uniform_frequencies = np.linspace(frequencies_hz[0], frequencies_hz[-1], frequencies_hz.shape[0])
    real_interp = np.interp(uniform_frequencies, frequencies_hz, s21.real)
    imag_interp = np.interp(uniform_frequencies, frequencies_hz, s21.imag)
    return uniform_frequencies, real_interp + 1j * imag_interp, True


def _hampel_mask(values: np.ndarray, window_size: int = 3, n_sigmas: float = 3.0) -> np.ndarray:
    n = values.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    window_size = max(1, int(window_size))
    n_sigmas = max(float(n_sigmas), 0.0)
    scale = 1.4826

    for index in range(n):
        start = max(0, index - window_size)
        stop = min(n, index + window_size + 1)
        window = values[start:stop]
        median = np.median(window)
        deviation = np.median(np.abs(window - median))
        threshold = n_sigmas * scale * deviation
        if deviation > 0 and abs(values[index] - median) > threshold:
            neighbours = np.delete(window, min(index - start, window.shape[0] - 1))
            if neighbours.size and abs(values[index] - np.median(neighbours)) > threshold:
                mask[index] = True
    return mask


def _correct_isolated_samples(
    s21: np.ndarray,
    method: str = "hampel",
    window_size: int = 3,
    n_sigmas: float = 3.0,
) -> tuple[np.ndarray, int]:
    if method.lower() not in {"hampel", "median", "local"}:
        raise ValueError(f"Unsupported invalid-point correction method '{method}'.")

    real_mask = _hampel_mask(s21.real, window_size=window_size, n_sigmas=n_sigmas)
    imag_mask = _hampel_mask(s21.imag, window_size=window_size, n_sigmas=n_sigmas)
    combined_mask = real_mask | imag_mask
    if not np.any(combined_mask):
        return s21.copy(), 0

    corrected_real = s21.real.copy()
    corrected_imag = s21.imag.copy()
    for index in np.flatnonzero(combined_mask):
        start = max(0, index - window_size)
        stop = min(s21.shape[0], index + window_size + 1)

        neighbourhood_real = np.delete(corrected_real[start:stop], min(index - start, stop - start - 1))
        neighbourhood_imag = np.delete(corrected_imag[start:stop], min(index - start, stop - start - 1))

        if neighbourhood_real.size:
            corrected_real[index] = np.median(neighbourhood_real)
        if neighbourhood_imag.size:
            corrected_imag[index] = np.median(neighbourhood_imag)

    return corrected_real + 1j * corrected_imag, int(np.count_nonzero(combined_mask))


def _extract_measurement_trace(measurement: MicrowaveDataset | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(measurement, MicrowaveDataset):
        return load_touchstone_s21_trace(measurement.file_path)[:2]

    array = np.asarray(measurement)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            "Repeated measurements provided as arrays must have shape (n_freq, 2) with frequency and complex S21 columns."
        )
    return np.asarray(array[:, 0], dtype=float), np.asarray(array[:, 1], dtype=complex)


def _prepare_external_trace(
    measurement: MicrowaveDataset | np.ndarray,
    expected_frequencies_hz: np.ndarray,
) -> np.ndarray | None:
    raw_frequencies, raw_trace = _extract_measurement_trace(measurement)
    _validate_frequency_and_s21_lengths(raw_frequencies, raw_trace)
    _validate_numeric_content(raw_frequencies, raw_trace)
    raw_frequencies, raw_trace = _sort_and_validate_unique_frequencies(raw_frequencies, raw_trace)
    candidate_frequencies, candidate_trace, _ = _interpolate_complex_trace(raw_frequencies, raw_trace)

    if candidate_frequencies.shape != expected_frequencies_hz.shape:
        return None
    if not np.allclose(candidate_frequencies, expected_frequencies_hz, rtol=1e-9, atol=1e-3):
        return None
    return candidate_trace


def _complex_average(
    base_trace: np.ndarray,
    repeated_measurements: Iterable[MicrowaveDataset | np.ndarray] | None,
    expected_frequencies_hz: np.ndarray,
) -> tuple[np.ndarray | None, bool, list[str]]:
    if not repeated_measurements:
        return None, False, []

    traces = [base_trace]
    skipped = 0
    for measurement in repeated_measurements:
        candidate = _prepare_external_trace(measurement, expected_frequencies_hz)
        if candidate is None:
            skipped += 1
            continue
        traces.append(candidate)

    notes: list[str] = []
    if skipped:
        notes.append(f"Skipped {skipped} repeated measurement(s) because the frequency points did not match.")
    if len(traces) == 1:
        return None, False, notes
    return np.mean(np.vstack(traces), axis=0), True, notes


def _complex_reference_subtraction(
    signal: np.ndarray,
    reference_measurement: MicrowaveDataset | np.ndarray | None,
    expected_frequencies_hz: np.ndarray,
) -> tuple[np.ndarray | None, bool, list[str]]:
    if reference_measurement is None:
        return None, False, ["Reference subtraction skipped because no suitable reference file was supplied."]

    reference_trace = _prepare_external_trace(reference_measurement, expected_frequencies_hz)
    if reference_trace is None:
        return None, False, ["Reference subtraction skipped because the reference frequency points did not match."]

    return signal - reference_trace, True, []


def _normalized_copy(s21: np.ndarray) -> np.ndarray | None:
    peak = float(np.max(np.abs(s21))) if s21.size else 0.0
    if peak <= 0.0:
        return None
    return s21 / peak


def _distortion_ratio(reference: np.ndarray, processed: np.ndarray) -> float:
    baseline = np.linalg.norm(reference)
    if baseline == 0:
        return 0.0
    return float(np.linalg.norm(processed - reference) / baseline)


def process_touchstone_s21_dataset(dataset: MicrowaveDataset, config) -> TouchstoneS21ProcessingResult:
    raw_frequencies_hz, raw_s21, header = load_touchstone_s21_trace(dataset.file_path)
    _validate_frequency_and_s21_lengths(raw_frequencies_hz, raw_s21)
    _validate_numeric_content(raw_frequencies_hz, raw_s21)
    raw_frequencies_hz, raw_s21 = _sort_and_validate_unique_frequencies(raw_frequencies_hz, raw_s21)

    raw_magnitude_db = 20.0 * np.log10(np.abs(raw_s21) + 1e-12)
    wrapped_phase_rad = np.arctan2(raw_s21.imag, raw_s21.real)
    wrapped_phase_deg = np.rad2deg(wrapped_phase_rad)
    unwrapped_phase_deg, unwrap_adjustments = _unwrap_phase_degrees(wrapped_phase_deg)
    unwrapped_phase_rad = np.deg2rad(unwrapped_phase_deg)

    uniform_frequencies_hz, uniform_raw_s21, interpolation_performed = _interpolate_complex_trace(
        raw_frequencies_hz,
        raw_s21,
    )
    corrected_s21, corrected_samples = _correct_isolated_samples(
        uniform_raw_s21,
        method=getattr(config, "spike_detection_method", "hampel"),
        **getattr(config, "spike_detection_kwargs", {}),
    )

    averaged_s21, averaging_performed, averaging_notes = _complex_average(
        corrected_s21,
        getattr(config, "repeated_measurements", None),
        uniform_frequencies_hz,
    )
    working_signal = averaged_s21 if averaged_s21 is not None else corrected_s21

    reference_subtracted_s21, reference_subtraction_performed, reference_notes = _complex_reference_subtraction(
        working_signal,
        getattr(config, "reference_dataset", None),
        uniform_frequencies_hz,
    )
    working_signal = reference_subtracted_s21 if reference_subtracted_s21 is not None else working_signal

    filtered_2d = apply_noise_filter(
        working_signal.reshape(-1, 1),
        method=config.filter_method,
        **config.filter_kwargs,
    )
    filtered_s21 = filtered_2d[:, 0]
    windowed_s21 = filtered_s21 * np.hamming(filtered_s21.shape[0])
    normalized_s21 = _normalized_copy(filtered_s21) if getattr(config, "generate_normalized_copy", True) else None

    _validate_frequency_and_s21_lengths(uniform_frequencies_hz, filtered_s21)
    _validate_numeric_content(uniform_frequencies_hz, filtered_s21)

    delta_f_hz = np.diff(uniform_frequencies_hz)
    distortion_ratio = _distortion_ratio(corrected_s21, filtered_s21)
    notes = []
    notes.extend(averaging_notes)
    notes.extend(reference_notes)
    if not _is_uniform_spacing(delta_f_hz):
        raise InvalidFrequencyError("Processed frequency vector is not uniformly spaced.")
    if distortion_ratio > 0.75:
        notes.append("Processed curve may be excessively distorted; reduce filter strength.")
    if corrected_samples == 0:
        notes.append("No isolated invalid points required correction.")

    report = TouchstoneS21ValidationReport(
        filter_used=config.filter_method,
        corrected_samples=corrected_samples,
        averaging_performed=averaging_performed,
        reference_subtraction_performed=reference_subtraction_performed,
        interpolation_performed=interpolation_performed,
        normalized_copy_generated=normalized_s21 is not None,
        phase_unwrap_adjustments=unwrap_adjustments,
        distortion_ratio=distortion_ratio,
        notes=notes,
    )

    return TouchstoneS21ProcessingResult(
        header=header,
        raw_frequencies_hz=raw_frequencies_hz,
        raw_s21=raw_s21,
        uniform_frequencies_hz=uniform_frequencies_hz,
        raw_magnitude_db=raw_magnitude_db,
        wrapped_phase_rad=wrapped_phase_rad,
        wrapped_phase_deg=wrapped_phase_deg,
        unwrapped_phase_rad=unwrapped_phase_rad,
        unwrapped_phase_deg=unwrapped_phase_deg,
        corrected_s21=corrected_s21,
        averaged_s21=averaged_s21,
        reference_subtracted_s21=reference_subtracted_s21,
        filtered_s21=filtered_s21,
        windowed_s21=windowed_s21,
        normalized_s21=normalized_s21,
        delta_f_hz=delta_f_hz,
        report=report,
    )
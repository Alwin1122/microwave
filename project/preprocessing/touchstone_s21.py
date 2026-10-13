from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from data_loader.dataset_info import MicrowaveDataset
from data_loader.touchstone_loader import (
    TouchstoneHeaderInfo,
    load_touchstone_s21_trace,
)
from preprocessing.filtering import apply_noise_filter, compare_noise_filters
from preprocessing.week3_time_domain import (
    Week3TimeDomainResult,
    hamming_window,
    process_week3_time_domain,
)
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
    nrmse: float
    spike_detection_method: str
    week3_time_domain_performed: bool = False
    notes: list[str] = field(default_factory=list)
    corrected_sample_indices: list[int] = field(default_factory=list)
    corrected_sample_frequencies_hz: list[float] = field(default_factory=list)
    filter_parameters: dict = field(default_factory=dict)

    @property
    def distortion_ratio(self) -> float:
        """Backward-compatible alias for NRMSE (||e|| / ||ref||)."""
        return self.nrmse

    def to_display_dict(self) -> dict:
        notes = "; ".join(self.notes) if self.notes else "None"
        return {
            "Module 2 Filter": self.filter_used,
            "Filter Parameters": str(self.filter_parameters or "defaults"),
            "Spike Detection": self.spike_detection_method,
            "Corrected Samples": str(self.corrected_samples),
            "Averaging Performed": "Yes" if self.averaging_performed else "No",
            "Reference Subtraction": "Yes"
            if self.reference_subtraction_performed
            else "No",
            "Interpolation Performed": "Yes" if self.interpolation_performed else "No",
            "Normalized Copy Generated": "Yes"
            if self.normalized_copy_generated
            else "No",
            "Phase Unwrap Adjustments": str(self.phase_unwrap_adjustments),
            "NRMSE": f"{self.nrmse:.3f}",
            "Week3 Time-Domain": "Yes" if self.week3_time_domain_performed else "No",
            "Validation Notes": notes,
        }


@dataclass
class TouchstoneS21ProcessingResult:
    header: TouchstoneHeaderInfo
    raw_frequencies_hz: np.ndarray
    raw_s21: np.ndarray
    uniform_frequencies_hz: np.ndarray
    uniform_s21: np.ndarray
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
    week3_result: Week3TimeDomainResult | None = None
    filter_input_s21: np.ndarray | None = None
    filter_comparison: dict[str, dict] = field(default_factory=dict)


def _validate_frequency_and_s21_lengths(
    frequencies_hz: np.ndarray, s21: np.ndarray
) -> None:
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
        raise InvalidFrequencyError(
            "Frequency vector contains missing, NaN, or infinite values."
        )
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
        raise InvalidFrequencyError(
            "Frequency vector contains duplicate frequency points."
        )

    return frequencies_sorted, s21_sorted


def _is_uniform_spacing(delta_f_hz: np.ndarray) -> bool:
    if delta_f_hz.size <= 1:
        return True
    return bool(
        np.allclose(
            delta_f_hz,
            delta_f_hz[0],
            rtol=1e-5,
            atol=max(abs(delta_f_hz[0]) * 1e-8, 1e-9),
        )
    )


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

    uniform_frequencies = np.linspace(
        frequencies_hz[0], frequencies_hz[-1], frequencies_hz.shape[0]
    )
    real_interp = np.interp(uniform_frequencies, frequencies_hz, s21.real)
    imag_interp = np.interp(uniform_frequencies, frequencies_hz, s21.imag)
    return uniform_frequencies, real_interp + 1j * imag_interp, True


def _hampel_mask(
    values: np.ndarray, window_size: int = 3, n_sigmas: float = 3.0
) -> np.ndarray:
    """
    Hampel detector: MAD threshold plus leave-one-out neighbour confirmation.

    The local sigma is floored at the trace-wide noise level (robust std of first
    differences / sqrt(2)); a 7-sample MAD alone is too unstable and flags ordinary noise.
    """
    n = values.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    window_size = max(1, int(window_size))
    n_sigmas = max(float(n_sigmas), 0.0)
    scale = 1.4826

    noise_floor = 0.0
    if n > 2:
        diffs = np.diff(values)
        noise_floor = (
            scale * float(np.median(np.abs(diffs - np.median(diffs)))) / np.sqrt(2.0)
        )

    for index in range(n):
        start = max(0, index - window_size)
        stop = min(n, index + window_size + 1)
        window = values[start:stop]
        median = np.median(window)
        sigma = max(scale * float(np.median(np.abs(window - median))), noise_floor)
        threshold = n_sigmas * sigma
        if sigma > 0 and abs(values[index] - median) > threshold:
            neighbours = np.delete(window, min(index - start, window.shape[0] - 1))
            if (
                neighbours.size
                and abs(values[index] - np.median(neighbours)) > threshold
            ):
                mask[index] = True
    return mask


def _median_spike_mask(
    values: np.ndarray, window_size: int = 3, n_sigmas: float = 3.0
) -> np.ndarray:
    """Flag samples that deviate from the local median beyond n_sigmas * MAD."""
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
        mad = np.median(np.abs(window - median))
        residual = abs(values[index] - median)
        if mad <= 0:
            if residual > 0:
                mask[index] = True
            continue
        if residual > n_sigmas * scale * mad:
            mask[index] = True
    return mask


def _local_spike_mask(
    values: np.ndarray, window_size: int = 3, n_sigmas: float = 3.0
) -> np.ndarray:
    """Flag samples that deviate from the mean of immediate neighbours beyond n_sigmas * std."""
    n = values.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    window_size = max(1, int(window_size))
    n_sigmas = max(float(n_sigmas), 0.0)

    for index in range(n):
        start = max(0, index - window_size)
        stop = min(n, index + window_size + 1)
        neighbours = np.delete(values[start:stop], min(index - start, stop - start - 1))
        if neighbours.size < 1:
            continue
        local_mean = float(np.mean(neighbours))
        residual = abs(values[index] - local_mean)
        if neighbours.size < 2:
            if residual > 0:
                mask[index] = True
            continue
        local_std = float(np.std(neighbours))
        if local_std <= 0:
            if residual > 0:
                mask[index] = True
            continue
        if residual > n_sigmas * local_std:
            mask[index] = True
    return mask


def _spike_mask(
    values: np.ndarray, method: str, window_size: int, n_sigmas: float
) -> np.ndarray:
    detectors = {
        "hampel": _hampel_mask,
        "median": _median_spike_mask,
        "local": _local_spike_mask,
    }
    return detectors[method](values, window_size=window_size, n_sigmas=n_sigmas)


def _correct_isolated_samples(
    s21: np.ndarray,
    method: str = "hampel",
    window_size: int = 3,
    n_sigmas: float = 3.0,
    return_indices: bool = False,
) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, np.ndarray]:
    method_key = method.lower()
    if method_key not in {"hampel", "median", "local"}:
        raise ValueError(f"Unsupported invalid-point correction method '{method}'.")

    real_mask = _spike_mask(
        s21.real, method_key, window_size=window_size, n_sigmas=n_sigmas
    )
    imag_mask = _spike_mask(
        s21.imag, method_key, window_size=window_size, n_sigmas=n_sigmas
    )
    combined_mask = real_mask | imag_mask
    corrected_indices = np.flatnonzero(combined_mask)
    if not corrected_indices.size:
        if return_indices:
            return s21.copy(), 0, corrected_indices
        return s21.copy(), 0

    corrected_real = s21.real.copy()
    corrected_imag = s21.imag.copy()
    for index in corrected_indices:
        start = max(0, index - window_size)
        stop = min(s21.shape[0], index + window_size + 1)

        neighbourhood_real = np.delete(
            corrected_real[start:stop], min(index - start, stop - start - 1)
        )
        neighbourhood_imag = np.delete(
            corrected_imag[start:stop], min(index - start, stop - start - 1)
        )

        if neighbourhood_real.size:
            corrected_real[index] = np.median(neighbourhood_real)
        if neighbourhood_imag.size:
            corrected_imag[index] = np.median(neighbourhood_imag)

    corrected = corrected_real + 1j * corrected_imag
    if return_indices:
        return corrected, int(corrected_indices.size), corrected_indices
    return corrected, int(corrected_indices.size)


def _extract_measurement_trace(
    measurement: MicrowaveDataset | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, TouchstoneHeaderInfo | None]:
    if isinstance(measurement, MicrowaveDataset):
        return load_touchstone_s21_trace(measurement.file_path)

    array = np.asarray(measurement)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            "Repeated measurements provided as arrays must have shape (n_freq, 2) with frequency and complex S21 columns."
        )
    frequencies = np.asarray(array[:, 0].real, dtype=float)
    trace = np.asarray(array[:, 1], dtype=complex)
    return frequencies, trace, None


def _prepare_external_trace(
    measurement: MicrowaveDataset | np.ndarray,
    expected_frequencies_hz: np.ndarray,
    *,
    return_details: bool = False,
):
    """
    Bring an external trace onto the target grid. With ``return_details`` returns
    (trace | None, resampled, header); otherwise just the trace (None if out of range).
    """
    raw_frequencies, raw_trace, header = _extract_measurement_trace(measurement)
    _validate_frequency_and_s21_lengths(raw_frequencies, raw_trace)
    _validate_numeric_content(raw_frequencies, raw_trace)
    raw_frequencies, raw_trace = _sort_and_validate_unique_frequencies(
        raw_frequencies, raw_trace
    )

    def _result(trace: np.ndarray | None, resampled: bool):
        return (trace, resampled, header) if return_details else trace

    if raw_frequencies.shape == expected_frequencies_hz.shape and np.allclose(
        raw_frequencies, expected_frequencies_hz, rtol=1e-9, atol=1e-3
    ):
        return _result(raw_trace, False)

    # Resample onto the target grid only inside the measured range; never extrapolate.
    tolerance_hz = 1e-3
    if (
        raw_frequencies.shape[0] < 2
        or raw_frequencies[0] > expected_frequencies_hz[0] + tolerance_hz
        or raw_frequencies[-1] < expected_frequencies_hz[-1] - tolerance_hz
    ):
        return _result(None, False)
    real_interp = np.interp(expected_frequencies_hz, raw_frequencies, raw_trace.real)
    imag_interp = np.interp(expected_frequencies_hz, raw_frequencies, raw_trace.imag)
    return _result(real_interp + 1j * imag_interp, True)


def _load_matched_reference(
    reference_measurement: MicrowaveDataset | np.ndarray,
    expected_frequencies_hz: np.ndarray,
    target_header: TouchstoneHeaderInfo | None,
    spike_method: str = "hampel",
    spike_kwargs: dict | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    """
    Load the tumour-free reference and put it through the target's chain (grid,
    isolated-sample correction). Rejects a reference with a different reference
    impedance, since target and reference must share calibration.
    """
    trace, resampled, header = _prepare_external_trace(
        reference_measurement, expected_frequencies_hz, return_details=True
    )
    if (
        header is not None
        and target_header is not None
        and not np.isclose(
            header.reference_impedance_ohm, target_header.reference_impedance_ohm
        )
    ):
        return None, [
            "Reference subtraction skipped: reference impedance "
            f"{header.reference_impedance_ohm:g} ohm differs from target "
            f"{target_header.reference_impedance_ohm:g} ohm (calibration mismatch)."
        ]
    if trace is None:
        return None, [
            "Reference subtraction skipped because the reference does not cover the target frequency range."
        ]

    notes: list[str] = []
    if resampled:
        notes.append(
            "Reference was measured on a different frequency grid and was resampled onto the target grid."
        )
    trace, _ = _correct_isolated_samples(
        trace, method=spike_method, **(spike_kwargs or {})
    )
    return trace, notes


def _complex_average(
    base_trace: np.ndarray,
    repeated_measurements: Iterable[MicrowaveDataset | np.ndarray] | None,
    expected_frequencies_hz: np.ndarray,
    spike_method: str = "hampel",
    spike_kwargs: dict | None = None,
    max_sweep_nrmse: float = 0.3,
) -> tuple[np.ndarray | None, bool, list[str]]:
    """
    Complex mean of the base sweep and the repeated sweeps (S21_bar = 1/M * sum S21^(m)).

    Each repeated sweep gets the same isolated-sample correction as the base sweep.
    Sweeps whose NRMSE against the base exceeds ``max_sweep_nrmse`` are treated as a
    different antenna position / phantom condition and are not averaged.
    """
    if not repeated_measurements:
        return None, False, []

    traces = [base_trace]
    out_of_range = 0
    inconsistent: list[str] = []
    accepted_nrmse: list[float] = []
    for number, measurement in enumerate(repeated_measurements, start=1):
        candidate = _prepare_external_trace(measurement, expected_frequencies_hz)
        if candidate is None:
            out_of_range += 1
            continue
        candidate, _ = _correct_isolated_samples(
            candidate, method=spike_method, **(spike_kwargs or {})
        )
        sweep_nrmse = _nrmse(base_trace, candidate)
        if sweep_nrmse > max_sweep_nrmse:
            inconsistent.append(f"#{number} (NRMSE={sweep_nrmse:.3f})")
            continue
        accepted_nrmse.append(sweep_nrmse)
        traces.append(candidate)

    notes: list[str] = []
    if out_of_range:
        notes.append(
            f"Skipped {out_of_range} repeated measurement(s) because they do not cover the target frequency range."
        )
    if inconsistent:
        notes.append(
            f"Skipped repeated measurement(s) {', '.join(inconsistent)}: deviation from the base sweep exceeds "
            f"{max_sweep_nrmse:.2f}, likely a different antenna position or phantom condition."
        )
    if len(traces) == 1:
        return None, False, notes
    notes.append(
        f"Complex-averaged M={len(traces)} sweeps (max sweep NRMSE vs base={max(accepted_nrmse):.3f})."
    )
    return np.mean(np.vstack(traces), axis=0), True, notes


def _complex_reference_subtraction(
    signal: np.ndarray,
    reference_measurement: MicrowaveDataset | np.ndarray | None,
    expected_frequencies_hz: np.ndarray,
    spike_method: str = "hampel",
    spike_kwargs: dict | None = None,
    target_header: TouchstoneHeaderInfo | None = None,
) -> tuple[np.ndarray | None, bool, list[str]]:
    if reference_measurement is None:
        return (
            None,
            False,
            [
                "Reference subtraction skipped because no suitable reference file was supplied."
            ],
        )

    reference_trace, notes = _load_matched_reference(
        reference_measurement,
        expected_frequencies_hz,
        target_header,
        spike_method=spike_method,
        spike_kwargs=spike_kwargs,
    )
    if reference_trace is None:
        return None, False, notes
    return signal - reference_trace, True, notes


def _normalized_copy(s21: np.ndarray) -> np.ndarray | None:
    peak = float(np.max(np.abs(s21))) if s21.size else 0.0
    if peak <= 0.0:
        return None
    return s21 / peak


def _nrmse(reference: np.ndarray, processed: np.ndarray) -> float:
    """
    Normalized RMSE for complex S21: ||processed - reference||_2 / ||reference||_2.
    Matches the Module 2 brief name; equals the former distortion_ratio definition.
    """
    baseline = np.linalg.norm(reference)
    if baseline == 0:
        return 0.0
    return float(np.linalg.norm(processed - reference) / baseline)


def process_touchstone_s21_dataset(
    dataset: MicrowaveDataset, config
) -> TouchstoneS21ProcessingResult:
    raw_frequencies_hz, raw_s21, header = load_touchstone_s21_trace(dataset.file_path)
    _validate_frequency_and_s21_lengths(raw_frequencies_hz, raw_s21)
    _validate_numeric_content(raw_frequencies_hz, raw_s21)
    raw_frequencies_hz, raw_s21 = _sort_and_validate_unique_frequencies(
        raw_frequencies_hz, raw_s21
    )

    uniform_frequencies_hz, uniform_raw_s21, interpolation_performed = (
        _interpolate_complex_trace(
            raw_frequencies_hz,
            raw_s21,
        )
    )

    raw_magnitude_db = 20.0 * np.log10(np.abs(uniform_raw_s21) + 1e-12)
    wrapped_phase_rad = np.arctan2(uniform_raw_s21.imag, uniform_raw_s21.real)
    wrapped_phase_deg = np.rad2deg(wrapped_phase_rad)
    unwrapped_phase_deg, unwrap_adjustments = _unwrap_phase_degrees(wrapped_phase_deg)
    unwrapped_phase_rad = np.deg2rad(unwrapped_phase_deg)

    corrected_s21, corrected_samples, corrected_indices = _correct_isolated_samples(
        uniform_raw_s21,
        method=getattr(config, "spike_detection_method", "hampel"),
        return_indices=True,
        **getattr(config, "spike_detection_kwargs", {}),
    )

    averaged_s21, averaging_performed, averaging_notes = _complex_average(
        corrected_s21,
        getattr(config, "repeated_measurements", None),
        uniform_frequencies_hz,
        spike_method=getattr(config, "spike_detection_method", "hampel"),
        spike_kwargs=getattr(config, "spike_detection_kwargs", {}),
        max_sweep_nrmse=float(getattr(config, "max_repeated_sweep_nrmse", 0.3)),
    )
    working_signal = averaged_s21 if averaged_s21 is not None else corrected_s21

    enable_week3 = bool(getattr(config, "enable_week3", True))
    reference_dataset = getattr(config, "reference_dataset", None)
    reference_subtracted_s21 = None
    reference_subtraction_performed = False
    reference_notes: list[str] = []
    reference_filtered_for_week3: np.ndarray | None = None

    if enable_week3:
        # Week 3: filter target (and reference) first; matched Hamming + subtract later.
        if reference_dataset is None:
            reference_notes.append(
                "Reference subtraction deferred to Week 3; no reference file supplied."
            )
        else:
            reference_trace, matched_notes = _load_matched_reference(
                reference_dataset,
                uniform_frequencies_hz,
                header,
                spike_method=getattr(config, "spike_detection_method", "hampel"),
                spike_kwargs=getattr(config, "spike_detection_kwargs", {}),
            )
            reference_notes.extend(matched_notes)
            if reference_trace is not None:
                # Same filter as the target; matched Hamming happens in Week 3.
                reference_filtered_for_week3 = apply_noise_filter(
                    reference_trace.reshape(-1, 1),
                    method=config.filter_method,
                    **config.filter_kwargs,
                )[:, 0]
    else:
        reference_subtracted_s21, reference_subtraction_performed, reference_notes = (
            _complex_reference_subtraction(
                working_signal,
                reference_dataset,
                uniform_frequencies_hz,
                spike_method=getattr(config, "spike_detection_method", "hampel"),
                spike_kwargs=getattr(config, "spike_detection_kwargs", {}),
                target_header=header,
            )
        )
        working_signal = (
            reference_subtracted_s21
            if reference_subtracted_s21 is not None
            else working_signal
        )

    filtered_2d = apply_noise_filter(
        working_signal.reshape(-1, 1),
        method=config.filter_method,
        **config.filter_kwargs,
    )
    filtered_s21 = filtered_2d[:, 0]
    filter_comparison = compare_noise_filters(working_signal)
    windowed_s21 = filtered_s21 * hamming_window(filtered_s21.shape[0])
    normalized_s21 = (
        _normalized_copy(filtered_s21)
        if getattr(config, "generate_normalized_copy", True)
        else None
    )

    week3_result: Week3TimeDomainResult | None = None
    if enable_week3:
        week3_result = process_week3_time_domain(
            uniform_frequencies_hz,
            filtered_s21,
            reference_filtered=reference_filtered_for_week3,
            enable_clutter_removal=bool(
                getattr(config, "enable_group_clutter_removal", True)
            ),
            enable_global_normalize=bool(
                getattr(config, "enable_global_normalize", True)
            ),
        )
        windowed_s21 = week3_result.s21_hamming[:, 0]
        if week3_result.s21_reference_subtracted_freq is not None:
            reference_subtracted_s21 = week3_result.s21_reference_subtracted_freq[:, 0]
            reference_subtraction_performed = True
        reference_notes.extend(week3_result.notes)

    _validate_frequency_and_s21_lengths(uniform_frequencies_hz, filtered_s21)
    _validate_numeric_content(uniform_frequencies_hz, filtered_s21)

    delta_f_hz = np.diff(uniform_frequencies_hz)
    spike_method = str(getattr(config, "spike_detection_method", "hampel"))
    # NRMSE isolates the filtering change: filter output vs the exact filter input (S21_bar).
    nrmse = _nrmse(working_signal, filtered_s21)
    notes = []
    notes.extend(averaging_notes)
    notes.extend(reference_notes)
    if not _is_uniform_spacing(delta_f_hz):
        raise InvalidFrequencyError(
            "Processed frequency vector is not uniformly spaced."
        )
    if nrmse > 0.75:
        notes.append(
            f"NRMSE={nrmse:.3f} exceeds 0.75; processed curve may be over-smoothed — reduce filter strength."
        )
    corrected_frequencies_hz = uniform_frequencies_hz[corrected_indices]
    if corrected_samples == 0:
        notes.append("No isolated invalid points required correction.")
    else:
        listed = ", ".join(f"{f / 1e9:.4f}" for f in corrected_frequencies_hz[:10])
        more = f" (+{corrected_samples - 10} more)" if corrected_samples > 10 else ""
        notes.append(f"Corrected isolated samples at GHz: {listed}{more}.")

    report = TouchstoneS21ValidationReport(
        filter_used=config.filter_method,
        filter_parameters=dict(getattr(config, "filter_kwargs", {}) or {}),
        corrected_samples=corrected_samples,
        corrected_sample_indices=corrected_indices.tolist(),
        corrected_sample_frequencies_hz=corrected_frequencies_hz.tolist(),
        averaging_performed=averaging_performed,
        reference_subtraction_performed=reference_subtraction_performed,
        interpolation_performed=interpolation_performed,
        normalized_copy_generated=normalized_s21 is not None,
        phase_unwrap_adjustments=unwrap_adjustments,
        nrmse=nrmse,
        spike_detection_method=spike_method,
        week3_time_domain_performed=week3_result is not None,
        notes=notes,
    )

    return TouchstoneS21ProcessingResult(
        header=header,
        raw_frequencies_hz=raw_frequencies_hz,
        raw_s21=raw_s21,
        uniform_frequencies_hz=uniform_frequencies_hz,
        uniform_s21=uniform_raw_s21,
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
        week3_result=week3_result,
        filter_input_s21=working_signal,
        filter_comparison=filter_comparison,
    )

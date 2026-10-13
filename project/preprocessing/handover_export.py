"""
Module 2 → Module 3 handover export (.mat / optional .csv).

Writes frequency/time axes, S21 variants, Tx/Rx coordinates, and processing
parameters per Signal Preprocessing_21082026.pdf step 16.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.io import savemat

from data_loader.dataset_info import MicrowaveDataset
from preprocessing.preprocessing_pipeline import PreprocessingResult
from preprocessing.week3_time_domain import (
    Week3TimeDomainResult,
    circular_tx_rx_coordinates,
)
from utils.exceptions import DatasetValidationError
from utils.logger import get_logger

logger = get_logger(__name__)

_FREQUENCY_ARRAYS = (
    "S21_raw",
    "S21_filtered",
    "S21_Hamming",
    "S21_reference_Hamming",
    "S21_reference_subtracted",
)
_TIME_ARRAYS = (
    "S21_clutter_removed",
    "time_filtered",
    "time_Hamming",
    "time_reference_subtracted",
    "time_clutter_removed",
    "time_global_normalized",
)


def _code_version() -> str:
    """Git commit of the project (with '+dirty' for uncommitted changes), else 'unknown'."""
    project_dir = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_dir, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_dir, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{commit}+dirty" if dirty else commit
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _as_2d_complex(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _coords_from_dataset(
    dataset: MicrowaveDataset, n_channels: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    notes: list[str] = []
    meta = dataset.metadata or {}

    if "tx_coordinates" in meta and "rx_coordinates" in meta:
        tx = np.asarray(meta["tx_coordinates"], dtype=float)
        rx = np.asarray(meta["rx_coordinates"], dtype=float)
        return tx, rx, notes

    radius = float(meta.get("antenna_radius_m", 0.08) or 0.08)
    offset = float(meta.get("antenna_angle_offset_deg", 0.0) or 0.0)
    tx, rx = circular_tx_rx_coordinates(n_channels, radius, angle_offset_deg=offset)
    notes.append(
        f"Tx/Rx coordinates synthesized on a circle (radius={radius:.4g} m, n={n_channels})."
    )
    return tx, rx, notes


def build_handover_payload(result: PreprocessingResult) -> dict[str, Any]:
    """Assemble the Module 3 handover dictionary (NumPy-friendly)."""
    processed = result.processed_dataset
    original = result.original_dataset
    touchstone = result.touchstone_s21_result
    week3: Week3TimeDomainResult | None = getattr(result, "week3_result", None)

    frequencies = np.asarray(processed.frequencies, dtype=float)
    s21_raw = _as_2d_complex(original.s_parameters)
    s21_filtered = _as_2d_complex(processed.s_parameters)

    s21_hamming = None
    s21_reference_hamming = None
    s21_ref_sub = None
    s21_clutter = None
    time_s = None
    time_filtered = None
    time_hamming = None
    time_ref_sub = None
    time_clutter = None
    time_norm = None
    alpha = None

    if touchstone is not None:
        s21_raw = touchstone.uniform_s21.reshape(-1, 1)
        s21_filtered = touchstone.filtered_s21.reshape(-1, 1)
        s21_hamming = touchstone.windowed_s21.reshape(-1, 1)
        if touchstone.reference_subtracted_s21 is not None:
            s21_ref_sub = touchstone.reference_subtracted_s21.reshape(-1, 1)
        frequencies = touchstone.uniform_frequencies_hz

    if week3 is not None:
        frequencies = week3.frequencies_hz
        s21_filtered = _as_2d_complex(week3.s21_filtered)
        s21_hamming = week3.s21_hamming
        s21_reference_hamming = week3.s21_reference_hamming
        if week3.s21_reference_subtracted_freq is not None:
            s21_ref_sub = week3.s21_reference_subtracted_freq
        time_s = week3.time_s
        time_filtered = week3.time_filtered
        time_hamming = week3.time_hamming
        time_ref_sub = week3.time_reference_subtracted
        time_clutter = week3.time_clutter_removed
        time_norm = week3.time_global_normalized
        alpha = week3.global_scale_alpha
        if week3.time_clutter_removed is not None and week3.clutter_removal_performed:
            # Frequency-domain clutter copy is not required; expose time-domain clean
            s21_clutter = week3.time_clutter_removed

    n_channels = int(s21_filtered.shape[1]) if s21_filtered is not None else 1
    tx, rx, coord_notes = _coords_from_dataset(processed, n_channels)
    labels = [f"ch{i}" for i in range(n_channels)]

    config = result.config
    reference = getattr(config, "reference_dataset", None)
    repeated = getattr(config, "repeated_measurements", None) or []
    processing_parameters: dict[str, Any] = {
        "source_file": getattr(original, "file_path", None),
        "reference_file": getattr(reference, "file_path", None),
        "repeated_files": [getattr(m, "file_path", "array") for m in repeated],
        "filter_method": config.filter_method,
        "filter_kwargs": dict(config.filter_kwargs),
        "spike_detection_method": config.spike_detection_method,
        "spike_detection_kwargs": dict(getattr(config, "spike_detection_kwargs", {}) or {}),
        "max_repeated_sweep_nrmse": getattr(config, "max_repeated_sweep_nrmse", None),
        "normalization_method": getattr(config, "normalization_method", None),
        "hamming_window": "w[k] = 0.54 - 0.46 cos(2*pi*k/(N-1))",
        "enable_week3": bool(getattr(config, "enable_week3", True)),
        "enable_group_clutter_removal": bool(
            getattr(config, "enable_group_clutter_removal", True)
        ),
        "clutter_channel_groups": getattr(config, "clutter_channel_groups", None),
        "enable_global_normalize": bool(
            getattr(config, "enable_global_normalize", True)
        ),
        "coordinate_notes": coord_notes,
    }
    if result.validation_report is not None:
        processing_parameters["validation"] = result.validation_report.to_display_dict()
    if week3 is not None:
        processing_parameters["week3"] = week3.to_display_dict()
        processing_parameters["week3_notes"] = list(week3.notes)

    payload: dict[str, Any] = {
        "frequency_Hz": frequencies,
        "time_s": time_s if time_s is not None else np.asarray([], dtype=float),
        "S21_raw": s21_raw,
        "S21_filtered": s21_filtered,
        "S21_Hamming": s21_hamming
        if s21_hamming is not None
        else np.asarray([], dtype=complex),
        "S21_reference_Hamming": s21_reference_hamming
        if s21_reference_hamming is not None
        else np.asarray([], dtype=complex),
        "S21_reference_subtracted": s21_ref_sub
        if s21_ref_sub is not None
        else np.asarray([], dtype=complex),
        "S21_clutter_removed": s21_clutter
        if s21_clutter is not None
        else np.asarray([], dtype=complex),
        "time_filtered": time_filtered
        if time_filtered is not None
        else np.asarray([], dtype=complex),
        "time_Hamming": time_hamming
        if time_hamming is not None
        else np.asarray([], dtype=complex),
        "time_reference_subtracted": time_ref_sub
        if time_ref_sub is not None
        else np.asarray([], dtype=complex),
        "time_clutter_removed": time_clutter
        if time_clutter is not None
        else np.asarray([], dtype=complex),
        "time_global_normalized": time_norm
        if time_norm is not None
        else np.asarray([], dtype=complex),
        "global_scale_alpha": float(alpha) if alpha is not None else np.nan,
        "Tx_coordinates": tx,
        "Rx_coordinates": rx,
        "measurement_labels": np.asarray(labels, dtype=object),
        "code_version": _code_version(),
        "processing_parameters_json": json.dumps(processing_parameters, default=str),
    }
    return payload


def validate_handover_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Step 16 checks: equal dimensions, uniform spacing, matching metadata,
    complex values and absence of NaN/Inf. Returns one entry per check:
    {"check": str, "passed": bool, "detail": str}.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    freqs = np.asarray(payload.get("frequency_Hz"), dtype=float).reshape(-1)
    n_freq = freqs.size
    freq_ok = n_freq >= 2 and bool(np.all(np.isfinite(freqs)))
    add("frequency axis finite, ≥2 samples", freq_ok, f"N={n_freq}")
    delta_f = float(np.mean(np.diff(freqs))) if freq_ok else float("nan")
    if freq_ok:
        steps = np.diff(freqs)
        add("frequency strictly increasing", bool(np.all(steps > 0)))
        max_dev = float(np.max(np.abs(steps - delta_f)) / delta_f) if delta_f > 0 else float("inf")
        add("frequency uniformly spaced", max_dev <= 1e-6, f"Δf={delta_f:.6g} Hz, max rel. deviation={max_dev:.2e}")

    filtered = np.asarray(payload.get("S21_filtered"))
    n_channels = int(filtered.shape[1]) if filtered.ndim == 2 else 1

    time_s = np.asarray(payload.get("time_s"), dtype=float).reshape(-1)
    n_time = time_s.size
    if n_time:
        add("time axis finite", bool(np.all(np.isfinite(time_s))))
        if n_time >= 2 and freq_ok:
            delta_t = float(time_s[1] - time_s[0])
            expected = 1.0 / (n_time * delta_f)
            uniform = bool(np.allclose(np.diff(time_s), delta_t, rtol=1e-9, atol=0.0))
            add("time axis uniform with Δt = 1/(NΔf)",
                uniform and np.isclose(delta_t, expected, rtol=1e-6),
                f"Δt={delta_t:.6g} s, expected {expected:.6g} s")

    for key, n_rows in [(k, n_freq) for k in _FREQUENCY_ARRAYS] + [(k, n_time) for k in _TIME_ARRAYS]:
        array = np.asarray(payload.get(key, np.asarray([])))
        if array.size == 0:
            continue
        shape_ok = array.ndim == 2 and array.shape == (n_rows, n_channels)
        add(f"{key} dimensions", shape_ok, f"shape={array.shape}, expected=({n_rows}, {n_channels})")
        add(f"{key} complex", np.iscomplexobj(array), str(array.dtype))
        add(f"{key} no NaN/Inf", bool(np.all(np.isfinite(array))))

    tx = np.asarray(payload.get("Tx_coordinates"), dtype=float)
    rx = np.asarray(payload.get("Rx_coordinates"), dtype=float)
    coords_ok = (
        tx.ndim == 2 and tx.shape == rx.shape and tx.shape[0] == n_channels and tx.shape[1] in (2, 3)
    )
    add("Tx/Rx coordinates match channel count", coords_ok,
        f"Tx {tx.shape}, Rx {rx.shape}, channels={n_channels}")
    add("Tx/Rx coordinates no NaN/Inf", bool(np.all(np.isfinite(tx)) and np.all(np.isfinite(rx))))

    labels = payload.get("measurement_labels")
    n_labels = len(labels) if labels is not None else 0
    add("measurement labels match channel count", n_labels == n_channels, f"{n_labels} labels")

    normalized = np.asarray(payload.get("time_global_normalized", np.asarray([])))
    if normalized.size:
        alpha = float(payload.get("global_scale_alpha", np.nan))
        add("global α finite and positive", np.isfinite(alpha) and alpha > 0, f"α={alpha:.6g}")

    try:
        json.loads(payload.get("processing_parameters_json", ""))
        add("processing parameters serialised", True)
    except (TypeError, ValueError) as exc:
        add("processing parameters serialised", False, str(exc))
    return checks


def _write_validation_plots(result: PreprocessingResult, out_dir: Path, stem: str) -> list[str]:
    from utils import session_report

    ctx = SimpleNamespace(preprocessing=result)
    writers = (
        session_report._save_magnitude_plot,
        session_report._save_phase_plot,
        session_report._save_filter_comparison_plot,
        session_report._save_time_domain_plot,
    )
    written: list[str] = []
    for writer in writers:
        try:
            path = writer(ctx, str(out_dir), stem)
        except Exception as exc:  # plots are supplementary; never block the export
            logger.warning("Handover plot %s skipped: %s", writer.__name__, exc)
            continue
        if path:
            written.append(path)
    return written


def export_module3_handover(
    result: PreprocessingResult,
    mat_path: str | Path,
    *,
    csv_path: str | Path | None = None,
    include_plots: bool = True,
    strict: bool = True,
) -> Path:
    """
    Validate, then write the Module 3 handover: `.mat`, optional magnitude CSV of
    filtered S21, `.json` report (parameters, validation checks, code version,
    plot list) and validation plots next to the `.mat`.

    With ``strict`` a failed check raises DatasetValidationError and nothing is
    written. Returns the path of the written `.mat` file.
    """
    mat_path = Path(mat_path)
    payload = build_handover_payload(result)
    checks = validate_handover_payload(payload)
    failed = [c for c in checks if not c["passed"]]
    if failed and strict:
        summary = "; ".join(f"{c['check']} ({c['detail']})" if c["detail"] else c["check"] for c in failed)
        raise DatasetValidationError(f"Module 3 handover validation failed: {summary}")

    mat_path.parent.mkdir(parents=True, exist_ok=True)
    payload["validation_passed"] = not failed
    payload["validation_checks_json"] = json.dumps(checks)

    # SciPy savemat needs plain arrays; strip None already handled.
    savemat(str(mat_path), payload, do_compression=True)

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        freqs = np.asarray(payload["frequency_Hz"], dtype=float)
        s21 = np.asarray(payload["S21_filtered"])
        mag_db = 20.0 * np.log10(np.abs(s21) + 1e-12)
        header = "frequency_Hz," + ",".join(
            f"|S21|_dB_ch{i}" for i in range(mag_db.shape[1])
        )
        rows = np.column_stack([freqs, mag_db])
        np.savetxt(str(csv_path), rows, delimiter=",", header=header, comments="")

    plots = _write_validation_plots(result, mat_path.parent, mat_path.stem) if include_plots else []

    report = {
        "code_version": payload["code_version"],
        "processing_parameters": json.loads(payload["processing_parameters_json"]),
        "validation_passed": not failed,
        "validation_checks": checks,
        "files": {
            "mat": str(mat_path),
            "csv": str(csv_path) if csv_path is not None else None,
            "plots": plots,
        },
    }
    report_path = mat_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return mat_path

"""
Module 2 → Module 3 handover export (.mat / optional .csv).

Writes frequency/time axes, S21 variants, Tx/Rx coordinates, and processing
parameters per Signal Preprocessing_21082026.pdf step 16.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from data_loader.dataset_info import MicrowaveDataset
from preprocessing.preprocessing_pipeline import PreprocessingResult
from preprocessing.week3_time_domain import (
    Week3TimeDomainResult,
    circular_tx_rx_coordinates,
)


def _as_2d_complex(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
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
    s21_ref_sub = None
    s21_clutter = None
    time_s = None
    time_hamming = None
    time_ref_sub = None
    time_clutter = None
    time_norm = None
    alpha = None

    if touchstone is not None:
        s21_raw = touchstone.raw_s21.reshape(-1, 1)
        s21_filtered = touchstone.filtered_s21.reshape(-1, 1)
        s21_hamming = touchstone.windowed_s21.reshape(-1, 1)
        if touchstone.reference_subtracted_s21 is not None:
            s21_ref_sub = touchstone.reference_subtracted_s21.reshape(-1, 1)
        frequencies = touchstone.uniform_frequencies_hz

    if week3 is not None:
        frequencies = week3.frequencies_hz
        s21_filtered = week3.s21_filtered
        s21_hamming = week3.s21_hamming
        if week3.s21_reference_subtracted_freq is not None:
            s21_ref_sub = week3.s21_reference_subtracted_freq
        time_s = week3.time_s
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

    processing_parameters: dict[str, Any] = {
        "filter_method": result.config.filter_method,
        "filter_kwargs": dict(result.config.filter_kwargs),
        "spike_detection_method": result.config.spike_detection_method,
        "enable_week3": bool(getattr(result.config, "enable_week3", True)),
        "enable_group_clutter_removal": bool(
            getattr(result.config, "enable_group_clutter_removal", True)
        ),
        "enable_global_normalize": bool(
            getattr(result.config, "enable_global_normalize", True)
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
        "S21_reference_subtracted": s21_ref_sub
        if s21_ref_sub is not None
        else np.asarray([], dtype=complex),
        "S21_clutter_removed": s21_clutter
        if s21_clutter is not None
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
        "processing_parameters_json": json.dumps(processing_parameters, default=str),
    }
    return payload


def export_module3_handover(
    result: PreprocessingResult,
    mat_path: str | Path,
    *,
    csv_path: str | Path | None = None,
) -> Path:
    """
    Write Module 3 handover `.mat` (and optional magnitude CSV of filtered S21).

    Returns the path of the written `.mat` file.
    """
    mat_path = Path(mat_path)
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_handover_payload(result)

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

    report_path = mat_path.with_suffix(".json")
    report_path.write_text(str(payload["processing_parameters_json"]), encoding="utf-8")
    return mat_path

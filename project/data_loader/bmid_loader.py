"""
data_loader/bmid_loader.py

Load a single scan from University of Manitoba BMID (UM-BMID) frequency-domain
cubes such as fd_data_s21_adi.mat, paired with md_list_* metadata when present.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat

from data_loader.dataset_info import MicrowaveDataset
from data_loader.validator import validate_dataset
from utils.exceptions import CorruptedFileError, MissingVariableError
from utils.logger import get_logger

logger = get_logger(__name__)

# Official UM-BMID clean examples use 1-8 GHz with 1001 samples (see data_use_ex.py).
BMID_FREQ_START_HZ = 1e9
BMID_FREQ_STOP_HZ = 8e9
BMID_N_FREQ = 1001

_FD_NAME_RE = re.compile(r"^fd_data_(s11|s21)(?:_(adi|emp))?$", re.IGNORECASE)
_FD_FILE_RE = re.compile(r"fd_data_(s11|s21)(?:_(adi|emp))?\.mat$", re.IGNORECASE)


@dataclass(frozen=True)
class BmidScanInfo:
    index: int
    scan_id: int | None
    phant_id: str
    has_tumor: bool
    tum_diam_cm: float | None
    tum_x_cm: float | None
    tum_y_cm: float | None
    birads: int | None
    ant_rad_cm: float | None
    label: str


class ScanSelectionRequiredError(MissingVariableError):
    """Raised when a multi-scan BMID cube is loaded without choosing a scan index."""

    def __init__(self, message: str, n_scans: int):
        super().__init__(message)
        self.n_scans = n_scans


def is_bmid_fd_filename(file_path: str) -> bool:
    return bool(_FD_FILE_RE.search(os.path.basename(file_path)))


def bmid_frequencies_hz() -> np.ndarray:
    return np.linspace(BMID_FREQ_START_HZ, BMID_FREQ_STOP_HZ, BMID_N_FREQ)


def companion_metadata_path(fd_file_path: str) -> str | None:
    """Map fd_data_s21_adi.mat -> md_list_s21_adi.mat in the same folder."""
    base = os.path.basename(fd_file_path)
    match = _FD_FILE_RE.search(base)
    if not match:
        return None
    sparam = match.group(1).lower()
    cal = match.group(2)
    name = f"md_list_{sparam}_{cal}.mat" if cal else f"md_list_{sparam}.mat"
    candidate = os.path.join(os.path.dirname(fd_file_path), name)
    return candidate if os.path.isfile(candidate) else None


def _safe_float(value) -> float | None:
    try:
        number = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _safe_int(value) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _safe_str(value) -> str:
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.dtype.kind in "UO" and arr.size:
        value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if text in {"", "None", "nan", "(0,)", "[]"}:
        return ""
    return text


def _load_metadata_structs(md_path: str) -> list:
    try:
        raw = loadmat(md_path, squeeze_me=True, struct_as_record=False)
    except (OSError, ValueError, TypeError) as exc:
        raise CorruptedFileError(
            f"Could not parse BMID metadata file '{os.path.basename(md_path)}': {exc}"
        ) from exc

    keys = [key for key in raw if not key.startswith("__")]
    if not keys:
        raise CorruptedFileError(f"No variables found in '{os.path.basename(md_path)}'.")

    preferred = [key for key in keys if key.lower().startswith("md_")]
    arr = raw[preferred[0] if preferred else keys[0]]
    items = list(np.ravel(arr))
    if not items:
        raise CorruptedFileError(f"Metadata file '{os.path.basename(md_path)}' is empty.")
    return items


def list_bmid_scans(fd_file_path: str) -> list[BmidScanInfo]:
    """Build a scan list from companion metadata (preferred) or cube length."""
    md_path = companion_metadata_path(fd_file_path)
    if md_path is None:
        n_scans = peek_bmid_n_scans(fd_file_path)
        return [
            BmidScanInfo(
                index=i,
                scan_id=i + 1,
                phant_id="unknown",
                has_tumor=False,
                tum_diam_cm=None,
                tum_x_cm=None,
                tum_y_cm=None,
                birads=None,
                ant_rad_cm=None,
                label=f"Scan {i} (no metadata file found)",
            )
            for i in range(n_scans)
        ]

    structs = _load_metadata_structs(md_path)
    scans: list[BmidScanInfo] = []
    for index, item in enumerate(structs):
        tum_diam = _safe_float(getattr(item, "tum_diam", np.nan))
        has_tumor = tum_diam is not None and tum_diam > 0
        phant_id = _safe_str(getattr(item, "phant_id", "")) or "unknown"
        tum_x = _safe_float(getattr(item, "tum_x", np.nan))
        tum_y = _safe_float(getattr(item, "tum_y", np.nan))
        birads = _safe_int(getattr(item, "birads", np.nan))
        ant_rad = _safe_float(getattr(item, "ant_rad", np.nan))
        scan_id = _safe_int(getattr(item, "id", index + 1))
        if has_tumor:
            label = (
                f"[{index}] id={scan_id} {phant_id} TUMOR "
                f"d={tum_diam:.1f}cm @ ({tum_x:.2f},{tum_y:.2f}) cm"
            )
        else:
            label = f"[{index}] id={scan_id} {phant_id} HEALTHY"
        scans.append(
            BmidScanInfo(
                index=index,
                scan_id=scan_id,
                phant_id=phant_id,
                has_tumor=has_tumor,
                tum_diam_cm=tum_diam if has_tumor else None,
                tum_x_cm=tum_x if has_tumor else None,
                tum_y_cm=tum_y if has_tumor else None,
                birads=birads,
                ant_rad_cm=ant_rad,
                label=label,
            )
        )
    return scans


def peek_bmid_n_scans(fd_file_path: str) -> int:
    variable, cube = _load_fd_cube(fd_file_path)
    del variable
    return int(cube.shape[0])


def _load_fd_cube(fd_file_path: str) -> tuple[str, np.ndarray]:
    try:
        raw = loadmat(fd_file_path, squeeze_me=True, struct_as_record=False)
    except (OSError, ValueError, TypeError) as exc:
        raise CorruptedFileError(
            f"Could not parse BMID data file '{os.path.basename(fd_file_path)}': {exc}"
        ) from exc

    candidates = []
    for name, value in raw.items():
        if name.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if not np.iscomplexobj(value) and value.dtype.kind not in "fc":
            continue
        if value.ndim != 3:
            continue
        if BMID_N_FREQ not in value.shape:
            continue
        name_score = 2 if _FD_NAME_RE.match(name) else (1 if "fd_data" in name.lower() else 0)
        candidates.append((name_score, value.size, name, value))

    if not candidates:
        raise MissingVariableError(
            f"No BMID frequency-domain cube found in '{os.path.basename(fd_file_path)}'."
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, name, cube = candidates[0]
    cube = np.asarray(cube)
    if cube.shape[1] == BMID_N_FREQ:
        pass
    elif cube.shape[0] == BMID_N_FREQ:
        cube = np.moveaxis(cube, 0, 1)
    elif cube.shape[2] == BMID_N_FREQ:
        cube = np.moveaxis(cube, 2, 1)
    else:
        raise MissingVariableError(
            f"BMID cube '{name}' has shape {cube.shape}; expected frequency axis of length {BMID_N_FREQ}."
        )
    return name, cube


def _metadata_dict_for_scan(fd_file_path: str, scan_index: int, scan_info: BmidScanInfo | None) -> dict:
    ant_rad_cm = scan_info.ant_rad_cm if scan_info and scan_info.ant_rad_cm else 18.0
    antenna_radius_m = float(ant_rad_cm) / 100.0
    # Imaging FOV covers phantom interior (tumors ~±2.25 cm); keep ±6 cm default.
    half_span = 0.06
    metadata = {
        "dataset_family": "UM-BMID",
        "bmid_scan_index": int(scan_index),
        "frequency_variable": "synthetic_bmid_frequency_hz",
        "sparameter_variable": "fd_data",
        "antenna_radius_m": antenna_radius_m,
        "wave_speed_m_per_s": 3e8,
        "reconstruction_x_span_m": (-half_span, half_span),
        "reconstruction_y_span_m": (-half_span, half_span),
        "bmid_calibration": None,
    }

    match = _FD_FILE_RE.search(os.path.basename(fd_file_path))
    if match:
        metadata["bmid_sparam"] = match.group(1).lower()
        metadata["bmid_calibration"] = (match.group(2) or "").lower() or None

    if scan_info is not None:
        metadata.update(
            {
                "bmid_scan_id": scan_info.scan_id,
                "bmid_phant_id": scan_info.phant_id,
                "bmid_has_tumor": scan_info.has_tumor,
                "bmid_birads": scan_info.birads,
                "tumor_diameter_m": (
                    scan_info.tum_diam_cm / 100.0 if scan_info.tum_diam_cm is not None else None
                ),
                "tumor_x_m": scan_info.tum_x_cm / 100.0 if scan_info.tum_x_cm is not None else None,
                "tumor_y_m": scan_info.tum_y_cm / 100.0 if scan_info.tum_y_cm is not None else None,
            }
        )
    return metadata


def load_bmid_scan(fd_file_path: str, scan_index: int) -> MicrowaveDataset:
    """Load one BMID scan as a (n_freq, n_antennas) MicrowaveDataset."""
    scans = list_bmid_scans(fd_file_path)
    if scan_index < 0 or scan_index >= len(scans):
        raise MissingVariableError(
            f"BMID scan index {scan_index} is out of range for {len(scans)} scans."
        )

    variable_name, cube = _load_fd_cube(fd_file_path)
    if scan_index >= cube.shape[0]:
        raise MissingVariableError(
            f"BMID scan index {scan_index} exceeds cube length {cube.shape[0]}."
        )

    scan_info = scans[scan_index]
    s_parameters = np.asarray(cube[scan_index], dtype=complex)  # (n_freq, n_ant)
    if s_parameters.ndim != 2 or s_parameters.shape[0] != BMID_N_FREQ:
        raise MissingVariableError(
            f"Unexpected BMID scan slice shape {s_parameters.shape}; expected ({BMID_N_FREQ}, n_ant)."
        )

    frequencies = bmid_frequencies_hz()
    metadata = _metadata_dict_for_scan(fd_file_path, scan_index, scan_info)
    metadata["sparameter_variable"] = variable_name
    metadata["bmid_n_scans"] = int(cube.shape[0])
    metadata["bmid_n_antennas"] = int(s_parameters.shape[1])

    dataset = MicrowaveDataset(
        file_path=fd_file_path,
        file_name=os.path.basename(fd_file_path),
        file_type="MATLAB (UM-BMID)",
        frequencies=frequencies,
        s_parameters=s_parameters,
        n_ports=int(s_parameters.shape[1]),
        available_variables=[variable_name, "bmid_metadata"],
        metadata=metadata,
        raw={"bmid_scan_info": scan_info},
    )
    validate_dataset(dataset)
    logger.info(
        "Loaded BMID scan %s/%s from %s -> S shape %s, ant_rad=%.2f m, tumor=%s",
        scan_index,
        cube.shape[0] - 1,
        os.path.basename(fd_file_path),
        s_parameters.shape,
        metadata["antenna_radius_m"],
        scan_info.has_tumor,
    )
    return dataset

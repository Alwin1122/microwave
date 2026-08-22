"""
data_loader/matlab_loader.py

Purpose:
    Load microwave measurement datasets stored as MATLAB .mat files,
    supporting both legacy MAT-file formats (v4/v5/v6/v7, read via
    scipy.io.loadmat) and the HDF5-based MAT v7.3 format (read via h5py).

Input:
    file_path (str): path to a .mat file.

Output:
    A MicrowaveDataset instance (see data_loader/dataset_info.py) with
    frequencies, S-parameters, port count, and available variable names
    populated.

Description:
    MATLAB datasets do not follow one fixed schema, so this loader uses
    heuristics to locate the frequency vector and S-parameter matrix among
    the file's variables:
      - Frequency vector: a real-valued 1D array whose variable name
        matches common conventions (freq, f, frequency, Frequencies...)
        OR, failing that, any real 1D array with monotonically increasing
        values in a plausible RF range.
      - S-parameters: the largest complex-valued array in the file, OR a
        real array explicitly named to indicate S-parameters (s_params,
        S11, etc.), in which case it is treated as already-magnitude data
        and kept real.
    This keeps the loader usable across the variety of ad-hoc MATLAB
    export schemes typically produced by microwave imaging test benches.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatReadError

from data_loader.dataset_info import MicrowaveDataset
from data_loader.physical_metadata import extract_physical_metadata_from_values
from data_loader.validator import validate_dataset, validate_file_path
from utils.exceptions import CorruptedFileError, MissingVariableError
from utils.logger import get_logger

logger = get_logger(__name__)

_FREQ_NAME_HINTS = ("freq", "frequency", "frequencies", "f_ghz", "f_hz", "fvec")
_SPARAM_NAME_HINTS = ("s_param", "sparam", "s11", "s21", "s_matrix", "smat", "s_data", "signal", "data")

# Some published datasets store raw S-parameter arrays WITHOUT any
# frequency vector in the file at all, because the frequency sweep is a
# fixed, documented property of the measurement system rather than
# per-scan data. Rather than fail (or silently guess wrong), known
# variable names are mapped to their documented frequency sweep here.
# Keyed by lowercased variable name; value is (start_hz, stop_hz, n_points).
#
# 'fd_data_s11' -> University of Manitoba Breast Microwave Imaging
#   Dataset (UM-BMID), generation 3: measurements taken at 1001
#   frequencies swept over 1-9 GHz using a Copper Mountain C1209 VNA.
#   Source: Reimer & Fear et al., "An Optimization-Based Approach to
#   Radar Image Reconstruction in Breast Microwave Sensing", PMC8704509.
_KNOWN_DATASET_FREQUENCY_SWEEPS = {
    "fd_data_s11": (1e9, 9e9, 1001),
}


def _is_legacy_mat(file_path: str) -> bool:
    """
    Detect MAT v7.3 (HDF5) vs legacy MAT format.

    MATLAB v7.3 files are valid HDF5 files, but MATLAB writes a plaintext
    compatibility header ("MATLAB 7.3 MAT-file...") at the start of the
    file, so the real HDF5 signature (\\x89HDF\\r\\n\\x1a\\n) is NOT at
    byte 0 - it sits at a later offset (typically 512). A naive check of
    only the first few bytes therefore misclassifies every v7.3 file as
    "legacy", causing scipy.io.loadmat to fail with its own internal
    error ("Please use HDF5 reader for matlab v7.3 files, e.g. h5py").

    h5py.is_hdf5() correctly scans for the signature wherever it actually
    appears, so it is used here instead.
    """
    return not h5py.is_hdf5(file_path)


def _load_legacy_mat(file_path: str) -> dict:
    try:
        raw = loadmat(file_path, squeeze_me=False, struct_as_record=False)
    except (MatReadError, ValueError, OSError) as exc:
        raise CorruptedFileError(
            f"Could not parse MATLAB file '{os.path.basename(file_path)}': {exc}"
        ) from exc
    # Drop MATLAB's internal metadata keys
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def _maybe_convert_matlab_complex(arr: np.ndarray) -> np.ndarray:
    """
    h5py reads MATLAB v7.3 complex arrays as a compound/structured dtype
    with 'real' and 'imag' fields (e.g. [('real', '<f8'), ('imag', '<f8')])
    rather than a native NumPy complex dtype. Detect that pattern and
    convert it to a proper complex128 array so the rest of the loader
    (which expects real numeric or native-complex arrays) can use it.
    """
    if arr.dtype.names and set(arr.dtype.names) >= {"real", "imag"}:
        return arr["real"].astype(float) + 1j * arr["imag"].astype(float)
    return arr


def _load_v73_mat(file_path: str) -> dict:
    variables: dict = {}
    try:
        with h5py.File(file_path, "r") as f:
            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        arr = np.array(obj)
                        variables[name] = _maybe_convert_matlab_complex(arr)
                    except Exception:  # pragma: no cover - defensive
                        pass

            f.visititems(_visit)
    except OSError as exc:
        raise CorruptedFileError(
            f"Could not parse MATLAB v7.3 file '{os.path.basename(file_path)}': {exc}"
        ) from exc
    return variables


def _is_plain_numeric_array(arr: np.ndarray) -> bool:
    """
    True only for arrays holding actual numbers (int/float/complex/bool).
    False for object-dtype arrays, which is what scipy.io.loadmat produces
    for MATLAB struct arrays, cell arrays, and strings — attempting
    .astype(float) on those raises ValueError/TypeError instead of a
    clean, catchable condition, so callers must check this first.
    """
    return isinstance(arr, np.ndarray) and arr.dtype.kind in ("f", "i", "u", "c", "b")


def _find_frequency_vector(variables: dict) -> tuple[str, np.ndarray] | None:
    # 1) name-based match
    for name, value in variables.items():
        if not isinstance(value, np.ndarray):
            continue
        arr = np.squeeze(value)
        if arr.ndim != 1 or np.iscomplexobj(arr) or not _is_plain_numeric_array(arr):
            continue
        if any(hint in name.lower() for hint in _FREQ_NAME_HINTS):
            return name, arr.astype(float)

    # 2) heuristic: any real, strictly increasing 1D array with plausible RF values
    candidates = []
    for name, value in variables.items():
        if not isinstance(value, np.ndarray):
            continue
        arr = np.squeeze(value)
        if arr.ndim != 1 or arr.size < 2 or np.iscomplexobj(arr):
            continue
        if not _is_plain_numeric_array(arr):
            continue
        arr = arr.astype(float)
        if np.all(np.diff(arr) > 0) and np.all(arr >= 0):
            candidates.append((name, arr))

    if candidates:
        # Prefer the candidate whose range looks like an RF frequency sweep
        candidates.sort(key=lambda item: item[1].size, reverse=True)
        return candidates[0]

    return None


def _find_s_parameters(variables: dict, freq_name: str, n_freq: int) -> tuple[str, np.ndarray] | None:
    # 1) name-based match among arrays whose leading (or trailing) dim == n_freq
    scored = []
    for name, value in variables.items():
        if name == freq_name or not isinstance(value, np.ndarray):
            continue
        arr = np.squeeze(value)
        if arr.ndim == 0 or arr.size == 0 or not _is_plain_numeric_array(arr):
            continue
        name_score = 1 if any(h in name.lower() for h in _SPARAM_NAME_HINTS) else 0
        dims_match = n_freq in arr.shape
        complex_bonus = 1 if np.iscomplexobj(arr) else 0
        scored.append((name_score * 3 + dims_match * 2 + complex_bonus, name, arr))

    if not scored:
        return None

    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, name, arr = scored[0]
    if best_score <= 0:
        return None
    return name, arr


def _orient_s_parameters(arr: np.ndarray, n_freq: int) -> np.ndarray:
    """
    Reorder an S-parameter array so the frequency axis is axis 0, then
    reshape trailing axes into a consistent (n_freq, n_ports, n_ports) or
    (n_freq, n_traces) form.
    """
    if arr.ndim == 1:
        if arr.shape[0] != n_freq:
            raise MissingVariableError(
                "S-parameter vector length does not match the frequency vector length."
            )
        return arr.reshape(n_freq, 1)

    if n_freq not in arr.shape:
        raise MissingVariableError(
            "Could not align S-parameter array dimensions with the frequency vector."
        )

    freq_axis = list(arr.shape).index(n_freq)
    arr = np.moveaxis(arr, freq_axis, 0)

    if arr.ndim == 2:
        return arr  # (n_freq, n_traces)
    if arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
        return arr  # (n_freq, n_ports, n_ports)

    # Flatten any extra dimensions into "traces"
    return arr.reshape(arr.shape[0], -1)


def _try_known_dataset_fallback(numeric_vars: dict) -> tuple[str, np.ndarray, str, np.ndarray] | None:
    """
    Fallback for datasets that store raw S-parameter arrays without any
    frequency vector at all, because the sweep is a fixed, documented
    property of the measurement system (see
    _KNOWN_DATASET_FREQUENCY_SWEEPS above) rather than per-file data.
    Returns (freq_name, frequencies, sparam_name, sparam_array) or None.
    """
    for name, value in numeric_vars.items():
        key = name.lower()
        if key not in _KNOWN_DATASET_FREQUENCY_SWEEPS:
            continue
        start_hz, stop_hz, n_points = _KNOWN_DATASET_FREQUENCY_SWEEPS[key]
        arr = np.squeeze(value)
        if n_points not in arr.shape:
            continue
        frequencies = np.linspace(start_hz, stop_hz, n_points)
        logger.warning(
            f"No frequency vector present in file for variable '{name}'; using "
            f"the documented frequency sweep for this known dataset format: "
            f"{start_hz / 1e9:.2f}-{stop_hz / 1e9:.2f} GHz, {n_points} points. "
            "Verify this matches your actual measurement system if in doubt."
        )
        return name, frequencies, name, arr
    return None


def load_matlab_dataset(file_path: str) -> MicrowaveDataset:
    """
    Purpose:
        Load a .mat file (any MATLAB version) into a MicrowaveDataset.
    Input:
        file_path (str): path to the .mat file.
    Output:
        MicrowaveDataset with frequencies, s_parameters, n_ports and
        available_variables populated.
    Raises:
        UnsupportedFileFormatError, EmptyDatasetError, CorruptedFileError,
        MissingVariableError, DatasetValidationError.
    """
    validate_file_path(file_path)
    logger.info(f"Loading MATLAB dataset: {file_path}")

    if _is_legacy_mat(file_path):
        variables = _load_legacy_mat(file_path)
        file_type_note = "MATLAB (legacy v4/v5/v7)"
    else:
        variables = _load_v73_mat(file_path)
        file_type_note = "MATLAB (v7.3 / HDF5)"

    if not variables:
        raise CorruptedFileError(
            f"No readable variables found in '{os.path.basename(file_path)}'."
        )

    numeric_vars = {
        name: value
        for name, value in variables.items()
        if isinstance(value, np.ndarray) and _is_plain_numeric_array(np.squeeze(value))
    }
    if not numeric_vars:
        raise MissingVariableError(
            f"'{os.path.basename(file_path)}' does not contain any plain numeric "
            "arrays (only MATLAB structs/cell arrays/strings were found: "
            f"{', '.join(variables.keys())}). This looks like a metadata-only "
            "file rather than a measurement file. Try loading the corresponding "
            "data file instead (e.g. if this is 'train_md.mat', look for "
            "'train_data.mat')."
        )

    physical_metadata = extract_physical_metadata_from_values(variables)

    freq_result = _find_frequency_vector(variables)
    if freq_result is not None:
        freq_name, frequencies = freq_result
        # Heuristic: MATLAB frequency vectors are very often stored in GHz.
        # If max value is small (< 1000), assume GHz and convert to Hz.
        if np.max(frequencies) < 1000:
            frequencies = frequencies * 1e9
        n_freq = frequencies.shape[0]

        sparam_result = _find_s_parameters(variables, freq_name, n_freq)
        if sparam_result is None:
            raise MissingVariableError(
                "Could not locate an S-parameter array in the MATLAB file. "
                f"Available variables: {', '.join(variables.keys())}"
            )
        sparam_name, sparam_raw = sparam_result
    else:
        known = _try_known_dataset_fallback(numeric_vars)
        if known is None:
            raise MissingVariableError(
                "Could not locate a frequency vector in the MATLAB file. "
                f"Numeric variables found: {', '.join(numeric_vars.keys())}. "
                f"All variables: {', '.join(variables.keys())}"
            )
        freq_name, frequencies, sparam_name, sparam_raw = known
        n_freq = frequencies.shape[0]

    s_parameters = _orient_s_parameters(np.squeeze(sparam_raw), n_freq)
    if not np.iscomplexobj(s_parameters):
        s_parameters = s_parameters.astype(complex)

    n_ports = s_parameters.shape[1] if s_parameters.ndim == 3 else 1

    dataset = MicrowaveDataset(
        file_path=file_path,
        file_name=os.path.basename(file_path),
        file_type=file_type_note,
        frequencies=frequencies,
        s_parameters=s_parameters,
        n_ports=n_ports,
        available_variables=sorted(variables.keys()),
        metadata={
            "frequency_variable": freq_name,
            "sparameter_variable": sparam_name,
            **physical_metadata,
            "antenna_radius_m": physical_metadata.get("antenna_radius_m", 0.08),
            "wave_speed_m_per_s": physical_metadata.get("wave_speed_m_per_s", 3e8),
            "reconstruction_x_span_m": physical_metadata.get("reconstruction_x_span_m", (-0.05, 0.05)),
            "reconstruction_y_span_m": physical_metadata.get("reconstruction_y_span_m", (-0.05, 0.05)),
        },
        raw={},
    )

    validate_dataset(dataset)
    logger.info(
        f"MATLAB dataset loaded: {n_freq} frequencies, "
        f"S-parameters shape {s_parameters.shape}"
    )
    return dataset

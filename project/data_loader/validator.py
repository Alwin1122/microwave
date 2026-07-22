"""
data_loader/validator.py

Purpose:
    Validate raw and loaded microwave datasets before they are handed to
    the preprocessing pipeline, catching corrupted files, missing
    variables, empty datasets, invalid frequency vectors, and invalid
    S-parameter data with clear, user-facing error messages.

Input:
    - File paths (for extension / existence checks)
    - MicrowaveDataset objects (for content checks)

Output:
    - Raises specific exceptions from utils.exceptions on failure
    - Returns True (and optionally a list of non-fatal warnings) on success

Description:
    Used by both loaders (matlab_loader.py, touchstone_loader.py) right
    after parsing, and can also be called again by the GUI/pipeline as a
    defensive second check before preprocessing starts.
"""

from __future__ import annotations

import os

import numpy as np

from utils.exceptions import (
    DatasetValidationError,
    EmptyDatasetError,
    InvalidFrequencyError,
    InvalidSParameterError,
    UnsupportedFileFormatError,
)

SUPPORTED_EXTENSIONS = {".mat", ".s1p", ".s2p", ".s4p", ".s8p"}


def validate_file_path(file_path: str) -> None:
    """
    Purpose:
        Verify a file exists and has a supported extension before any
        parsing is attempted.
    Input:
        file_path (str): path to the candidate dataset file.
    Output:
        None. Raises FileNotFoundError or UnsupportedFileFormatError.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if os.path.getsize(file_path) == 0:
        raise EmptyDatasetError(f"File '{os.path.basename(file_path)}' is empty (0 bytes).")


def validate_frequencies(frequencies: np.ndarray) -> list[str]:
    """
    Purpose:
        Validate a frequency vector extracted from a dataset.
    Input:
        frequencies (np.ndarray): 1D array of frequency values in Hz.
    Output:
        list[str]: non-fatal warnings (e.g. non-monotonic ordering that was
        auto-sorted). Raises InvalidFrequencyError / EmptyDatasetError for
        fatal problems.
    """
    warnings: list[str] = []

    if frequencies is None or np.size(frequencies) == 0:
        raise EmptyDatasetError("Frequency vector is empty.")

    if not np.issubdtype(frequencies.dtype, np.number):
        raise InvalidFrequencyError("Frequency vector contains non-numeric values.")

    if np.any(~np.isfinite(frequencies)):
        raise InvalidFrequencyError("Frequency vector contains NaN or Inf values.")

    if np.any(frequencies < 0):
        raise InvalidFrequencyError("Frequency vector contains negative values.")

    if np.size(frequencies) > 1 and not np.all(np.diff(frequencies) > 0):
        if np.all(np.diff(np.sort(frequencies)) >= 0):
            warnings.append(
                "Frequency values were not monotonically increasing; "
                "consider re-sorting before analysis."
            )
        else:
            raise InvalidFrequencyError(
                "Frequency vector contains duplicate or unusable values."
            )

    return warnings


def validate_s_parameters(s_params: np.ndarray, n_frequencies: int) -> list[str]:
    """
    Purpose:
        Validate an S-parameter array for shape/content correctness.
    Input:
        s_params (np.ndarray): complex S-parameter array. Its first axis
            must correspond to frequency points.
        n_frequencies (int): expected number of frequency points, used to
            cross-check array shape.
    Output:
        list[str]: non-fatal warnings. Raises InvalidSParameterError /
        EmptyDatasetError for fatal problems.
    """
    warnings: list[str] = []

    if s_params is None or np.size(s_params) == 0:
        raise EmptyDatasetError("S-parameter data is empty.")

    if s_params.shape[0] != n_frequencies:
        raise InvalidSParameterError(
            f"S-parameter array's first dimension ({s_params.shape[0]}) does not "
            f"match the number of frequency points ({n_frequencies})."
        )

    if np.any(~np.isfinite(s_params.real)) or (
        np.iscomplexobj(s_params) and np.any(~np.isfinite(s_params.imag))
    ):
        raise InvalidSParameterError("S-parameter data contains NaN or Inf values.")

    magnitude = np.abs(s_params)
    if np.any(magnitude > 10):
        warnings.append(
            "Some S-parameter magnitudes exceed 10 (20 dB gain), which is "
            "unusual for passive measurements; verify calibration."
        )

    if np.all(magnitude == 0):
        raise InvalidSParameterError("All S-parameter values are zero.")

    return warnings


def validate_dataset(dataset) -> list[str]:
    """
    Purpose:
        Run all content-level validation checks on a fully loaded
        MicrowaveDataset in one call. Intended as the final gate before a
        dataset reaches the preprocessing pipeline.
    Input:
        dataset (MicrowaveDataset)
    Output:
        list[str] of non-fatal warnings. Raises DatasetValidationError
        (wrapping the first fatal error encountered) on failure.
    """
    warnings: list[str] = []
    try:
        warnings += validate_frequencies(dataset.frequencies)
        warnings += validate_s_parameters(dataset.s_parameters, dataset.n_frequencies)
    except (EmptyDatasetError, InvalidFrequencyError, InvalidSParameterError) as exc:
        raise DatasetValidationError(str(exc), errors=[str(exc)]) from exc

    return warnings

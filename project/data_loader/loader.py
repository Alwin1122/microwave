"""
data_loader/loader.py

Purpose:
    Single entry point for Module 1 (Data Acquisition). Inspects a file's
    extension and dispatches to the appropriate format-specific loader
    (matlab_loader or touchstone_loader), presenting one unified,
    format-agnostic API to the GUI and to Module 2.

Input:
    file_path (str): path to a .mat / .s1p / .s2p / .s4p / .s8p file.

Output:
    MicrowaveDataset (see data_loader/dataset_info.py)

Description:
    This is the function the "Load Dataset" GUI button calls. It also
    catches and re-raises framework exceptions unchanged so the GUI's
    single except-block can display a meaningful message regardless of
    file type.
"""

from __future__ import annotations

import os

from data_loader.dataset_info import DatasetSummary, MicrowaveDataset, build_summary
from data_loader.matlab_loader import load_matlab_dataset
from data_loader.touchstone_loader import load_touchstone_dataset
from data_loader.validator import SUPPORTED_EXTENSIONS
from utils.exceptions import UnsupportedFileFormatError
from utils.logger import get_logger

logger = get_logger(__name__)

_TOUCHSTONE_EXTENSIONS = {".s1p", ".s2p", ".s4p", ".s8p"}


def load_dataset(file_path: str, scan_index: int | None = None) -> MicrowaveDataset:
    """
    Purpose:
        Load any supported microwave measurement file into a
        MicrowaveDataset, regardless of format.
    Input:
        file_path (str): path to the dataset file.
        scan_index (int | None): required for multi-scan UM-BMID cubes.
    Output:
        MicrowaveDataset instance.
    Raises:
        UnsupportedFileFormatError if the extension is not recognized;
        propagates any exception raised by the underlying format loader.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".mat":
        return load_matlab_dataset(file_path, scan_index=scan_index)
    if ext in _TOUCHSTONE_EXTENSIONS:
        return load_touchstone_dataset(file_path)

    raise UnsupportedFileFormatError(
        f"Unsupported file format '{ext}'. "
        f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def load_dataset_with_summary(
    file_path: str, scan_index: int | None = None
) -> tuple[MicrowaveDataset, DatasetSummary]:
    """
    Purpose:
        Convenience wrapper used by the GUI: loads a dataset and
        immediately builds its display summary.
    Input:
        file_path (str): path to the dataset file.
        scan_index (int | None): required for multi-scan UM-BMID cubes.
    Output:
        (MicrowaveDataset, DatasetSummary) tuple.
    """
    dataset = load_dataset(file_path, scan_index=scan_index)
    summary = build_summary(dataset)
    return dataset, summary

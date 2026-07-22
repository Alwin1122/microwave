"""
data_loader/dataset_info.py

Purpose:
    Define the core in-memory representation of a loaded microwave
    measurement dataset (`MicrowaveDataset`) and a human-readable summary
    structure (`DatasetSummary`) used to populate the "Dataset Information"
    panel in the GUI.

Input:
    Raw arrays produced by the individual loaders (matlab_loader.py,
    touchstone_loader.py).

Output:
    Structured, strongly-typed objects that Module 2 (preprocessing) and
    the GUI both consume, decoupling them from the specifics of any one
    file format.

Description:
    Every loader (MATLAB or Touchstone) ultimately produces one
    `MicrowaveDataset` instance so downstream code never needs to know
    which file type the data originally came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MicrowaveDataset:
    """
    Purpose:
        Canonical container for a single loaded microwave measurement
        dataset, regardless of original file format.

    Fields:
        file_path (str): Absolute path to the source file.
        file_name (str): Base file name.
        file_type (str): 'MATLAB' or 'Touchstone'.
        frequencies (np.ndarray): 1D array of frequency points in Hz.
        s_parameters (np.ndarray): Complex S-parameter array.
            Shape convention: (n_freq, n_ports, n_ports) for a full
            multi-port matrix (Touchstone), or (n_freq, n_traces) for
            MATLAB data organized as independent measurement traces
            (common in UWB breast-imaging style antenna array datasets).
        n_ports (int): Number of physical ports (1, 2, 4, 8...). For
            trace-organized MATLAB data this is the number of antennas.
        available_variables (list[str]): Variable/field names found in the
            raw file (mainly relevant for .mat files).
        metadata (dict): Any extra information captured during loading
            (e.g. Touchstone comments, MATLAB struct names, units).
        raw (dict): Reference to the raw loaded object, kept for advanced
            use, debugging, or re-export. Not required by Module 2.
    """

    file_path: str
    file_name: str
    file_type: str
    frequencies: np.ndarray
    s_parameters: np.ndarray
    n_ports: int
    available_variables: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def n_frequencies(self) -> int:
        return int(self.frequencies.shape[0]) if self.frequencies is not None else 0

    @property
    def n_samples(self) -> int:
        """Total number of scalar complex samples in the S-parameter data."""
        return int(np.size(self.s_parameters)) if self.s_parameters is not None else 0

    @property
    def freq_range_hz(self) -> tuple[float, float]:
        if self.frequencies is None or self.n_frequencies == 0:
            return (0.0, 0.0)
        return float(np.min(self.frequencies)), float(np.max(self.frequencies))

    @property
    def dataset_size_bytes(self) -> int:
        size = 0
        if self.s_parameters is not None:
            size += self.s_parameters.nbytes
        if self.frequencies is not None:
            size += self.frequencies.nbytes
        return size


@dataclass
class DatasetSummary:
    """
    Purpose:
        Lightweight, display-ready summary of a MicrowaveDataset, matching
        exactly the fields the GUI's "Dataset Information" panel must show.
    """

    file_name: str
    file_type: str
    n_samples: int
    n_frequencies: int
    freq_range_hz: tuple[float, float]
    n_ports: int
    dataset_size_bytes: int
    available_variables: list[str]

    def freq_range_str(self) -> str:
        low, high = self.freq_range_hz
        return f"{low / 1e9:.4f} GHz – {high / 1e9:.4f} GHz"

    def dataset_size_str(self) -> str:
        size = self.dataset_size_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"

    def to_display_dict(self) -> dict:
        """Return an ordered dict exactly matching the required GUI fields."""
        return {
            "File Name": self.file_name,
            "File Type": self.file_type,
            "Number of Samples": f"{self.n_samples:,}",
            "Number of Frequencies": f"{self.n_frequencies:,}",
            "Frequency Range": self.freq_range_str(),
            "Number of Ports": str(self.n_ports),
            "Dataset Size": self.dataset_size_str(),
            "Available Variables": ", ".join(self.available_variables) or "N/A",
        }

    def to_text(self) -> str:
        lines = [f"{k}: {v}" for k, v in self.to_display_dict().items()]
        return "\n".join(lines)


def build_summary(dataset: MicrowaveDataset) -> DatasetSummary:
    """
    Purpose:
        Convert a MicrowaveDataset into a DatasetSummary for display.
    Input:
        dataset (MicrowaveDataset): a fully loaded dataset.
    Output:
        DatasetSummary instance.
    """
    return DatasetSummary(
        file_name=dataset.file_name,
        file_type=dataset.file_type,
        n_samples=dataset.n_samples,
        n_frequencies=dataset.n_frequencies,
        freq_range_hz=dataset.freq_range_hz,
        n_ports=dataset.n_ports,
        dataset_size_bytes=dataset.dataset_size_bytes,
        available_variables=dataset.available_variables,
    )

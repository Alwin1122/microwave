"""
data_loader/touchstone_loader.py

Purpose:
    Load microwave measurement datasets stored as Touchstone files
    (.s1p, .s2p, .s4p, .s8p) using scikit-rf, and convert them into the
    framework's canonical MicrowaveDataset representation.

Input:
    file_path (str): path to a Touchstone file.

Output:
    A MicrowaveDataset instance with frequencies (Hz), a full complex
    S-parameter matrix of shape (n_freq, n_ports, n_ports), n_ports, and
    any Touchstone comments captured in metadata.

Description:
    scikit-rf's `Network` class already implements a robust Touchstone
    parser (including port renumbering, noise data, and comment
    extraction), so this loader is primarily a thin adapter that maps a
    parsed Network into MicrowaveDataset and applies the framework's
    validation rules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import skrf

from data_loader.dataset_info import MicrowaveDataset
from data_loader.physical_metadata import extract_physical_metadata_from_text
from data_loader.validator import validate_dataset, validate_file_path
from utils.exceptions import CorruptedFileError, UnsupportedFileFormatError
from utils.logger import get_logger

logger = get_logger(__name__)

_EXT_TO_PORTS = {".s1p": 1, ".s2p": 2, ".s4p": 4, ".s8p": 8}
_FREQUENCY_UNIT_SCALE = {
    "HZ": 1.0,
    "KHZ": 1e3,
    "MHZ": 1e6,
    "GHZ": 1e9,
}
_SUPPORTED_DATA_FORMATS = {"DB", "MA", "RI"}


@dataclass(frozen=True)
class TouchstoneHeaderInfo:
    """Parsed metadata from a Touchstone option line and frequency sweep."""

    frequency_unit: str
    data_format: str
    reference_impedance_ohm: float
    start_frequency_hz: float
    end_frequency_hz: float
    n_frequency_samples: int

    def to_dict(self) -> dict:
        return {
            "frequency_unit": self.frequency_unit,
            "data_format": self.data_format,
            "reference_impedance_ohm": self.reference_impedance_ohm,
            "start_frequency_hz": self.start_frequency_hz,
            "end_frequency_hz": self.end_frequency_hz,
            "n_frequency_samples": self.n_frequency_samples,
        }


def _clean_touchstone_line(line: str) -> str:
    return line.split("!", 1)[0].strip()


def _require_s2p(file_path: str) -> None:
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".s2p":
        raise UnsupportedFileFormatError(
            f"S21 extraction requires a Touchstone .s2p file, received '{ext}'."
        )


def _read_touchstone_option_tokens(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            if stripped.startswith("#"):
                return stripped[1:].split()
    raise CorruptedFileError(
        f"Touchstone file '{os.path.basename(file_path)}' is missing the option line header."
    )


def _parse_touchstone_numeric_rows(file_path: str) -> tuple[list[float], list[list[float]]]:
    header_seen = False
    numeric_rows: list[list[float]] = []
    current_frequency: float | None = None
    current_values: list[float] = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            if stripped.startswith("#"):
                header_seen = True
                continue
            if not header_seen:
                continue

            content = _clean_touchstone_line(raw_line)
            if not content:
                continue

            tokens = content.split()
            try:
                numeric_tokens = [float(token) for token in tokens]
            except ValueError as exc:
                raise CorruptedFileError(
                    f"Non-numeric Touchstone sample found on line {line_number}: {content}"
                ) from exc

            index = 0
            while index < len(numeric_tokens):
                if current_frequency is None:
                    current_frequency = numeric_tokens[index]
                    index += 1

                remaining_needed = 8 - len(current_values)
                take = min(remaining_needed, len(numeric_tokens) - index)
                current_values.extend(numeric_tokens[index:index + take])
                index += take

                if len(current_values) == 8:
                    numeric_rows.append([current_frequency, *current_values])
                    current_frequency = None
                    current_values = []

    if current_frequency is not None or current_values:
        raise CorruptedFileError(
            f"Touchstone file '{os.path.basename(file_path)}' ended with an incomplete S-parameter row."
        )

    if not numeric_rows:
        raise CorruptedFileError(
            f"Touchstone file '{os.path.basename(file_path)}' does not contain any usable S-parameter rows."
        )

    frequencies = [row[0] for row in numeric_rows]
    return frequencies, numeric_rows


def parse_touchstone_header(file_path: str) -> TouchstoneHeaderInfo:
    """Parse the option line and sweep statistics from a Touchstone .s2p file."""
    _require_s2p(file_path)
    validate_file_path(file_path)

    option_tokens = _read_touchstone_option_tokens(file_path)
    if len(option_tokens) < 5:
        raise CorruptedFileError(
            f"Touchstone option line in '{os.path.basename(file_path)}' is incomplete."
        )

    frequency_unit = option_tokens[0].upper()
    parameter_type = option_tokens[1].upper()
    data_format = option_tokens[2].upper()
    reference_marker = option_tokens[3].upper()

    if frequency_unit not in _FREQUENCY_UNIT_SCALE:
        raise CorruptedFileError(f"Unsupported Touchstone frequency unit '{frequency_unit}'.")
    if parameter_type != "S":
        raise CorruptedFileError(
            f"Only S-parameter Touchstone files are supported, received '{parameter_type}'."
        )
    if data_format not in _SUPPORTED_DATA_FORMATS:
        raise CorruptedFileError(f"Unsupported Touchstone data format '{data_format}'.")
    if reference_marker != "R":
        raise CorruptedFileError(
            f"Touchstone option line must declare the reference impedance with 'R', received '{reference_marker}'."
        )

    try:
        reference_impedance = float(option_tokens[4])
    except ValueError as exc:
        raise CorruptedFileError(
            f"Invalid reference impedance in Touchstone option line: {' '.join(option_tokens)}"
        ) from exc

    frequencies_raw, _ = _parse_touchstone_numeric_rows(file_path)
    frequencies_hz = np.asarray(frequencies_raw, dtype=float) * _FREQUENCY_UNIT_SCALE[frequency_unit]

    return TouchstoneHeaderInfo(
        frequency_unit=frequency_unit,
        data_format=data_format,
        reference_impedance_ohm=reference_impedance,
        start_frequency_hz=float(np.min(frequencies_hz)),
        end_frequency_hz=float(np.max(frequencies_hz)),
        n_frequency_samples=int(frequencies_hz.shape[0]),
    )


def load_touchstone_s21_trace(file_path: str) -> tuple[np.ndarray, np.ndarray, TouchstoneHeaderInfo]:
    """Extract the complex S21 trace from a Touchstone .s2p file."""
    header = parse_touchstone_header(file_path)
    _, numeric_rows = _parse_touchstone_numeric_rows(file_path)
    numeric = np.asarray(numeric_rows, dtype=float)
    frequencies_hz = numeric[:, 0] * _FREQUENCY_UNIT_SCALE[header.frequency_unit]

    first = numeric[:, 3]
    second = numeric[:, 4]

    if header.data_format == "DB":
        magnitude = np.power(10.0, first / 20.0)
        phase_radians = np.deg2rad(second)
        s21 = magnitude * (np.cos(phase_radians) + 1j * np.sin(phase_radians))
    elif header.data_format == "MA":
        phase_radians = np.deg2rad(second)
        s21 = first * (np.cos(phase_radians) + 1j * np.sin(phase_radians))
    else:
        s21 = first + 1j * second

    return frequencies_hz.astype(float), np.asarray(s21, dtype=complex), header


def load_touchstone_dataset(file_path: str) -> MicrowaveDataset:
    """
    Purpose:
        Load a Touchstone (.sNp) file into a MicrowaveDataset.
    Input:
        file_path (str): path to the .s1p/.s2p/.s4p/.s8p file.
    Output:
        MicrowaveDataset with frequencies, s_parameters (n_freq, n_ports,
        n_ports), and n_ports populated.
    Raises:
        UnsupportedFileFormatError, EmptyDatasetError, CorruptedFileError,
        DatasetValidationError.
    """
    validate_file_path(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    expected_ports = _EXT_TO_PORTS.get(ext)
    logger.info(f"Loading Touchstone dataset: {file_path}")

    try:
        network = skrf.Network(file_path)
    except Exception as exc:  # scikit-rf raises plain Exception/ValueError on bad files
        raise CorruptedFileError(
            f"Could not parse Touchstone file '{os.path.basename(file_path)}': {exc}"
        ) from exc

    frequencies = np.asarray(network.f, dtype=float)  # already in Hz
    s_parameters = np.asarray(network.s, dtype=complex)  # shape (n_freq, n_ports, n_ports)
    n_ports = network.nports

    if expected_ports is not None and n_ports != expected_ports:
        logger.warning(
            f"File extension '{ext}' suggests {expected_ports} port(s) but "
            f"parsed data has {n_ports} port(s); using parsed value."
        )

    comments = getattr(network, "comments", "") or ""
    port_names = getattr(network, "port_names", None)
    physical_metadata = extract_physical_metadata_from_text(comments)
    touchstone_header = None
    touchstone_s21 = None
    if ext == ".s2p":
        try:
            _, touchstone_s21, touchstone_header = load_touchstone_s21_trace(file_path)
        except CorruptedFileError:
            raise
        except Exception as exc:
            raise CorruptedFileError(
                f"Could not extract S21 from Touchstone file '{os.path.basename(file_path)}': {exc}"
            ) from exc

    dataset = MicrowaveDataset(
        file_path=file_path,
        file_name=os.path.basename(file_path),
        file_type=f"Touchstone (.s{n_ports}p)",
        frequencies=frequencies,
        s_parameters=s_parameters,
        n_ports=n_ports,
        available_variables=[f"S{i + 1}{j + 1}" for i in range(n_ports) for j in range(n_ports)],
        metadata={
            "comments": comments.strip(),
            "port_names": port_names,
            "z0": np.asarray(network.z0[0]).tolist() if network.z0 is not None else None,
            "touchstone_header": touchstone_header.to_dict() if touchstone_header is not None else None,
            **physical_metadata,
            "antenna_radius_m": physical_metadata.get("antenna_radius_m", 0.08),
            "wave_speed_m_per_s": physical_metadata.get("wave_speed_m_per_s", 3e8),
            "reconstruction_x_span_m": physical_metadata.get("reconstruction_x_span_m", (-0.05, 0.05)),
            "reconstruction_y_span_m": physical_metadata.get("reconstruction_y_span_m", (-0.05, 0.05)),
        },
        raw={
            "touchstone_s21": touchstone_s21,
        },
    )

    validate_dataset(dataset)
    logger.info(
        f"Touchstone dataset loaded: {frequencies.shape[0]} frequencies, "
        f"{n_ports} port(s)"
    )
    return dataset

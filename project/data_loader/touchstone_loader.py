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

import numpy as np
import skrf

from data_loader.dataset_info import MicrowaveDataset
from data_loader.validator import validate_dataset, validate_file_path
from utils.exceptions import CorruptedFileError
from utils.logger import get_logger

logger = get_logger(__name__)

_EXT_TO_PORTS = {".s1p": 1, ".s2p": 2, ".s4p": 4, ".s8p": 8}


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
        },
        raw={},
    )

    validate_dataset(dataset)
    logger.info(
        f"Touchstone dataset loaded: {frequencies.shape[0]} frequencies, "
        f"{n_ports} port(s)"
    )
    return dataset

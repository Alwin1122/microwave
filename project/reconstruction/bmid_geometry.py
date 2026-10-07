"""Calibrated UM-BMID imaging geometry per dataset generation.

Values come from ``tools/calibrate_bmid_geometry.py`` (empty-chamber plus
rotational-mean subtraction, coherent DAS). Only gen3 localises tumours better
than guessing the centre; gen1/gen2 settings are the least-bad found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BmidGeometry:
    generation: str
    wave_speed: float
    clockwise: bool
    offset_deg: float
    use_phase_delay_radius: bool
    span_deg: float = 355.0
    localises: bool = False


BMID_GEOMETRY = {
    "gen1": BmidGeometry("gen1", 2.7e8, False, 315.0, False),
    "gen2": BmidGeometry("gen2", 3.0e8, True, 222.5, False),
    "gen3": BmidGeometry("gen3", 2.1e8, True, 170.0, True, localises=True),
}

_NAME_TO_GEN = {"one": "gen1", "two": "gen2", "three": "gen3"}


def generation_from_path(path: str) -> str | None:
    """Return 'gen1'/'gen2'/'gen3' from a UM-BMID file name such as fd_data_gen_two_s11.mat."""
    match = re.search(r"gen[_-]?(one|two|three|1|2|3)", str(path).lower())
    if not match:
        return None
    token = match.group(1)
    return _NAME_TO_GEN.get(token, f"gen{token}")


def remove_rotational_mean(s_parameters: np.ndarray) -> np.ndarray:
    """Subtract the average over antennas; cancels anything that looks the same from every angle."""
    data = np.asarray(s_parameters)
    return data - data.mean(axis=1, keepdims=True)


def remove_low_rank(s_parameters: np.ndarray, rank: int) -> np.ndarray:
    """Remove the ``rank`` strongest patterns shared by all antennas (SVD over the antenna axis)."""
    data = np.asarray(s_parameters, dtype=complex)
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    return data - (u[:, :rank] * s[:rank]) @ vh[:rank]


def time_gate(s_parameters: np.ndarray, frequencies: np.ndarray, start_s: float) -> np.ndarray:
    """Zero every echo arriving before ``start_s`` (e.g. the skin reflection) and return to frequency."""
    data = np.asarray(s_parameters, dtype=complex)
    freqs = np.asarray(frequencies, dtype=float)
    if start_s <= 0:
        return data
    df = float(np.mean(np.diff(freqs)))
    n_time = data.shape[0]
    t = np.arange(n_time) / (n_time * df)
    baseband = np.fft.ifft(data, axis=0)
    baseband[t < start_s] = 0.0
    return np.fft.fft(baseband, axis=0)

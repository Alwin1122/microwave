"""
datasets/generate_sample_data.py

Purpose:
    Generate small, synthetic sample datasets (.mat and .sNp) so the
    application and its unit tests can be exercised without requiring
    real lab measurement files.

Input:
    None (run as a script).

Output:
    Writes sample files into the datasets/ folder:
        sample_matlab.mat        (legacy MAT format, 8 antenna traces)
        sample_matlab_v73.mat    (MAT v7.3 / HDF5 format)
        sample_touchstone.s2p    (2-port Touchstone file)
        corrupted.mat            (intentionally invalid file, for testing
                                   error handling)

Description:
    Synthetic S-parameters are built from a smooth resonance-like curve
    plus small per-trace phase/amplitude variation and additive noise, so
    that filtering/calibration/normalization/artifact-suppression stages
    all have something meaningful to act on.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import skrf
from scipy.io import savemat

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _synthetic_traces(n_freq: int = 201, n_traces: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    freqs = np.linspace(1e9, 9e9, n_freq)  # 1-9 GHz, typical UWB breast imaging band

    center = 5e9
    bandwidth = 3e9
    base_response = 0.6 * np.exp(-((freqs - center) ** 2) / (2 * bandwidth**2))

    traces = np.zeros((n_freq, n_traces), dtype=complex)
    for i in range(n_traces):
        amp_variation = 1.0 + 0.05 * np.sin(2 * np.pi * i / n_traces)
        phase_shift = 0.3 * i
        target_bump = 0.03 * np.exp(-((freqs - (5.5e9 + 0.05e9 * i)) ** 2) / (2 * (0.15e9) ** 2))
        magnitude = amp_variation * base_response + target_bump
        phase = -freqs / 1e9 * 0.8 + phase_shift
        clean = magnitude * np.exp(1j * phase)
        noise = (rng.normal(scale=0.01, size=n_freq) + 1j * rng.normal(scale=0.01, size=n_freq))
        traces[:, i] = clean + noise

    return freqs, traces


def generate_legacy_mat():
    freqs, traces = _synthetic_traces(seed=1)
    path = os.path.join(OUT_DIR, "sample_matlab.mat")
    savemat(
        path,
        {
            "frequency": freqs / 1e9,  # stored in GHz, loader auto-converts to Hz
            "s_parameters": traces,
            "notes": (
                "Synthetic 8-antenna UWB breast imaging dataset (legacy .mat); "
                "antenna radius=8 cm; wave speed=3.0e8 m/s; x span=-5 cm to 5 cm; y span=-5 cm to 5 cm"
            ),
        },
    )
    print(f"Wrote {path}")


def generate_v73_mat():
    freqs, traces = _synthetic_traces(seed=2)
    path = os.path.join(OUT_DIR, "sample_matlab_v73.mat")
    with h5py.File(path, "w") as f:
        f.create_dataset("frequency", data=freqs / 1e9)
        f.create_dataset("s_parameters_real", data=traces.real)
        f.create_dataset("s_parameters_imag", data=traces.imag)
        # store as a single complex-like structured field too, for realism
        f.create_dataset("s_data", data=traces.real)  # magnitude-only variable, alt path
    print(f"Wrote {path}")


def generate_touchstone():
    freqs, traces = _synthetic_traces(n_traces=4, seed=3)
    # Build a pseudo 2-port network from the first two traces
    n_freq = freqs.shape[0]
    s = np.zeros((n_freq, 2, 2), dtype=complex)
    s[:, 0, 0] = traces[:, 0]
    s[:, 1, 1] = traces[:, 1]
    s[:, 0, 1] = traces[:, 2] * 0.5
    s[:, 1, 0] = traces[:, 2] * 0.5

    freq_obj = skrf.Frequency.from_f(freqs / 1e9, unit="ghz")
    network = skrf.Network(frequency=freq_obj, s=s, z0=50)
    network.comments = (
        "Synthetic 2-port Touchstone sample dataset; antenna radius=8 cm; "
        "wave speed=3.0e8 m/s; x span=-5 cm to 5 cm; y span=-5 cm to 5 cm"
    )
    path = os.path.join(OUT_DIR, "sample_touchstone.s2p")
    network.write_touchstone(path, form="ri")
    print(f"Wrote {path}")


def generate_corrupted_file():
    path = os.path.join(OUT_DIR, "corrupted.mat")
    with open(path, "wb") as f:
        f.write(b"THIS IS NOT A VALID MAT FILE" * 5)
    print(f"Wrote {path}")


if __name__ == "__main__":
    generate_legacy_mat()
    generate_v73_mat()
    generate_touchstone()
    generate_corrupted_file()

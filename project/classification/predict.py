"""Per-scan features and predictions with the saved tumor models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from classification.criticality import criticality_from_scan
from classification.features import calibrate_empty, image_features, signal_features
from classification.scattering_math import scattering_descriptors
from data_loader.bmid_loader import bmid_frequencies_hz
from reconstruction.bmid_geometry import BMID_GEOMETRY, remove_rotational_mean
from reconstruction.das import das_coherent_from_frequency
from reconstruction.reconstruction_manager import (
    ReconstructionConfig,
    _build_grid,
    _reconstruct_single,
    antenna_positions_from_config,
)

_RESULTS = Path(__file__).resolve().parent.parent / "results" / "classification"
MODEL_PATH = _RESULTS / "tumor_models.joblib"
MODEL_PATHS = {
    "emp": MODEL_PATH,
    "twin": _RESULTS / "tumor_models_twin.joblib",
    "adi": _RESULTS / "tumor_models_adi_gen3.joblib",
}
SPAN = (-0.06, 0.06)
GRID = 64
# Chosen by tools/compare_background_removal.py (best blind localisation on gen3, both phantom halves).
LOW_BAND_HZ = (1.0e9, 4.0e9)


def recon_config(generation: str, ant_rad_cm: float) -> ReconstructionConfig:
    geometry = BMID_GEOMETRY[generation]
    return ReconstructionConfig(
        x_span=SPAN,
        y_span=SPAN,
        n_x=GRID,
        n_y=GRID,
        antenna_radius=float(ant_rad_cm) / 100.0,
        wave_speed=geometry.wave_speed,
        antenna_angle_offset_deg=geometry.offset_deg,
        antenna_clockwise=geometry.clockwise,
        antenna_span_deg=geometry.span_deg,
        use_bmid_phase_delay_radius=geometry.use_phase_delay_radius,
    )


def scan_features(generation: str, ant_rad_cm: float, raw: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """All features of one scan; ``reference`` is the scan subtracted as background."""
    freqs = bmid_frequencies_hz()
    raw = np.asarray(raw, dtype=complex)
    cal = calibrate_empty(raw, reference)
    cal_rot = remove_rotational_mean(cal)
    config = recon_config(generation, ant_rad_cm)
    positions = antenna_positions_from_config(raw.shape[1], config)
    grid_x, grid_y = _build_grid(config.x_span, config.y_span, n_x=config.n_x, n_y=config.n_y)

    feats = signal_features(raw, cal, freqs)
    feats.update(signal_features(raw, cal_rot, freqs, prefix="rsig"))
    feats.update(scattering_descriptors(cal, freqs))
    das = das_coherent_from_frequency(cal_rot, freqs, positions, grid_x, grid_y, wave_speed=config.wave_speed)
    feats.update(image_features(das, config.x_span, config.y_span, "img"))
    low = (freqs >= LOW_BAND_HZ[0]) & (freqs <= LOW_BAND_HZ[1])
    das_low = das_coherent_from_frequency(cal_rot[low], freqs[low], positions, grid_x, grid_y, wave_speed=config.wave_speed)
    feats.update(image_features(das_low, config.x_span, config.y_span, "imgl"))
    d4 = _reconstruct_single("DMAS-D4", cal_rot, freqs, positions, grid_x, grid_y, wave_speed=config.wave_speed, config=config)
    feats.update(image_features(d4, config.x_span, config.y_span, "d4img"))
    return feats


@lru_cache(maxsize=1)
def load_models(path: str = str(MODEL_PATH)) -> dict:
    import joblib

    return joblib.load(path)


def predict_from_features(feats: dict[str, float], bundle: dict, generation: str | None = None) -> dict:
    """Tumor probability, size class, diameter and the image-spot position from one feature dict."""
    vector = np.nan_to_num(np.array([feats.get(name, 0.0) for name in bundle["feature_names"]], dtype=float), nan=0.0)
    out = {}
    for task, spec in bundle["models"].items():
        x = vector[spec["columns"]].reshape(1, -1)
        est = spec["estimator"]
        if task == "detection":
            out["tumor_probability"] = float(est.predict_proba(x)[0, list(est.classes_).index(1)])
            out["tumor_flag"] = out["tumor_probability"] >= spec.get("threshold", 0.5)
        elif task == "size_class":
            out["size_class"] = bundle["size_classes"][int(est.predict(x)[0])]
        elif task == "diameter":
            out["diameter_cm"] = float(est[0].predict(x)[0])
    out["image_spot_cm"] = (float(feats["imgl_peak_x_cm"]), float(feats["imgl_peak_y_cm"]))
    geometry = BMID_GEOMETRY.get(generation or "")
    out["image_spot_reliable"] = bool(geometry and geometry.localises)
    if "diameter_cm" in out:
        # The saved diameter model does not predict margin, so the index uses
        # the spherical factor. A spiculated margin is applied only when a
        # shape string is supplied by the caller via feats["known_shape"].
        index, criticality = criticality_from_scan(out["diameter_cm"], feats.get("known_shape"))
        out["criticality_index"] = index
        out["criticality_class"] = criticality
    return out


def predict_scan(
    raw: np.ndarray,
    reference: np.ndarray,
    ant_rad_cm: float,
    generation: str,
    bundle: dict | None = None,
    reference_kind: str = "emp",
) -> dict:
    """Predict from one scan.

    ``reference_kind`` selects the model trained for that subtraction:
    ``emp`` empty chamber, ``adi`` gen3 fat-only scan, ``twin`` earlier healthy
    scans of the same phantom. The array passed as ``reference`` must be that scan.
    """
    if reference_kind not in MODEL_PATHS:
        raise ValueError(f"reference_kind must be one of {tuple(MODEL_PATHS)}")
    bundle = bundle or load_models(str(MODEL_PATHS[reference_kind]))
    return predict_from_features(scan_features(generation, ant_rad_cm, raw, reference), bundle, generation)

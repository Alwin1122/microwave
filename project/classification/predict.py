"""Per-scan features and predictions with the saved tumor models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from classification.features import calibrate_empty, image_features, signal_features
from data_loader.bmid_loader import bmid_frequencies_hz
from reconstruction.bmid_geometry import BMID_GEOMETRY, remove_rotational_mean
from reconstruction.das import das_coherent_from_frequency
from reconstruction.reconstruction_manager import (
    ReconstructionConfig,
    _build_grid,
    _reconstruct_single,
    antenna_positions_from_config,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "results" / "classification" / "tumor_models.joblib"
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
    return out


def predict_scan(raw: np.ndarray, empty_reference: np.ndarray, ant_rad_cm: float, generation: str, bundle: dict | None = None) -> dict:
    bundle = bundle or load_models()
    return predict_from_features(scan_features(generation, ant_rad_cm, raw, empty_reference), bundle, generation)

"""Detect → locate pipeline: friend detection + our blind localization.

Keeps the two jobs separate:
  1. Classification model answers tumour? (yes/no + probability)
  2. If yes (or always_localize=True), Week 3 DAS + smoothed-peak marks (x, y)

Does not git-merge branches; this module only *calls* both APIs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from classification.predict import MODEL_PATHS, load_models, predict_scan
from data_loader.loader import MicrowaveDataset
from preprocessing.preprocessing_pipeline import PreprocessingConfig, run_preprocessing_pipeline
from reconstruction.reconstruction_manager import (
    ReconstructionConfig,
    infer_reconstruction_config,
    preferred_week3_key,
    reconstruct_all_from_time,
    week3_time_inputs,
)
from roi.roi_detector import (
    PROMINENCE_TRUST_THRESHOLD,
    peak_prominence_confidence,
    smoothed_peak_centroid_m,
)


@dataclass
class DetectLocateResult:
    """Combined detection + localization for one scan."""

    tumor_detected: bool
    tumor_probability: float
    detection_model: str
    location_cm: tuple[float, float] | None
    location_prominence: float | None
    location_trusted: bool | None
    localization_ran: bool
    notes: list[str] = field(default_factory=list)
    detection_details: dict[str, Any] = field(default_factory=dict)
    localization_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _generation(dataset: MicrowaveDataset) -> str:
    meta = dataset.metadata or {}
    gen = meta.get("bmid_generation") or "gen1"
    return str(gen)


def _ant_rad_cm(dataset: MicrowaveDataset) -> float:
    meta = dataset.metadata or {}
    if "antenna_radius_m" in meta:
        return float(meta["antenna_radius_m"]) * 100.0
    return 18.0


def resolve_reference(
    dataset: MicrowaveDataset,
    reference: np.ndarray | None = None,
    reference_kind: str | None = None,
) -> tuple[np.ndarray, str, list[str]]:
    """Pick a background reference and matching classifier bundle.

    Preference: caller override → empty-chamber in dataset.raw → fail with notes.
    """
    notes: list[str] = []
    raw = getattr(dataset, "raw", None) or {}
    if reference is not None:
        kind = reference_kind or "emp"
        if kind not in MODEL_PATHS:
            kind = "emp"
        notes.append(f"Using caller-supplied reference ({kind} model).")
        return np.asarray(reference, dtype=complex), kind, notes

    empty = raw.get("bmid_empty_reference")
    if empty is not None:
        notes.append("Using empty-chamber reference from dataset (emp model).")
        return np.asarray(empty, dtype=complex), "emp", notes

    raise ValueError(
        "No background reference available for detection. Pass `reference=` "
        "(empty-chamber or same-phantom healthy scan) and `reference_kind` "
        "('emp', 'adi', or 'twin')."
    )


def localize_blind(dataset: MicrowaveDataset) -> dict[str, Any]:
    """Our localization path: Week 3 clutter-removed → DAS → smoothed peak."""
    meta = dataset.metadata or {}
    is_bmid = str(meta.get("dataset_family", "")).upper() == "UM-BMID"
    pre_cfg = PreprocessingConfig(
        filter_method="none" if is_bmid else "savgol",
        calibration_method="none",
        normalization_method="none",
        artifact_method="none",
        do_background_subtraction=False,
        enable_week3=True,
        enable_group_clutter_removal=True,
        enable_global_normalize=True,
    )
    if not is_bmid:
        pre_cfg.filter_kwargs = {"window_length": 7, "polyorder": 3}

    prep = run_preprocessing_pipeline(dataset, pre_cfg)
    processed = prep.processed_dataset
    week3 = week3_time_inputs(processed)
    key = preferred_week3_key(processed)
    if key is None:
        raise ValueError("Week 3 time signals missing; cannot localize.")

    config = infer_reconstruction_config(processed)
    config.n_x = max(48, config.n_x)
    config.n_y = max(48, config.n_y)
    # Week 3 clutter-removed already cleaned; do not subtract rotational mean again.
    if key == "time_clutter_removed":
        config.clutter_removal = "none"

    images = reconstruct_all_from_time(
        week3[key], processed.frequencies, config=config
    )
    image = np.asarray(images["DAS"], dtype=float)
    peak_m = smoothed_peak_centroid_m(image, config.x_span, config.y_span)
    prominence = peak_prominence_confidence(image)
    return {
        "location_m": peak_m,
        "location_cm": (peak_m[0] * 100.0, peak_m[1] * 100.0),
        "prominence": prominence,
        "trusted": prominence >= PROMINENCE_TRUST_THRESHOLD,
        "week3_key": key,
        "beamformer": "DAS",
        "x_span": config.x_span,
        "y_span": config.y_span,
    }


def detect_and_locate(
    dataset: MicrowaveDataset,
    *,
    reference: np.ndarray | None = None,
    reference_kind: str | None = None,
    always_localize: bool = False,
    bundle: dict | None = None,
) -> DetectLocateResult:
    """Run friend detection, then our localization when a tumour is flagged.

    Parameters
    ----------
    always_localize:
        If True, still mark a location even when detection says healthy
        (useful for lab scoring). Default False = product behaviour.
    """
    notes: list[str] = []
    ref, kind, ref_notes = resolve_reference(dataset, reference, reference_kind)
    notes.extend(ref_notes)

    generation = _generation(dataset)
    ant_rad = _ant_rad_cm(dataset)
    models = bundle or load_models(str(MODEL_PATHS[kind]))
    detection = predict_scan(
        np.asarray(dataset.s_parameters, dtype=complex),
        ref,
        ant_rad,
        generation,
        bundle=models,
        reference_kind=kind,
    )
    tumor = bool(detection.get("tumor_flag"))
    prob = float(detection.get("tumor_probability", 0.0))
    notes.append(
        f"Detection ({kind}): probability={prob:.3f}, flag={'tumor' if tumor else 'healthy'}."
    )

    location_cm = None
    prominence = None
    trusted = None
    loc_details: dict[str, Any] = {}
    ran = False

    if tumor or always_localize:
        if not tumor and always_localize:
            notes.append("Detection said healthy; localizing anyway (always_localize=True).")
        loc = localize_blind(dataset)
        ran = True
        location_cm = loc["location_cm"]
        prominence = float(loc["prominence"])
        trusted = bool(loc["trusted"])
        loc_details = loc
        notes.append(
            f"Localization (Week 3 DAS smoothed peak): "
            f"({location_cm[0]:.2f}, {location_cm[1]:.2f}) cm, "
            f"prominence={prominence:.2f}"
            + (" (trusted)." if trusted else " (indicative only).")
        )
    else:
        notes.append("No tumour flagged — localization skipped.")

    return DetectLocateResult(
        tumor_detected=tumor,
        tumor_probability=prob,
        detection_model=kind,
        location_cm=location_cm,
        location_prominence=prominence,
        location_trusted=trusted,
        localization_ran=ran,
        notes=notes,
        detection_details=detection,
        localization_details=loc_details,
    )

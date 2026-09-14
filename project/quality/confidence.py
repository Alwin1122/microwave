"""Module 10 — unified reconstruction confidence assessment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

_EPS = 1e-12


@dataclass
class ReconstructionConfidence:
    """Single 0–1 confidence score with interpretable components."""

    overall: float
    label: str
    components: dict[str, float]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["overall"] = float(self.overall)
        payload["components"] = {k: float(v) for k, v in self.components.items()}
        return payload


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _label_for(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _image_quality_component(quality_metrics: dict[str, Any] | None) -> float:
    if not quality_metrics:
        return 0.4
    scr = float(quality_metrics.get("scr", 0.0) or 0.0)
    contrast = float(quality_metrics.get("contrast", 0.0) or 0.0)
    ccr = float(quality_metrics.get("ccr", 0.0) or 0.0)
    # Soft saturating maps so typical BMID SCR/contrast land in a usable range.
    scr_n = 1.0 - np.exp(-max(scr, 0.0) / 8.0)
    contrast_n = _clamp01(contrast)
    ccr_n = 1.0 - np.exp(-max(ccr, 0.0) / 4.0)
    return _clamp01(0.45 * scr_n + 0.30 * contrast_n + 0.25 * ccr_n)


def _beamformer_margin_component(
    quality_metrics: dict[str, dict[str, float]] | None,
    selected_beamformer: str | None,
) -> float:
    if not quality_metrics or not selected_beamformer:
        return 0.5
    scores = []
    selected_score = None
    for name, metrics in quality_metrics.items():
        score = float(metrics.get("score", 0.0) or 0.0)
        scores.append(score)
        if name == selected_beamformer:
            selected_score = score
    if selected_score is None or not scores:
        return 0.5
    others = [s for name, s in zip(quality_metrics.keys(), scores) if name != selected_beamformer]
    if not others:
        return 0.6
    best_other = max(others)
    margin = selected_score - best_other
    # Positive margin -> higher confidence; negative -> lower.
    return _clamp01(0.55 + 0.9 * np.tanh(margin * 2.0))


def _localization_component(
    *,
    tumor_gt_distance_m: float | None,
    characterization: dict[str, float] | None,
) -> float:
    if tumor_gt_distance_m is not None and np.isfinite(tumor_gt_distance_m):
        dist_cm = float(tumor_gt_distance_m) * 100.0
        # <=1.5 cm excellent, ~3 cm moderate, >5 cm poor.
        return _clamp01(np.exp(-dist_cm / 2.5))

    if not characterization:
        return 0.45
    compactness = float(characterization.get("compactness", 0.0) or 0.0)
    diameter = float(characterization.get("equivalent_diameter_cm", 0.0) or 0.0)
    # Prefer compact foci of a few centimeters (typical tumor-like sizes).
    size_score = np.exp(-abs(diameter - 2.0) / 2.5)
    return _clamp01(0.55 * compactness + 0.45 * size_score)


def _focus_component(characterization: dict[str, float] | None) -> float:
    if not characterization:
        return 0.5
    fwhm = float(characterization.get("fwhm_cm", 0.0) or 0.0)
    area = float(characterization.get("area_cm2", 0.0) or 0.0)
    # Very large FWHM / area usually means diffuse clutter.
    fwhm_score = np.exp(-max(fwhm - 1.5, 0.0) / 3.0)
    area_score = np.exp(-max(area - 4.0, 0.0) / 6.0)
    return _clamp01(0.6 * fwhm_score + 0.4 * area_score)


def assess_reconstruction_confidence(
    *,
    selected_quality: dict[str, Any] | None,
    all_quality_metrics: dict[str, dict[str, float]] | None,
    selected_beamformer: str | None,
    tumor_candidate: dict[str, Any] | None,
    characterization: dict[str, float] | None,
    tumor_gt_distance_m: float | None = None,
) -> ReconstructionConfidence:
    """
    Combine image quality, detection evidence, beamformer margin, localization,
    and focus sharpness into one interpretable confidence score.
    """
    image_q = _image_quality_component(selected_quality)
    detection = _clamp01(float((tumor_candidate or {}).get("confidence", 0.35) or 0.35))
    margin = _beamformer_margin_component(all_quality_metrics, selected_beamformer)
    localization = _localization_component(
        tumor_gt_distance_m=tumor_gt_distance_m,
        characterization=characterization,
    )
    focus = _focus_component(characterization)

    # Weighted blend — detection and localization dominate decision usefulness.
    overall = _clamp01(
        0.22 * image_q
        + 0.28 * detection
        + 0.15 * margin
        + 0.22 * localization
        + 0.13 * focus
    )

    reasons: list[str] = []
    if detection >= 0.6:
        reasons.append("Tumor-candidate features support a suspicious focus.")
    elif detection < 0.35:
        reasons.append("Tumor-candidate evidence is weak or clutter-like.")

    if tumor_gt_distance_m is not None:
        dist_cm = float(tumor_gt_distance_m) * 100.0
        if dist_cm <= 1.5:
            reasons.append(f"ROI is close to tumor GT ({dist_cm:.2f} cm).")
        elif dist_cm > 3.0:
            reasons.append(f"ROI is far from tumor GT ({dist_cm:.2f} cm).")

    if image_q >= 0.65:
        reasons.append("Selected reconstruction has solid SCR/contrast.")
    elif image_q < 0.35:
        reasons.append("Selected reconstruction contrast/SCR is low.")

    if margin >= 0.7:
        reasons.append("Selected beamformer clearly outscores alternatives.")
    elif margin < 0.4:
        reasons.append("Beamformer scores are close — selection is less decisive.")

    if focus < 0.4:
        reasons.append("Focus is diffuse (large FWHM/area), lowering confidence.")

    if not reasons:
        reasons.append("Mixed evidence across quality, detection, and localization.")

    return ReconstructionConfidence(
        overall=overall,
        label=_label_for(overall),
        components={
            "image_quality": image_q,
            "detection": detection,
            "beamformer_margin": margin,
            "localization": localization,
            "focus": focus,
        },
        reasons=reasons,
    )

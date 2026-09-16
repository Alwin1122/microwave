"""Threshold-based quality policy presets for auto validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ThresholdProfile:
    name: str
    require_finite: bool = True
    min_energy: float = 1e-12
    min_roi_area: int = 4
    max_localization_error_cm: float | None = 3.0
    min_scr: float | None = 2.0
    min_snr: float | None = 0.5
    min_confidence: float | None = 0.35
    require_tumor_candidate_if_gt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROFILES: dict[str, ThresholdProfile] = {
    "strict": ThresholdProfile(
        name="strict",
        min_energy=1e-10,
        min_roi_area=8,
        max_localization_error_cm=1.5,
        min_scr=5.0,
        min_snr=1.0,
        min_confidence=0.55,
        require_tumor_candidate_if_gt=True,
    ),
    "balanced": ThresholdProfile(
        name="balanced",
        min_energy=1e-12,
        min_roi_area=4,
        max_localization_error_cm=3.0,
        min_scr=2.0,
        min_snr=0.5,
        min_confidence=0.35,
        require_tumor_candidate_if_gt=False,
    ),
    "lenient": ThresholdProfile(
        name="lenient",
        min_energy=1e-14,
        min_roi_area=1,
        max_localization_error_cm=5.0,
        min_scr=0.5,
        min_snr=0.1,
        min_confidence=0.2,
        require_tumor_candidate_if_gt=False,
    ),
}


def get_threshold_profile(name: str = "balanced") -> ThresholdProfile:
    key = (name or "balanced").strip().lower()
    if key not in _PROFILES:
        raise ValueError(
            f"Unknown threshold profile '{name}'. "
            f"Choose: {', '.join(sorted(_PROFILES))}."
        )
    return _PROFILES[key]


def list_threshold_profiles() -> list[str]:
    return sorted(_PROFILES)

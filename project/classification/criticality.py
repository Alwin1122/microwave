"""Research criticality from tumour volume and margin. Not a clinical BI-RADS.

UM-BMID has no benign-versus-malignant histology. The metadata field named
BI-RADS is breast density. Criticality here is a phantom label built from the
two facts the dataset does record: diameter and shape.

Under the Born approximation the scattered field of a small dielectric sphere
grows with volume, so the size part of the index is

    η = (d / d_ref)³ ,    d_ref = 2 cm

A sphere with d = 2 cm has η = 1. A sphere with d = 4 cm has η = 8.
Spiculated margins are the morphological feature associated with higher
suspicion in the microwave-characterisation literature (Davis et al.,
Conceição et al.). They scale the index by 1.5. That moves a 2 cm spiculated
inclusion from the low bin into the moderate bin, and leaves a 1 cm
spiculated inclusion in the low bin.

    low        η ≤ 1
    moderate   1 < η ≤ 8
    high       η > 8

Healthy scans have no index. This label is for research comparison only.
"""

from __future__ import annotations

import math

REFERENCE_DIAMETER_CM = 2.0
SPICULATED_FACTOR = 1.5
MODERATE_MAX = 8.0  # (4 cm / 2 cm)³ for a sphere

CRITICALITY_CLASSES = ("low", "moderate", "high")
MORPHOLOGY_TYPES = ("spherical", "spiculated")


def morphology_type(shape: str | None) -> str | None:
    """Map a metadata shape string onto spherical or spiculated."""
    text = (shape or "").strip().lower()
    if not text:
        return None
    if "spic" in text:
        return "spiculated"
    if text in {"sphere", "spherical", "smooth", "round", "oval"}:
        return "spherical"
    return None


def criticality_index(diameter_cm: float | None, shape: str | None = None) -> float | None:
    """Born volume index, raised when the recorded margin is spiculated."""
    if diameter_cm is None:
        return None
    try:
        diameter = float(diameter_cm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(diameter) or diameter <= 0:
        return None
    factor = SPICULATED_FACTOR if morphology_type(shape) == "spiculated" else 1.0
    return float((diameter / REFERENCE_DIAMETER_CM) ** 3 * factor)


def criticality_class(index: float | None) -> str | None:
    """Bin the volume index. See the module docstring for the cut points."""
    if index is None:
        return None
    value = float(index)
    if not math.isfinite(value):
        return None
    if value <= 1.0:
        return "low"
    if value <= MODERATE_MAX:
        return "moderate"
    return "high"


def criticality_from_scan(diameter_cm: float | None, shape: str | None = None) -> tuple[float | None, str | None]:
    index = criticality_index(diameter_cm, shape)
    return index, criticality_class(index)

"""Research/educational tumor-type labels from UM-BMID metadata.

This is a classification *base* only: buckets for presence, size, BIRADS-based
severity, shape, and location. A trained classifier is future work.

Not a clinical diagnosis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SIZE_SMALL_CM = 2.0
SIZE_MEDIUM_CM = 4.0


@dataclass(frozen=True)
class TumorTaxonomy:
    presence: str  # "healthy" | "tumor"
    size_class: str | None = None  # small / medium / large
    birads: int | None = None
    severity_class: str | None = None  # low / moderate / high (placeholder)
    shape: str | None = None
    location_quadrant: str | None = None
    source: str = "metadata"

    def short_label(self) -> str:
        if self.presence != "tumor":
            return "healthy"
        parts = [
            part
            for part in (
                self.size_class,
                self.severity_class,
                self.shape,
                self.location_quadrant,
            )
            if part
        ]
        return "/".join(parts) or "tumor"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["short_label"] = self.short_label()
        return payload

    def to_display_dict(self) -> dict[str, str]:
        return {
            "Presence": self.presence,
            "Size class": self.size_class or "n/a",
            "BIRADS": "n/a" if self.birads is None else str(self.birads),
            "Severity (placeholder)": self.severity_class or "n/a",
            "Shape": self.shape or "n/a",
            "Quadrant": self.location_quadrant or "n/a",
        }


def size_class_from_diameter_cm(diam_cm: float | None) -> str | None:
    if diam_cm is None or diam_cm <= 0:
        return None
    if diam_cm <= SIZE_SMALL_CM:
        return "small"
    if diam_cm <= SIZE_MEDIUM_CM:
        return "medium"
    return "large"


def severity_from_birads_and_size(
    birads: int | None, size_class: str | None
) -> str | None:
    """Placeholder mapping for later classification — not clinical.

    Size is the primary seriousness cue in this phantom set. BIRADS 3+ can
    raise the bucket; BIRADS 1–2 does not force a 6 cm tumor to "low".
    """
    size_map = {"small": "low", "medium": "moderate", "large": "high"}
    rank = {"low": 0, "moderate": 1, "high": 2}
    severity = size_map.get(size_class or "")
    if birads is None:
        return severity
    if birads >= 4:
        birads_sev = "high"
    elif birads == 3:
        birads_sev = "moderate"
    else:
        birads_sev = None
    if severity is None:
        return birads_sev
    if birads_sev is None:
        return severity
    return severity if rank[severity] >= rank[birads_sev] else birads_sev


def location_quadrant(
    x_cm: float | None, y_cm: float | None, *, center_cm: float = 1.0
) -> str | None:
    if x_cm is None or y_cm is None:
        return None
    if (x_cm**2 + y_cm**2) ** 0.5 < center_cm:
        return "central"
    ns = "upper" if y_cm >= 0 else "lower"
    ew = "right" if x_cm >= 0 else "left"
    return f"{ns}-{ew}"


def classify_scan(
    *,
    has_tumor: bool,
    tum_diam_cm: float | None = None,
    tum_x_cm: float | None = None,
    tum_y_cm: float | None = None,
    birads: int | None = None,
    tum_shape: str | None = None,
) -> TumorTaxonomy:
    if not has_tumor:
        return TumorTaxonomy(presence="healthy", birads=birads)
    size = size_class_from_diameter_cm(tum_diam_cm)
    shape = (tum_shape or "").strip().lower() or None
    return TumorTaxonomy(
        presence="tumor",
        size_class=size,
        birads=birads,
        severity_class=severity_from_birads_and_size(birads, size),
        shape=shape,
        location_quadrant=location_quadrant(tum_x_cm, tum_y_cm),
    )


def taxonomy_from_metadata(metadata: dict | None) -> TumorTaxonomy:
    meta = metadata or {}
    diam_m = meta.get("tumor_diameter_m")
    x_m = meta.get("tumor_x_m")
    y_m = meta.get("tumor_y_m")
    return classify_scan(
        has_tumor=bool(meta.get("bmid_has_tumor")),
        tum_diam_cm=None if diam_m is None else float(diam_m) * 100.0,
        tum_x_cm=None if x_m is None else float(x_m) * 100.0,
        tum_y_cm=None if y_m is None else float(y_m) * 100.0,
        birads=meta.get("bmid_birads"),
        tum_shape=meta.get("bmid_tum_shape"),
    )


def taxonomy_from_scan(scan) -> TumorTaxonomy:
    return classify_scan(
        has_tumor=bool(getattr(scan, "has_tumor", False)),
        tum_diam_cm=getattr(scan, "tum_diam_cm", None),
        tum_x_cm=getattr(scan, "tum_x_cm", None),
        tum_y_cm=getattr(scan, "tum_y_cm", None),
        birads=getattr(scan, "birads", None),
        tum_shape=getattr(scan, "tum_shape", None),
    )

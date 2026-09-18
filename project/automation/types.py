"""Shared types for automated validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DatasetVerdict(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    BLOCKED = "blocked"


class FileKind(str, Enum):
    MEASUREMENT = "measurement"
    METADATA = "metadata"
    UNKNOWN = "unknown"


@dataclass
class DiscoveredFile:
    path: str
    kind: FileKind
    basename: str
    reason: str = ""


@dataclass
class ResolvedTarget:
    measurement_path: str
    metadata_path: str | None = None
    scan_index: int | None = None
    scan_label: str | None = None
    source_note: str = ""


@dataclass
class MetricEvidence:
    name: str
    value: float | None
    threshold: float | None = None
    comparison: str | None = None
    ok: bool | None = None


@dataclass
class DatasetValidationResult:
    target: ResolvedTarget
    status: DatasetVerdict
    reasons: list[str] = field(default_factory=list)
    remediation: list[str] = field(default_factory=list)
    metrics: list[MetricEvidence] = field(default_factory=list)
    recommended_upload: bool = False
    cached: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["target"]["measurement_path"] = self.target.measurement_path
        return payload


@dataclass
class ValidationBatchResult:
    profile: str
    roots: list[str]
    results: list[DatasetValidationResult] = field(default_factory=list)
    report_paths: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    message: str = ""
    tumor_index: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "roots": list(self.roots),
            "results": [r.to_dict() for r in self.results],
            "report_paths": dict(self.report_paths),
            "counts": dict(self.counts),
            "message": self.message,
            "tumor_index": list(self.tumor_index or []),
        }

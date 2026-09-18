"""Resolve metadata files to measurement pairs and BMID scan targets."""

from __future__ import annotations

import os
import re
from pathlib import Path

from automation.types import DiscoveredFile, FileKind, ResolvedTarget
from data_loader.bmid_loader import (
    is_bmid_fd_filename,
    is_likely_metadata_file,
    list_bmid_scans,
    resolve_measurement_from_metadata_path,
)

_MD_TO_FD = re.compile(r"^md_list_(s11|s21)(?:_(adi|emp))?\.mat$", re.IGNORECASE)


def _looks_like_bmid_fd(path: str) -> bool:
    """Accept classic BMID names plus generic fd_data_* files."""
    base = os.path.basename(path).lower()
    return is_bmid_fd_filename(path) or base.startswith("fd_data_")


def metadata_to_measurement_candidate(metadata_path: str) -> str | None:
    """Resolve metadata paths to likely paired measurement files."""
    direct = resolve_measurement_from_metadata_path(metadata_path)
    if direct:
        return direct

    if not is_likely_metadata_file(metadata_path):
        return None

    # Fallback for canonical md_list_* naming in same folder.
    base = os.path.basename(metadata_path)
    match = _MD_TO_FD.match(base)
    if not match:
        return None
    sparam = match.group(1).lower()
    cal = match.group(2)
    name = f"fd_data_{sparam}_{cal}.mat" if cal else f"fd_data_{sparam}.mat"
    candidate = os.path.join(os.path.dirname(metadata_path), name)
    return candidate if os.path.isfile(candidate) else None


def resolve_measurement_targets(
    files: list[DiscoveredFile],
    *,
    bmid_strategy: str = "diverse",
    max_bmid_scans: int = 8,
) -> list[ResolvedTarget]:
    """Build unique measurement targets from discovered files.

    Assumption: metadata-only files are useful only when a paired
    ``fd_data_*.mat`` exists beside them.
    """
    targets: list[ResolvedTarget] = []
    seen: set[tuple[str, int | None]] = set()

    for item in files:
        if item.kind != FileKind.MEASUREMENT:
            continue
        _extend_targets(
            targets,
            seen,
            item.path,
            metadata_path=None,
            bmid_strategy=bmid_strategy,
            max_bmid_scans=max_bmid_scans,
            source_note=item.reason,
        )

    for item in files:
        if item.kind != FileKind.METADATA:
            continue
        paired = metadata_to_measurement_candidate(item.path)
        if not paired:
            continue
        key_path = str(Path(paired).resolve())
        if any(str(Path(t.measurement_path).resolve()) == key_path for t in targets):
            continue
        _extend_targets(
            targets,
            seen,
            paired,
            metadata_path=item.path,
            bmid_strategy=bmid_strategy,
            max_bmid_scans=max_bmid_scans,
            source_note=f"resolved from metadata {item.basename}",
        )
    return targets


def _extend_targets(
    targets: list[ResolvedTarget],
    seen: set[tuple[str, int | None]],
    measurement_path: str,
    *,
    metadata_path: str | None,
    bmid_strategy: str,
    max_bmid_scans: int,
    source_note: str,
) -> None:
    path = str(Path(measurement_path).resolve())
    if _looks_like_bmid_fd(path):
        try:
            scans = list_bmid_scans(path)
        except Exception:
            scans = []
        chosen = _choose_bmid_scans(
            scans, bmid_strategy=bmid_strategy, max_scans=max_bmid_scans
        )
        if chosen:
            for scan in chosen:
                key = (path, scan.index)
                if key in seen:
                    continue
                seen.add(key)
                targets.append(
                    ResolvedTarget(
                        measurement_path=path,
                        metadata_path=metadata_path,
                        scan_index=scan.index,
                        scan_label=scan.label,
                        source_note=source_note,
                    )
                )
            return

    key = (path, None)
    if key in seen:
        return
    seen.add(key)
    targets.append(
        ResolvedTarget(
            measurement_path=path,
            metadata_path=metadata_path,
            scan_index=None,
            scan_label=None,
            source_note=source_note,
        )
    )


def _choose_bmid_scans(scans, *, bmid_strategy: str, max_scans: int):
    if not scans:
        return []
    strategy = (bmid_strategy or "tumor_and_healthy").lower()
    unlimited = int(max_scans) <= 0
    max_scans = len(scans) if unlimited else max(1, int(max_scans))
    tumors = [s for s in scans if s.has_tumor]
    healthy = [s for s in scans if not s.has_tumor]

    def _tumor_priority(scan) -> tuple:
        diam = float(scan.tum_diam_cm or 0.0)
        off = abs(float(scan.tum_x_cm or 0.0)) + abs(float(scan.tum_y_cm or 0.0))
        # Prefer large labeled tumors that are off-center but not on the FOV edge.
        offset_pref = -abs(off - 3.0)
        return (-diam, offset_pref, scan.index)

    tumors_ranked = sorted(tumors, key=_tumor_priority)

    if strategy == "all":
        ordered = tumors_ranked + healthy
        return ordered[:max_scans]
    if strategy == "diverse":
        return _choose_diverse_scans(scans, max_scans=max_scans)
    if strategy == "first_healthy":
        return (healthy or scans)[:max_scans]
    if strategy == "first":
        return scans[:max_scans]
    if strategy in {"tumor_and_healthy", "first_tumor_else_first"}:
        chosen = []
        if tumors_ranked:
            chosen.append(tumors_ranked[0])
        elif scans:
            chosen.append(scans[0])
        if max_scans >= 2 and healthy:
            chosen.append(healthy[0])
        elif max_scans >= 2 and len(tumors_ranked) > 1:
            chosen.append(tumors_ranked[1])
        return chosen[:max_scans]
    return (tumors_ranked or scans)[:max_scans]


def _choose_diverse_scans(scans, *, max_scans: int):
    """Pick a small mix of healthy + tumor size/severity/shape/location buckets."""
    from quality.tumor_taxonomy import taxonomy_from_scan

    chosen: list = []
    used: set[int] = set()

    def take(predicate) -> bool:
        for scan in scans:
            if scan.index in used:
                continue
            if predicate(scan):
                chosen.append(scan)
                used.add(scan.index)
                return True
        return False

    take(lambda scan: not scan.has_tumor)
    for size in ("small", "medium", "large"):
        take(
            lambda scan, size=size: scan.has_tumor
            and taxonomy_from_scan(scan).size_class == size
        )

    seen_quads = {
        taxonomy_from_scan(scan).location_quadrant
        for scan in chosen
        if taxonomy_from_scan(scan).location_quadrant
    }
    for scan in scans:
        if len(chosen) >= max_scans:
            break
        if scan.index in used or not scan.has_tumor:
            continue
        quad = taxonomy_from_scan(scan).location_quadrant
        if quad and quad not in seen_quads:
            chosen.append(scan)
            used.add(scan.index)
            seen_quads.add(quad)

    seen_birads = {taxonomy_from_scan(scan).birads for scan in chosen}
    for scan in scans:
        if len(chosen) >= max_scans:
            break
        if scan.index in used or not scan.has_tumor:
            continue
        birads = taxonomy_from_scan(scan).birads
        if birads is not None and birads not in seen_birads:
            chosen.append(scan)
            used.add(scan.index)
            seen_birads.add(birads)

    seen_shapes = {
        taxonomy_from_scan(scan).shape
        for scan in chosen
        if taxonomy_from_scan(scan).shape
    }
    for scan in scans:
        if len(chosen) >= max_scans:
            break
        if scan.index in used or not scan.has_tumor:
            continue
        shape = taxonomy_from_scan(scan).shape
        if shape and shape not in seen_shapes:
            chosen.append(scan)
            used.add(scan.index)
            seen_shapes.add(shape)

    for scan in scans:
        if len(chosen) >= max_scans:
            break
        if scan.index not in used:
            chosen.append(scan)
            used.add(scan.index)
    return chosen[:max_scans]


def resolve_roots_to_targets(
    roots: list[str],
    *,
    max_files: int = 50,
    bmid_strategy: str = "diverse",
    max_bmid_scans: int = 8,
) -> list[ResolvedTarget]:
    from automation.discovery import discover_files

    files = discover_files(roots, max_files=max_files)
    return resolve_measurement_targets(
        files, bmid_strategy=bmid_strategy, max_bmid_scans=max_bmid_scans
    )


def build_tumor_index(measurement_paths: list[str]) -> list[dict]:
    """List labeled tumor/healthy scans from BMID metadata without reconstructing."""
    from data_loader.bmid_loader import list_bmid_scans

    index: list[dict] = []
    seen: set[str] = set()
    for raw in measurement_paths:
        path = str(Path(raw).resolve()) if Path(raw).exists() else raw
        if path in seen:
            continue
        seen.add(path)
        if not _looks_like_bmid_fd(path):
            continue
        try:
            scans = list_bmid_scans(path)
        except Exception:
            continue
        tumors = [s for s in scans if s.has_tumor]
        healthy = [s for s in scans if not s.has_tumor]
        examples = []
        for scan in tumors[:8]:
            examples.append(
                {
                    "index": scan.index,
                    "label": scan.label,
                    "tum_diam_cm": scan.tum_diam_cm,
                    "tum_x_cm": scan.tum_x_cm,
                    "tum_y_cm": scan.tum_y_cm,
                }
            )
        index.append(
            {
                "measurement_path": path,
                "file_name": Path(path).name,
                "n_scans": len(scans),
                "n_tumor": len(tumors),
                "n_healthy": len(healthy),
                "tumor_indices": [s.index for s in tumors],
                "examples": examples,
            }
        )
    return index

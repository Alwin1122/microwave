"""Resolve metadata files to measurement pairs and BMID scan targets."""

from __future__ import annotations

import os
import re
from pathlib import Path

from automation.types import DiscoveredFile, FileKind, ResolvedTarget
from data_loader.bmid_loader import is_bmid_fd_filename, list_bmid_scans

_MD_TO_FD = re.compile(r"^md_list_(s11|s21)(?:_(adi|emp))?\.mat$", re.IGNORECASE)


def metadata_to_measurement_candidate(metadata_path: str) -> str | None:
    """Map md_list_s21_adi.mat -> fd_data_s21_adi.mat when present."""
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
    bmid_strategy: str = "first_tumor_else_first",
    max_bmid_scans: int = 1,
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
    if is_bmid_fd_filename(path):
        scans = list_bmid_scans(path)
        chosen = _choose_bmid_scans(
            scans, bmid_strategy=bmid_strategy, max_scans=max_bmid_scans
        )
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
    strategy = (bmid_strategy or "first_tumor_else_first").lower()
    max_scans = max(1, int(max_scans))
    if strategy == "all":
        return scans[:max_scans]
    if strategy == "first_healthy":
        healthy = [s for s in scans if not s.has_tumor]
        return (healthy or scans)[:max_scans]
    if strategy == "first":
        return scans[:max_scans]
    tumors = [s for s in scans if s.has_tumor]
    return (tumors or scans)[:max_scans]


def resolve_roots_to_targets(
    roots: list[str],
    *,
    max_files: int = 50,
    bmid_strategy: str = "first_tumor_else_first",
    max_bmid_scans: int = 1,
) -> list[ResolvedTarget]:
    from automation.discovery import discover_files

    files = discover_files(roots, max_files=max_files)
    return resolve_measurement_targets(
        files, bmid_strategy=bmid_strategy, max_bmid_scans=max_bmid_scans
    )

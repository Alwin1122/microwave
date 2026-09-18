"""Discover measurement and metadata files under configurable roots."""

from __future__ import annotations

import os
import re
from pathlib import Path

from automation.types import DiscoveredFile, FileKind
from data_loader.validator import SUPPORTED_EXTENSIONS

_MD_RE = re.compile(r"^(md_list_|metadata_).*\.mat$", re.IGNORECASE)
_FD_RE = re.compile(r"^fd_data_.*\.mat$", re.IGNORECASE)


def classify_path(path: str) -> DiscoveredFile:
    base = os.path.basename(path)
    ext = os.path.splitext(base)[1].lower()
    if _MD_RE.match(base):
        return DiscoveredFile(
            path=path,
            kind=FileKind.METADATA,
            basename=base,
            reason="UM-BMID metadata list",
        )
    if ext in SUPPORTED_EXTENSIONS:
        reason = "Touchstone measurement" if ext.startswith(".s") else "MATLAB measurement"
        if _FD_RE.match(base):
            reason = "UM-BMID frequency-domain cube"
        return DiscoveredFile(
            path=path,
            kind=FileKind.MEASUREMENT,
            basename=base,
            reason=reason,
        )
    return DiscoveredFile(
        path=path,
        kind=FileKind.UNKNOWN,
        basename=base,
        reason=f"unsupported extension {ext or '(none)'}",
    )


def discover_files(roots: list[str], max_files: int = 50) -> list[DiscoveredFile]:
    found: list[DiscoveredFile] = []
    seen: set[str] = set()
    for root in roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        candidates = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            item = classify_path(resolved)
            if item.kind == FileKind.UNKNOWN:
                continue
            seen.add(resolved)
            found.append(item)

    def _rank(item: DiscoveredFile) -> tuple[int, str]:
        name = item.basename.lower()
        if name.startswith("fd_data_"):
            return (0, name)
        if item.kind == FileKind.MEASUREMENT:
            return (1, name)
        return (2, name)

    found.sort(key=_rank)
    return found[: max(1, int(max_files))]

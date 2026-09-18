"""Disk cache for duplicate auto-validation runs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


class ValidationCache:
    """JSON file cache keyed by file identity + validation options."""

    def __init__(self, cache_dir: str | None = None) -> None:
        root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent.parent / "results" / ".auto_validation_cache"
        self.cache_dir = root
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(
        self,
        measurement_path: str,
        scan_index: int | None,
        profile: str,
        include_reconstruction_checks: bool,
    ) -> str:
        st = os.stat(measurement_path)
        raw = "|".join(
            [
                str(Path(measurement_path).resolve()),
                str(scan_index),
                profile,
                str(bool(include_reconstruction_checks)),
                "detect-v8",
                str(int(st.st_mtime)),
                str(int(st.st_size)),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        measurement_path: str,
        scan_index: int | None,
        profile: str,
        include_reconstruction_checks: bool,
    ) -> dict[str, Any] | None:
        key = self._key(measurement_path, scan_index, profile, include_reconstruction_checks)
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(
        self,
        measurement_path: str,
        scan_index: int | None,
        profile: str,
        include_reconstruction_checks: bool,
        payload: dict[str, Any],
    ) -> None:
        key = self._key(measurement_path, scan_index, profile, include_reconstruction_checks)
        path = self.cache_dir / f"{key}.json"
        envelope = {"saved_at": time.time(), "payload": payload}
        path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")

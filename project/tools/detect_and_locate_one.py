"""CLI: detect tumour (friend model) then mark location (our imaging).

Examples:
  python tools/detect_and_locate_one.py datasets/fd_data_s21_adi.mat --scan 0
  python tools/detect_and_locate_one.py path/to/cube.mat --scan 12 --kind twin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from automation.detect_locate import detect_and_locate  # noqa: E402
from data_loader.bmid_loader import list_bmid_scans, load_bmid_scan  # noqa: E402
from data_loader.loader import load_dataset  # noqa: E402


def _healthy_twin_reference(fd_path: str, scan_index: int) -> np.ndarray | None:
    """Mean of healthy scans of the same phantom (follow-up style reference)."""
    scans = list_bmid_scans(fd_path)
    target = next((s for s in scans if s.index == scan_index), None)
    if target is None:
        return None
    healthy = [s for s in scans if s.phant_id == target.phant_id and not s.has_tumor]
    if not healthy:
        return None
    stack = [
        np.asarray(load_bmid_scan(fd_path, s.index).s_parameters, dtype=complex)
        for s in healthy
    ]
    return np.mean(np.stack(stack, axis=0), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=str, help="BMID .mat cube or single dataset file")
    parser.add_argument("--scan", type=int, default=0, help="BMID scan index")
    parser.add_argument(
        "--kind",
        choices=("emp", "adi", "twin", "auto"),
        default="auto",
        help="Detection reference/model (auto tries emp then twin)",
    )
    parser.add_argument(
        "--always-localize",
        action="store_true",
        help="Mark a location even if detection says healthy",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    path = str(Path(args.path))
    try:
        dataset = load_dataset(path, scan_index=args.scan)
    except TypeError:
        dataset = load_bmid_scan(path, args.scan)

    reference = None
    kind = None if args.kind == "auto" else args.kind
    notes_extra: list[str] = []

    raw = getattr(dataset, "raw", None) or {}
    if args.kind in ("auto", "emp") and raw.get("bmid_empty_reference") is None:
        if args.kind == "emp":
            print("ERROR: no empty reference in this cube; try --kind twin", file=sys.stderr)
            return 2
        twin = _healthy_twin_reference(path, args.scan)
        if twin is not None:
            reference = twin
            kind = "twin"
            notes_extra.append("auto: no emp reference → using same-phantom healthy mean (twin).")
        else:
            print(
                "ERROR: no emp or twin reference available for detection.",
                file=sys.stderr,
            )
            return 2
    elif args.kind == "twin":
        reference = _healthy_twin_reference(path, args.scan)
        if reference is None:
            print("ERROR: no healthy twin scans for this phantom.", file=sys.stderr)
            return 2
        kind = "twin"

    result = detect_and_locate(
        dataset,
        reference=reference,
        reference_kind=kind,
        always_localize=args.always_localize,
    )
    result.notes = notes_extra + result.notes

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0

    print("=== Detect → Locate ===")
    for note in result.notes:
        print(f"- {note}")
    print(f"Tumor detected: {result.tumor_detected}  (p={result.tumor_probability:.3f})")
    if result.localization_ran and result.location_cm is not None:
        print(
            f"Location: ({result.location_cm[0]:.2f}, {result.location_cm[1]:.2f}) cm"
            f"  prominence={result.location_prominence:.2f}"
            f"  trusted={result.location_trusted}"
        )
    else:
        print("Location: (skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI entry for automated validation.

Examples:
  python tools/run_auto_validation.py --profile balanced
  python tools/run_auto_validation.py --bmid-strategy all --max-bmid-scans 0 --max-files 20
  python tools/run_auto_validation.py --roots datasets --no-reconstruction
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from automation.service import AutoValidationService, default_validation_roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run microwave auto validation")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help="Files or directories to scan (default: umbmid/simple-clean only)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Max measurement cubes (default: 1 for periodic testing)",
    )
    parser.add_argument(
        "--profile",
        choices=["strict", "balanced", "lenient"],
        default="balanced",
    )
    parser.add_argument(
        "--bmid-strategy",
        choices=[
            "diverse",
            "tumor_and_healthy",
            "first_tumor_else_first",
            "first",
            "first_healthy",
            "all",
        ],
        default="diverse",
        help="Which BMID scans to reconstruct (default: diverse subset for periodic tests)",
    )
    parser.add_argument(
        "--max-bmid-scans",
        type=int,
        default=8,
        help="Max scans per cube; 0 means every scan (use with --bmid-strategy all)",
    )
    parser.add_argument(
        "--no-reconstruction",
        action="store_true",
        help="Load/energy checks only (faster)",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "results"),
    )
    args = parser.parse_args(argv)

    service = AutoValidationService(output_dir=args.output_dir)
    roots = args.roots or default_validation_roots()

    def progress(pct: int, msg: str) -> None:
        print(f"[{pct:3d}%] {msg}")

    batch = service.run(
        roots,
        max_files=args.max_files,
        threshold_profile=args.profile,
        include_reconstruction_checks=not args.no_reconstruction,
        bmid_strategy=args.bmid_strategy,
        max_bmid_scans=args.max_bmid_scans,
        use_cache=not args.no_cache,
        progress=progress,
    )
    print(batch.message)
    print(json.dumps(batch.counts, indent=2))
    n_yes = sum(
        1
        for item in batch.results
        if ((item.details or {}).get("tumor_candidate") or {}).get("is_tumor_candidate")
    )
    print(f"tumor-candidate Yes: {n_yes}")
    print("Reports:")
    for key, path in batch.report_paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

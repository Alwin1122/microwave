"""CLI entry for automated validation.

Examples:
  python tools/run_auto_validation.py --roots datasets --profile balanced --max-files 5
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

from automation.service import AutoValidationService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run microwave auto validation")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[os.path.join(ROOT, "datasets")],
        help="Files or directories to scan",
    )
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument(
        "--profile",
        choices=["strict", "balanced", "lenient"],
        default="balanced",
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

    def progress(pct: int, msg: str) -> None:
        print(f"[{pct:3d}%] {msg}")

    batch = service.run(
        args.roots,
        max_files=args.max_files,
        threshold_profile=args.profile,
        include_reconstruction_checks=not args.no_reconstruction,
        use_cache=not args.no_cache,
        progress=progress,
    )
    print(batch.message)
    print(json.dumps(batch.counts, indent=2))
    print("Reports:")
    for key, path in batch.report_paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

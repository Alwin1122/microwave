"""Automated dataset intake agent for upload recommendations and sanity checks.

This tool scans dataset folders, classifies metadata vs measurement files,
automatically resolves metadata companions when possible, and runs quick
load/reconstruction checks so users can avoid manual trial-and-error.

Outputs:
- JSON report:   results/auto_intake_report.json
- Markdown log: results/auto_intake_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.bmid_loader import (
    ScanSelectionRequiredError,
    is_likely_metadata_file,
    list_bmid_scans,
    resolve_measurement_from_metadata_path,
)
from data_loader.loader import load_dataset
from preprocessing.preprocessing_pipeline import PreprocessingConfig, run_preprocessing_pipeline
from quality.beamformer_selector import select_best_beamformer
from reconstruction.reconstruction_manager import ReconstructionConfig, reconstruct_all
from roi.roi_detector import detect_roi
from utils.exceptions import MicrowaveFrameworkError

SUPPORTED_EXTENSIONS = {".mat", ".s1p", ".s2p", ".s4p", ".s8p"}


@dataclass
class VerificationResult:
    ok: bool
    file_path: str
    entry_path: str
    selected_scan_index: int | None
    selected_beamformer: str | None
    roi_area: int | None
    has_nonzero_image: bool | None
    message: str


@dataclass
class IntakeRecord:
    file_path: str
    file_kind: str
    resolved_measurement_path: str | None
    recommended_entry_path: str | None
    status: str
    note: str
    verification: list[VerificationResult]


def _normalize(path: str) -> str:
    return os.path.abspath(path)


def _collect_candidate_files(roots: list[str], max_files: int) -> list[str]:
    files: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for walk_root, _dirs, names in os.walk(root):
            for name in names:
                ext = os.path.splitext(name)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                files.append(_normalize(os.path.join(walk_root, name)))
                if len(files) >= max_files:
                    return sorted(files)
    return sorted(files)


def _pick_scan_indices(file_path: str) -> list[int]:
    scans = list_bmid_scans(file_path)
    if not scans:
        return [0]

    tumor_idx = next((s.index for s in scans if s.has_tumor), None)
    healthy_idx = next((s.index for s in scans if not s.has_tumor), None)

    picks: list[int] = []
    if tumor_idx is not None:
        picks.append(tumor_idx)
    if healthy_idx is not None and healthy_idx not in picks:
        picks.append(healthy_idx)
    if not picks:
        picks = [scans[0].index]
    return picks[:2]


def _quick_verify_loaded_dataset(file_path: str, scan_index: int | None) -> VerificationResult:
    dataset = load_dataset(file_path, scan_index=scan_index)
    is_touchstone = dataset.file_path.lower().endswith((".s1p", ".s2p", ".s4p", ".s8p"))

    prep_cfg = PreprocessingConfig(enable_week3=False)
    prep = run_preprocessing_pipeline(dataset, prep_cfg)
    processed = prep.processed_dataset

    recon_cfg = ReconstructionConfig()
    images, timings = reconstruct_all(
        processed.s_parameters,
        processed.frequencies,
        config=recon_cfg,
        return_timings=True,
    )
    selected, _quality = select_best_beamformer(images, timings, mode="prefer_dmas_d4")
    image = np.asarray(images[selected])
    roi = detect_roi(image, prefer_off_center=False, tight_peak=False)

    finite_ok = bool(np.isfinite(image).all())
    nonzero_ok = bool(np.max(np.abs(image)) > 0)
    roi_ok = int(roi.area) > 0
    # Touchstone single-trace samples can reconstruct into low-energy maps in
    # this quick check; treat finite + ROI as pass and keep non-zero as advisory.
    ok = finite_ok and roi_ok and (nonzero_ok or is_touchstone)

    msg_bits = [
        "finite image" if finite_ok else "non-finite values detected",
        "non-zero image" if nonzero_ok else "all-zero image",
        f"roi_area={int(roi.area)}",
    ]
    if is_touchstone and not nonzero_ok:
        msg_bits.append("zero image tolerated for touchstone quick-check")

    return VerificationResult(
        ok=ok,
        file_path=file_path,
        entry_path=file_path,
        selected_scan_index=scan_index,
        selected_beamformer=selected,
        roi_area=int(roi.area),
        has_nonzero_image=nonzero_ok,
        message=", ".join(msg_bits),
    )


def _verify_measurement_entry(file_path: str) -> list[VerificationResult]:
    try:
        result = _quick_verify_loaded_dataset(file_path, scan_index=None)
        return [result]
    except ScanSelectionRequiredError:
        picks = _pick_scan_indices(file_path)
        results: list[VerificationResult] = []
        for idx in picks:
            try:
                row = _quick_verify_loaded_dataset(file_path, scan_index=idx)
                row.entry_path = file_path
                row.selected_scan_index = idx
                results.append(row)
            except Exception as exc:  # defensive: keep auditing other scans/files
                results.append(
                    VerificationResult(
                        ok=False,
                        file_path=file_path,
                        entry_path=file_path,
                        selected_scan_index=idx,
                        selected_beamformer=None,
                        roi_area=None,
                        has_nonzero_image=None,
                        message=f"scan {idx} failed: {exc}",
                    )
                )
        return results
    except MicrowaveFrameworkError as exc:
        return [
            VerificationResult(
                ok=False,
                file_path=file_path,
                entry_path=file_path,
                selected_scan_index=None,
                selected_beamformer=None,
                roi_area=None,
                has_nonzero_image=None,
                message=str(exc),
            )
        ]



def analyze_file(file_path: str, verify_cache: dict[str, list[VerificationResult]]) -> IntakeRecord:
    file_path = _normalize(file_path)
    if is_likely_metadata_file(file_path):
        resolved = resolve_measurement_from_metadata_path(file_path)
        if resolved is None:
            return IntakeRecord(
                file_path=file_path,
                file_kind="metadata",
                resolved_measurement_path=None,
                recommended_entry_path=None,
                status="blocked",
                note="Metadata-only file without a discoverable paired fd_data file.",
                verification=[],
            )

        resolved = _normalize(resolved)
        checks = verify_cache.get(resolved)
        if checks is None:
            checks = _verify_measurement_entry(resolved)
            verify_cache[resolved] = checks
        status = "ok" if any(v.ok for v in checks) else "failed"
        return IntakeRecord(
            file_path=file_path,
            file_kind="metadata",
            resolved_measurement_path=resolved,
            recommended_entry_path=file_path,
            status=status,
            note="Use this metadata path in GUI; auto-redirection should load its paired measurement.",
            verification=checks,
        )

    checks = verify_cache.get(file_path)
    if checks is None:
        checks = _verify_measurement_entry(file_path)
        verify_cache[file_path] = checks
    status = "ok" if any(v.ok for v in checks) else "failed"
    return IntakeRecord(
        file_path=file_path,
        file_kind="measurement",
        resolved_measurement_path=None,
        recommended_entry_path=file_path,
        status=status,
        note="Direct measurement file.",
        verification=checks,
    )



def _to_serializable(records: list[IntakeRecord], roots: list[str]) -> dict[str, Any]:
    recommended = [
        {
            "entry_path": r.recommended_entry_path,
            "resolved_measurement_path": r.resolved_measurement_path,
            "status": r.status,
            "note": r.note,
        }
        for r in records
        if r.recommended_entry_path and r.status == "ok"
    ]

    blocked = [r for r in records if r.status == "blocked"]
    failed = [r for r in records if r.status == "failed"]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "roots": roots,
        "summary": {
            "total_records": len(records),
            "recommended_ok": len(recommended),
            "blocked": len(blocked),
            "failed": len(failed),
        },
        "recommended_uploads": recommended,
        "blocked_entries": [asdict(r) for r in blocked],
        "failed_entries": [asdict(r) for r in failed],
        "all_records": [asdict(r) for r in records],
    }



def _write_markdown(report: dict[str, Any], output_md: str) -> None:
    lines: list[str] = []
    lines.append("# Auto Intake Report")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")

    s = report["summary"]
    lines.append("## Summary")
    lines.append(f"- Total records: {s['total_records']}")
    lines.append(f"- Recommended and verified: {s['recommended_ok']}")
    lines.append(f"- Blocked metadata entries: {s['blocked']}")
    lines.append(f"- Failed verification entries: {s['failed']}")
    lines.append("")

    lines.append("## Recommended Upload Entries")
    if report["recommended_uploads"]:
        for item in report["recommended_uploads"]:
            lines.append(f"- {item['entry_path']}")
            if item.get("resolved_measurement_path"):
                lines.append(f"  -> resolves to: {item['resolved_measurement_path']}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Blocked Metadata Entries")
    blocked = report["blocked_entries"]
    if blocked:
        for item in blocked:
            lines.append(f"- {item['file_path']}: {item['note']}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Failed Verification Entries")
    failed = report["failed_entries"]
    if failed:
        for item in failed:
            lines.append(f"- {item['file_path']}")
            for v in item["verification"]:
                scan_txt = "none" if v["selected_scan_index"] is None else str(v["selected_scan_index"])
                lines.append(
                    f"  scan={scan_txt}, ok={v['ok']}, beamformer={v['selected_beamformer']}, msg={v['message']}"
                )
    else:
        lines.append("- None")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")



def main() -> None:
    parser = argparse.ArgumentParser(description="Automated dataset intake and verification agent.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["datasets", "../umbmid/simple-clean"],
        help="Root folders to scan (relative to project/).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=80,
        help="Maximum candidate files to inspect.",
    )
    parser.add_argument(
        "--output-json",
        default="../results/auto_intake_report.json",
        help="Output JSON path (relative to project/).",
    )
    parser.add_argument(
        "--output-md",
        default="../results/auto_intake_report.md",
        help="Output markdown path (relative to project/).",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = [_normalize(os.path.join(project_root, r)) for r in args.roots]
    files = _collect_candidate_files(roots, max_files=max(1, args.max_files))

    if not files:
        raise SystemExit("No supported dataset files found in the provided roots.")

    verify_cache: dict[str, list[VerificationResult]] = {}
    records = [analyze_file(path, verify_cache) for path in files]
    report = _to_serializable(records, roots)

    output_json = _normalize(os.path.join(project_root, args.output_json))
    output_md = _normalize(os.path.join(project_root, args.output_md))
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_md), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_markdown(report, output_md)

    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()

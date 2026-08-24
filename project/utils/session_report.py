"""
utils/session_report.py

Build a downloadable review report (Markdown + JSON) from the current
pipeline session: loaded dataset, preprocessing, reconstruction, ROI,
and data-quality checks with suggested next steps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from data_loader.dataset_info import MicrowaveDataset, build_summary
from preprocessing.preprocessing_pipeline import PreprocessingResult
from reconstruction.reconstruction_manager import ReconstructionConfig, ROIRefinement
from roi.roi_detector import ROIResult


@dataclass
class ReconstructionSnapshot:
    """Serializable reconstruction outcomes for the session report."""

    selected_beamformer: str
    source_label: str
    config: ReconstructionConfig
    quality_metrics: dict[str, dict[str, float]]
    roi: ROIResult
    refinement: ROIRefinement
    tumor_xy_m: tuple[float, float] | None = None
    roi_centroid_m: tuple[float, float] | None = None
    tumor_gt_distance_m: float | None = None
    image_peak: float | None = None
    image_mean: float | None = None
    selection_mode: str | None = None
    prefer_off_center_roi: bool | None = None


@dataclass
class SessionReportContext:
    dataset: Optional[MicrowaveDataset] = None
    preprocessing: Optional[PreprocessingResult] = None
    reconstruction: Optional[ReconstructionSnapshot] = None
    notes: list[str] = field(default_factory=list)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "N/A"
        return f"{value:.6g}"
    return str(value)


def _dataset_section(dataset: MicrowaveDataset) -> tuple[str, dict[str, Any], list[str]]:
    summary = build_summary(dataset).to_display_dict()
    meta = dataset.metadata or {}
    checks: list[str] = []

    freqs = np.asarray(dataset.frequencies, dtype=float)
    s_params = np.asarray(dataset.s_parameters)
    if freqs.size < 2:
        checks.append("FAIL: frequency vector has fewer than 2 samples.")
    elif not np.all(np.diff(freqs) > 0):
        checks.append("FAIL: frequencies are not strictly increasing.")
    else:
        checks.append("OK: frequency vector is strictly increasing.")

    if freqs.size >= 2 and np.allclose(np.diff(freqs), np.diff(freqs)[0], rtol=1e-4, atol=0.0):
        checks.append("OK: frequency spacing is uniform (IFFT-ready).")
    elif freqs.size >= 2:
        checks.append("WARN: frequency spacing is non-uniform.")

    if not np.all(np.isfinite(s_params.real)) or not np.all(np.isfinite(s_params.imag)):
        checks.append("FAIL: S-parameters contain NaN/Inf.")
    else:
        checks.append("OK: S-parameters are finite.")

    peak = float(np.max(np.abs(s_params))) if s_params.size else 0.0
    mean = float(np.mean(np.abs(s_params))) if s_params.size else 0.0
    if peak <= 0:
        checks.append("FAIL: S-parameter magnitude peak is zero.")
    elif mean < 1e-8:
        checks.append("WARN: mean |S| is very small — check calibration / units.")
    else:
        checks.append(f"OK: |S| mean={mean:.3g}, peak={peak:.3g}.")

    lines = [
        "## 1. Dataset",
        "",
        f"- File: `{dataset.file_name}`",
        f"- Type: {dataset.file_type}",
        f"- Path: `{dataset.file_path}`",
        f"- Frequencies: {summary['Number of Frequencies']} ({summary['Frequency Range']})",
        f"- Ports / traces: {summary['Number of Ports']}",
        f"- Samples: {summary['Number of Samples']}",
        f"- In-memory size: {summary['Dataset Size']}",
    ]
    extra: dict[str, Any] = {"summary": summary, "metadata": {}}
    for key in (
        "dataset_family",
        "bmid_scan_index",
        "bmid_scan_id",
        "bmid_phant_id",
        "bmid_has_tumor",
        "bmid_sparam",
        "bmid_calibration",
        "antenna_radius_m",
        "wave_speed_m_per_s",
        "tumor_diameter_m",
        "tumor_x_m",
        "tumor_y_m",
    ):
        if key in meta and meta[key] is not None:
            extra["metadata"][key] = meta[key]
            lines.append(f"- {key}: {_fmt(meta[key])}")

    return "\n".join(lines), extra, checks


def _preprocessing_section(result: PreprocessingResult) -> tuple[str, dict[str, Any], list[str]]:
    q = result.quality_report
    checks: list[str] = []
    if q.snr_after_db >= q.snr_before_db:
        checks.append(
            f"OK: SNR improved ({q.snr_before_db:.2f} → {q.snr_after_db:.2f} dB)."
        )
    else:
        checks.append(
            f"WARN: SNR decreased ({q.snr_before_db:.2f} → {q.snr_after_db:.2f} dB); "
            "filter may be too aggressive or data already clean."
        )
    if abs(q.artifact_reduction_ratio) > 0.9:
        checks.append(
            "WARN: artifact reduction ratio is very high — possible over-suppression of target energy."
        )
    else:
        checks.append(f"OK: artifact reduction ratio = {q.artifact_reduction_ratio:.3f}.")

    lines = [
        "## 2. Preprocessing",
        "",
        f"- Filter: `{result.config.filter_method}`",
        f"- Calibration: `{result.config.calibration_method}`",
        f"- Normalization: `{result.config.normalization_method}`",
        f"- Artifact method: `{result.config.artifact_method}`",
        f"- Background subtraction enabled: {result.config.do_background_subtraction}",
        f"- Stages recorded: {', '.join(result.stage_outputs.keys()) or 'none'}",
        "",
        "### Signal quality",
    ]
    for key, value in q.to_display_dict().items():
        lines.append(f"- {key}: {value}")

    payload: dict[str, Any] = {
        "config": {
            "filter_method": result.config.filter_method,
            "calibration_method": result.config.calibration_method,
            "normalization_method": result.config.normalization_method,
            "artifact_method": result.config.artifact_method,
            "do_background_subtraction": result.config.do_background_subtraction,
        },
        "quality": q.to_display_dict(),
        "stages": list(result.stage_outputs.keys()),
    }

    if result.validation_report is not None:
        lines.append("")
        lines.append("### Module 2 validation report")
        for key, value in result.validation_report.to_display_dict().items():
            lines.append(f"- {key}: {value}")
        payload["validation"] = result.validation_report.to_display_dict()

    return "\n".join(lines), payload, checks


def _reconstruction_section(snap: ReconstructionSnapshot) -> tuple[str, dict[str, Any], list[str]]:
    checks: list[str] = []
    cfg = snap.config
    lines = [
        "## 3. Reconstruction",
        "",
        f"- Source: {snap.source_label}",
        f"- Selected beamformer: **{snap.selected_beamformer}**",
        f"- Selection mode: {snap.selection_mode or 'quality'}",
        f"- ROI mode: {'prefer off-center' if snap.prefer_off_center_roi else 'peak score'}",
        f"- Grid: {cfg.n_x} x {cfg.n_y}",
        f"- Antenna radius: {cfg.antenna_radius * 100:.1f} cm",
        f"- Phase-delay radius: {'on' if cfg.use_bmid_phase_delay_radius else 'off'}",
        f"- Antenna angle offset: {cfg.antenna_angle_offset_deg:.0f}°",
        f"- Antenna rotation: {'CW' if cfg.antenna_clockwise else 'CCW'}",
        f"- Axis flip: x={cfg.antenna_flip_x}, y={cfg.antenna_flip_y}",
        f"- Antenna arc: {cfg.antenna_span_deg:.0f}°",
        f"- Wave speed: {cfg.wave_speed:.3g} m/s",
        f"- FOV x: ({cfg.x_span[0]*100:.1f}, {cfg.x_span[1]*100:.1f}) cm",
        f"- FOV y: ({cfg.y_span[0]*100:.1f}, {cfg.y_span[1]*100:.1f}) cm",
        f"- Image peak / mean: {_fmt(snap.image_peak)} / {_fmt(snap.image_mean)}",
        "",
        "### Beamformer quality scores",
    ]
    for name, metrics in snap.quality_metrics.items():
        lines.append(
            f"- {name}: score={metrics.get('score', float('nan')):.4f}, "
            f"snr={metrics.get('snr', float('nan')):.4f}, "
            f"scr={metrics.get('scr', float('nan')):.4f}, "
            f"contrast={metrics.get('contrast', float('nan')):.4f}, "
            f"time={metrics.get('time', float('nan')):.4f}s"
        )

    roi = snap.roi
    lines.extend(
        [
            "",
            "### ROI",
            f"- Bounding box (px): {roi.bounding_box}",
            f"- Centroid (px): ({roi.centroid[0]:.2f}, {roi.centroid[1]:.2f})",
            f"- Area: {roi.area}",
            f"- Threshold: {roi.threshold:.3f}",
            f"- Score: {roi.score:.5g}",
        ]
    )
    if snap.roi_centroid_m is not None:
        lines.append(
            f"- Centroid (cm): ({snap.roi_centroid_m[0]*100:.2f}, {snap.roi_centroid_m[1]*100:.2f})"
        )
    if snap.tumor_xy_m is not None:
        lines.append(
            f"- Tumor GT (cm): ({snap.tumor_xy_m[0]*100:.2f}, {snap.tumor_xy_m[1]*100:.2f})"
        )
    if snap.tumor_gt_distance_m is not None:
        dist_cm = snap.tumor_gt_distance_m * 100.0
        lines.append(f"- ROI ↔ tumor GT distance: **{dist_cm:.2f} cm**")
        if dist_cm <= 1.5:
            checks.append(f"OK: ROI is within 1.5 cm of tumor GT ({dist_cm:.2f} cm).")
        elif dist_cm <= 3.0:
            checks.append(
                f"WARN: ROI is {dist_cm:.2f} cm from tumor GT — try angle offset / axis flip "
                "before lowering wave speed."
            )
        else:
            checks.append(
                f"FAIL-ish: ROI is {dist_cm:.2f} cm from tumor GT — check antenna geometry "
                "(offset / flip / 355° arc / phase-delay radius), FOV, and avoid heavy SVD on _adi."
            )
    else:
        checks.append("INFO: no tumor ground truth in metadata (healthy scan or non-BMID data).")

    ref = snap.refinement
    lines.extend(
        [
            "",
            "### ROI refinement",
            f"- Algorithm: {ref.algorithm}",
            f"- Grid: {ref.grid_shape[1]} x {ref.grid_shape[0]}",
            f"- X span (m): {ref.x_span}",
            f"- Y span (m): {ref.y_span}",
        ]
    )

    payload = {
        "selected_beamformer": snap.selected_beamformer,
        "source_label": snap.source_label,
        "selection_mode": snap.selection_mode,
        "prefer_off_center_roi": snap.prefer_off_center_roi,
        "config": {
            "n_x": cfg.n_x,
            "n_y": cfg.n_y,
            "antenna_radius_m": cfg.antenna_radius,
            "use_bmid_phase_delay_radius": cfg.use_bmid_phase_delay_radius,
            "antenna_angle_offset_deg": cfg.antenna_angle_offset_deg,
            "antenna_clockwise": cfg.antenna_clockwise,
            "antenna_flip_x": cfg.antenna_flip_x,
            "antenna_flip_y": cfg.antenna_flip_y,
            "antenna_span_deg": cfg.antenna_span_deg,
            "wave_speed_m_per_s": cfg.wave_speed,
            "x_span_m": list(cfg.x_span),
            "y_span_m": list(cfg.y_span),
        },
        "quality_metrics": snap.quality_metrics,
        "roi": {
            "bounding_box": list(roi.bounding_box),
            "centroid_px": list(roi.centroid),
            "area": roi.area,
            "threshold": roi.threshold,
            "score": roi.score,
            "centroid_m": list(snap.roi_centroid_m) if snap.roi_centroid_m else None,
        },
        "tumor_xy_m": list(snap.tumor_xy_m) if snap.tumor_xy_m else None,
        "tumor_gt_distance_m": snap.tumor_gt_distance_m,
        "image_peak": snap.image_peak,
        "image_mean": snap.image_mean,
    }
    return "\n".join(lines), payload, checks


def _next_steps(checks: list[str], ctx: SessionReportContext) -> list[str]:
    steps = [
        "Re-read this report’s Data quality checks before changing code.",
        "For BMID `_adi` scans: keep wave speed 3.0e8, preprocess none/mild, and try "
        "antenna angle offset / Flip X-Y / CW before lowering c.",
        "Use Prefer DMAS-D4 or Closest to tumor GT when quality auto-pick locks onto DAS ring clutter.",
        "Compare one tumor scan vs one healthy scan with identical reconstruction settings.",
        "Save Hamming-windowed / time-domain Module 2 exports when completing the UG Week-3 handoff.",
    ]
    if ctx.preprocessing is None:
        steps.insert(0, "Run Module 2 preprocessing before treating reconstruction as final.")
    if ctx.reconstruction is None:
        steps.insert(0, "Run Module 3 reconstruction to fill ROI / beamformer sections.")
    if any(c.startswith("FAIL") for c in checks):
        steps.insert(0, "Fix FAIL checks first (data integrity) before algorithm tuning.")
    return steps


def build_session_report(ctx: SessionReportContext) -> tuple[str, dict[str, Any]]:
    """Return (markdown_text, json_payload)."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections: list[str] = [
        "# Microwave Imaging Framework — Session Report",
        "",
        f"- Generated: {generated}",
        "",
    ]
    payload: dict[str, Any] = {"generated_at": generated, "checks": [], "sections": {}}
    all_checks: list[str] = []

    if ctx.dataset is None:
        sections.append("## 1. Dataset\n\n- No dataset loaded.\n")
        all_checks.append("FAIL: no dataset loaded.")
    else:
        md, data, checks = _dataset_section(ctx.dataset)
        sections.append(md)
        payload["sections"]["dataset"] = data
        all_checks.extend(checks)

    sections.append("")
    if ctx.preprocessing is None:
        sections.append("## 2. Preprocessing\n\n- Not run yet.\n")
        all_checks.append("WARN: preprocessing not run.")
    else:
        md, data, checks = _preprocessing_section(ctx.preprocessing)
        sections.append(md)
        payload["sections"]["preprocessing"] = data
        all_checks.extend(checks)

    sections.append("")
    if ctx.reconstruction is None:
        sections.append("## 3. Reconstruction\n\n- Not run yet.\n")
        all_checks.append("WARN: reconstruction not run.")
    else:
        md, data, checks = _reconstruction_section(ctx.reconstruction)
        sections.append(md)
        payload["sections"]["reconstruction"] = data
        all_checks.extend(checks)

    sections.extend(["", "## 4. Data quality checks", ""])
    for check in all_checks:
        sections.append(f"- {check}")

    steps = _next_steps(all_checks, ctx)
    sections.extend(["", "## 5. Suggested next steps", ""])
    for idx, step in enumerate(steps, start=1):
        sections.append(f"{idx}. {step}")

    if ctx.notes:
        sections.extend(["", "## 6. Notes", ""])
        for note in ctx.notes:
            sections.append(f"- {note}")

    sections.append("")
    payload["checks"] = all_checks
    payload["next_steps"] = steps
    payload["notes"] = list(ctx.notes)
    return "\n".join(sections), payload


def write_session_report(
    ctx: SessionReportContext,
    output_dir: str,
    basename: str | None = None,
    figure_paths: list[str] | None = None,
) -> dict[str, str]:
    """
    Write Markdown + JSON report under output_dir.
    Returns dict of artifact paths (markdown, json, and any copied figure notes).
    """
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stub = basename or f"session_report_{stamp}"
    markdown, payload = build_session_report(ctx)

    if figure_paths:
        payload["figures"] = [os.path.basename(path) for path in figure_paths if path]
        markdown += "\n## Figures\n\n"
        for path in figure_paths:
            if path and os.path.isfile(path):
                markdown += f"- `{os.path.basename(path)}`\n"

    md_path = os.path.join(output_dir, f"{file_stub}.md")
    json_path = os.path.join(output_dir, f"{file_stub}.json")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    return {"markdown": md_path, "json": json_path}

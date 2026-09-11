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

import matplotlib
import numpy as np

from quality.metrics import compute_image_ccr, compute_image_fwhm_pixels

from data_loader.dataset_info import MicrowaveDataset, build_summary
from preprocessing.preprocessing_pipeline import PreprocessingResult
from reconstruction.reconstruction_manager import ReconstructionConfig, ROIRefinement
from roi.roi_detector import ROIResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt


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
    selected_image: np.ndarray | None = None
    beamformer_images: dict[str, np.ndarray] | None = None
    tumor_candidate_result: dict[str, Any] | None = None


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
        f"- Filter parameters: {result.config.filter_kwargs or {}}",
        f"- Calibration: `{result.config.calibration_method}`",
        f"- Normalization: `{result.config.normalization_method}`",
        f"- Artifact method: `{result.config.artifact_method}`",
        f"- Background subtraction enabled: {result.config.do_background_subtraction}",
        f"- Spike-removal method: `{result.config.spike_detection_method}`",
        f"- Spike-removal parameters: {result.config.spike_detection_kwargs or {}}",
        f"- Stages recorded: {', '.join(result.stage_outputs.keys()) or 'none'}",
        "",
        "### Signal quality",
    ]
    for key, value in q.to_display_dict().items():
        lines.append(f"- {key}: {value}")

    payload: dict[str, Any] = {
        "config": {
            "filter_method": result.config.filter_method,
            "filter_kwargs": dict(result.config.filter_kwargs),
            "calibration_method": result.config.calibration_method,
            "normalization_method": result.config.normalization_method,
            "artifact_method": result.config.artifact_method,
            "do_background_subtraction": result.config.do_background_subtraction,
            "spike_detection_method": result.config.spike_detection_method,
            "spike_detection_kwargs": dict(result.config.spike_detection_kwargs),
        },
        "quality": q.to_display_dict(),
        "stages": list(result.stage_outputs.keys()),
    }

    if result.validation_report is not None:
        lines.append("")
        lines.append("### Module 2 validation report")
        for key, value in result.validation_report.to_display_dict().items():
            lines.append(f"- {key}: {value}")
        if result.validation_report.interpolation_performed:
            checks.append("OK: interpolation was applied to enforce uniform spacing.")
        else:
            checks.append("OK: interpolation not required (input spacing already uniform).")
        checks.append(
            f"OK: raw vs filtered comparison tracked via NRMSE={result.validation_report.nrmse:.3f}."
        )
        payload["validation"] = result.validation_report.to_display_dict()

    return "\n".join(lines), payload, checks


def _selected_scr(snap: ReconstructionSnapshot) -> float | None:
    metrics = snap.quality_metrics.get(snap.selected_beamformer, {})
    value = metrics.get("scr")
    if value is None:
        return None
    return float(value)


def _selected_ccr(snap: ReconstructionSnapshot) -> float | None:
    if snap.selected_image is None:
        return None
    return float(compute_image_ccr(np.asarray(snap.selected_image, dtype=float)))


def _selected_fwhm_cm(snap: ReconstructionSnapshot) -> float | None:
    if snap.selected_image is None:
        return None
    image = np.asarray(snap.selected_image, dtype=float)
    if image.ndim != 2:
        return None

    fwhm_px = compute_image_fwhm_pixels(image)
    x_span_cm = abs(float(snap.config.x_span[1] - snap.config.x_span[0])) * 100.0
    y_span_cm = abs(float(snap.config.y_span[1] - snap.config.y_span[0])) * 100.0
    px_to_cm_x = x_span_cm / max(image.shape[1] - 1, 1)
    px_to_cm_y = y_span_cm / max(image.shape[0] - 1, 1)
    px_to_cm = 0.5 * (px_to_cm_x + px_to_cm_y)
    return float(fwhm_px * px_to_cm)


def _localization_validation_row(snap: ReconstructionSnapshot) -> dict[str, Any]:
    return {
        "localization_error_cm": None
        if snap.tumor_gt_distance_m is None
        else float(snap.tumor_gt_distance_m * 100.0),
        "scr": _selected_scr(snap),
        "ccr": _selected_ccr(snap),
        "fwhm_cm": _selected_fwhm_cm(snap),
    }


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

    row = _localization_validation_row(snap)
    lines.extend(
        [
            "",
            "### Localization validation",
            "| Metric | Value |",
            "|---|---|",
            f"| Localization error (cm) | {_fmt(row['localization_error_cm'])} |",
            f"| SCR | {_fmt(row['scr'])} |",
            f"| CCR | {_fmt(row['ccr'])} |",
            f"| FWHM (cm) | {_fmt(row['fwhm_cm'])} |",
        ]
    )

    if snap.tumor_candidate_result:
        candidate = snap.tumor_candidate_result
        lines.extend(
            [
                "",
                "### Tumor candidate decision",
                f"- Candidate: {'Yes' if candidate.get('is_tumor_candidate') else 'No'}",
                f"- Confidence: {_fmt(candidate.get('confidence'))}",
                f"- Suspicion score: {_fmt(candidate.get('suspicion_score'))}",
                f"- Threshold: {_fmt(candidate.get('threshold'))}",
                f"- Reason: {candidate.get('reason', 'N/A')}",
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
        "localization_validation": row,
        "tumor_candidate_result": snap.tumor_candidate_result,
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


def _healthy_tumor_validation_status(ctx: SessionReportContext) -> tuple[str, dict[str, Any]]:
    dataset = ctx.dataset or (ctx.preprocessing.processed_dataset if ctx.preprocessing else None)
    if dataset is None:
        return (
            "- Status: Not evaluated (no dataset loaded).",
            {"status": "not-evaluated", "reason": "no-dataset"},
        )

    meta = dataset.metadata or {}
    has_tumor = meta.get("bmid_has_tumor")
    if has_tumor is None:
        return (
            "- Status: Single-case run only (healthy/tumour labels not available in this dataset).",
            {"status": "single-case", "labeled_case": False},
        )

    case_label = "tumour" if bool(has_tumor) else "healthy"
    return (
        "\n".join(
            [
                f"- Current case: {case_label}",
                "- Healthy + tumour pair validation: pending in this report unless both case runs are aggregated.",
            ]
        ),
        {
            "status": "single-labeled-case",
            "current_case": case_label,
            "pair_validation": "pending-aggregate",
        },
    )


def _save_magnitude_plot(
    ctx: SessionReportContext, output_dir: str, file_stub: str
) -> str | None:
    if ctx.preprocessing is None:
        return None

    p = ctx.preprocessing
    figure_path = os.path.join(output_dir, f"{file_stub}_s21_magnitude_compare.png")

    if p.touchstone_s21_result is not None:
        freqs = np.asarray(p.touchstone_s21_result.uniform_frequencies_hz, dtype=float)
        raw = np.asarray(p.touchstone_s21_result.raw_s21, dtype=complex)
        filtered = np.asarray(p.touchstone_s21_result.filtered_s21, dtype=complex)
    else:
        freqs = np.asarray(p.original_dataset.frequencies, dtype=float)
        raw = np.asarray(p.original_dataset.s_parameters, dtype=complex).reshape(freqs.shape[0], -1)[:, 0]
        filtered = (
            np.asarray(p.processed_dataset.s_parameters, dtype=complex)
            .reshape(freqs.shape[0], -1)[:, 0]
        )

    mag_raw_db = 20.0 * np.log10(np.abs(raw) + 1e-12)
    mag_filtered_db = 20.0 * np.log10(np.abs(filtered) + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout=True)
    ax.plot(freqs / 1e9, mag_raw_db, label="Raw |S21| (dB)", linewidth=1.3)
    ax.plot(freqs / 1e9, mag_filtered_db, label="Preprocessed |S21| (dB)", linewidth=1.3)
    ax.set_title("S21 Magnitude: Raw vs Preprocessed")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _save_phase_plot(
    ctx: SessionReportContext, output_dir: str, file_stub: str
) -> str | None:
    if ctx.preprocessing is None or ctx.preprocessing.touchstone_s21_result is None:
        return None

    s21 = ctx.preprocessing.touchstone_s21_result
    freqs = np.asarray(s21.uniform_frequencies_hz, dtype=float)
    figure_path = os.path.join(output_dir, f"{file_stub}_phase_wrapped_unwrapped.png")

    fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout=True)
    ax.plot(freqs / 1e9, s21.wrapped_phase_deg, label="Wrapped phase (deg)", linewidth=1.2)
    ax.plot(freqs / 1e9, s21.unwrapped_phase_deg, label="Unwrapped phase (deg)", linewidth=1.2)
    ax.set_title("S21 Phase: Wrapped vs Unwrapped")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Phase (deg)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _save_time_domain_plot(
    ctx: SessionReportContext, output_dir: str, file_stub: str
) -> str | None:
    if ctx.preprocessing is None or ctx.preprocessing.week3_result is None:
        return None

    week3 = ctx.preprocessing.week3_result
    figure_path = os.path.join(output_dir, f"{file_stub}_ifft_time_domain.png")
    time_ns = np.asarray(week3.time_s, dtype=float) * 1e9

    time_hamming = np.asarray(week3.time_hamming)
    if time_hamming.ndim == 2:
        trace = np.abs(time_hamming[:, 0])
    else:
        trace = np.abs(time_hamming)

    fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout=True)
    ax.plot(time_ns, trace, label="|IFFT(S21_hamming)|", linewidth=1.3)

    if week3.time_reference_subtracted is not None:
        ref_sub = np.asarray(week3.time_reference_subtracted)
        ref_trace = np.abs(ref_sub[:, 0]) if ref_sub.ndim == 2 else np.abs(ref_sub)
        ax.plot(time_ns, ref_trace, label="|Time reference subtracted|", linewidth=1.2)

    ax.set_title("IFFT Time-Domain Signal")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Magnitude (a.u.)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _save_same_scale_beamformer_plot(
    ctx: SessionReportContext, output_dir: str, file_stub: str
) -> str | None:
    if ctx.reconstruction is None or not ctx.reconstruction.beamformer_images:
        return None

    images = ctx.reconstruction.beamformer_images
    required = ["DAS", "DMAS", "DMAS-D4"]
    if any(name not in images for name in required):
        return None

    figure_path = os.path.join(output_dir, f"{file_stub}_beamformers_same_scale.png")
    cfg = ctx.reconstruction.config
    extent = [
        cfg.x_span[0] * 100.0,
        cfg.x_span[1] * 100.0,
        cfg.y_span[0] * 100.0,
        cfg.y_span[1] * 100.0,
    ]
    all_vals = np.concatenate([np.asarray(images[name], dtype=float).ravel() for name in required])
    vmin = float(np.percentile(all_vals, 5))
    vmax = float(np.percentile(all_vals, 99))
    if vmax <= vmin:
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals) + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    for ax, name in zip(axes, required):
        image = np.asarray(images[name], dtype=float)
        im = ax.imshow(
            image,
            cmap="inferno",
            origin="lower",
            aspect="equal",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(name)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")

        if ctx.reconstruction.roi_centroid_m is not None:
            ax.plot(
                ctx.reconstruction.roi_centroid_m[0] * 100.0,
                ctx.reconstruction.roi_centroid_m[1] * 100.0,
                "wo",
                markersize=4,
            )
        if ctx.reconstruction.tumor_xy_m is not None:
            ax.plot(
                ctx.reconstruction.tumor_xy_m[0] * 100.0,
                ctx.reconstruction.tumor_xy_m[1] * 100.0,
                marker="x",
                markersize=8,
                markeredgewidth=1.8,
                color="lime",
            )

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.030, pad=0.03)
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _generate_validation_figures(
    ctx: SessionReportContext, output_dir: str, file_stub: str
) -> list[str]:
    generated = [
        _save_magnitude_plot(ctx, output_dir, file_stub),
        _save_phase_plot(ctx, output_dir, file_stub),
        _save_time_domain_plot(ctx, output_dir, file_stub),
        _save_same_scale_beamformer_plot(ctx, output_dir, file_stub),
    ]
    return [path for path in generated if path and os.path.isfile(path)]


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

    validation_lines, validation_payload = _healthy_tumor_validation_status(ctx)
    sections.extend(["", "## 6. Healthy vs tumour validation", "", validation_lines])
    payload["healthy_tumour_validation"] = validation_payload

    if ctx.notes:
        sections.extend(["", "## 7. Notes", ""])
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

    generated_validation_figures = _generate_validation_figures(ctx, output_dir, file_stub)
    all_figures = list(figure_paths or []) + generated_validation_figures
    dedup_figures: list[str] = []
    seen: set[str] = set()
    for path in all_figures:
        if not path:
            continue
        normalized = os.path.normpath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        dedup_figures.append(path)

    if dedup_figures:
        payload["figures"] = [os.path.basename(path) for path in dedup_figures if path]
        markdown += "\n## Figures\n\n"
        for path in dedup_figures:
            if path and os.path.isfile(path):
                figure_name = os.path.basename(path)
                markdown += f"### {figure_name}\n\n"
                markdown += f"![{figure_name}]({figure_name})\n\n"

    md_path = os.path.join(output_dir, f"{file_stub}.md")
    json_path = os.path.join(output_dir, f"{file_stub}.json")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    return {"markdown": md_path, "json": json_path}

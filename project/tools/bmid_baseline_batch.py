"""Batch BMID check: peak ROI vs tight ROI + quick Auto Tweak per scan."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.bmid_loader import list_bmid_scans
from data_loader.loader import load_dataset
from reconstruction.auto_calibrate import auto_calibrate_reconstruction
from reconstruction.reconstruction_manager import ReconstructionConfig, reconstruct_all
from roi.roi_detector import detect_roi, roi_centroid_meters, tumor_likelihood_features


def _locked_config() -> ReconstructionConfig:
    return ReconstructionConfig(
        x_span=(-0.06, 0.06),
        y_span=(-0.06, 0.06),
        n_x=64,
        n_y=64,
        antenna_radius=0.18,
        wave_speed=3.0e8,
        antenna_angle_offset_deg=0.0,
        antenna_clockwise=False,
        antenna_flip_x=False,
        antenna_flip_y=False,
        antenna_span_deg=360.0,
        use_bmid_phase_delay_radius=False,
    )


def _gt_cm(meta: dict) -> tuple[float, float] | None:
    if not meta.get("bmid_has_tumor"):
        return None
    if meta.get("tumor_x_m") is None or meta.get("tumor_y_m") is None:
        return None
    return float(meta["tumor_x_m"]) * 100.0, float(meta["tumor_y_m"]) * 100.0


def _dist_cm(centroid_m, tumor_xy_m) -> float | None:
    if tumor_xy_m is None:
        return None
    return float(np.hypot(centroid_m[0] - tumor_xy_m[0], centroid_m[1] - tumor_xy_m[1]) * 100.0)


def evaluate(path: str, scan_index: int, *, do_auto: bool) -> dict:
    ds = load_dataset(path, scan_index=scan_index)
    cfg = _locked_config()
    images, _ = reconstruct_all(ds.s_parameters, ds.frequencies, config=cfg, return_timings=True)
    image = images["DMAS-D4"]
    meta = ds.metadata or {}
    tumor_xy_m = None
    if meta.get("bmid_has_tumor") and meta.get("tumor_x_m") is not None:
        tumor_xy_m = (float(meta["tumor_x_m"]), float(meta["tumor_y_m"]))

    peak_roi = detect_roi(image, prefer_off_center=False, tight_peak=False)
    tight_roi = detect_roi(image, prefer_off_center=False, tight_peak=True)
    peak_c = roi_centroid_meters(peak_roi, image.shape, cfg.x_span, cfg.y_span)
    tight_c = roi_centroid_meters(tight_roi, image.shape, cfg.x_span, cfg.y_span)
    peak_feat = tumor_likelihood_features(image, peak_roi)
    tight_feat = tumor_likelihood_features(image, tight_roi)

    auto_dist = None
    auto_label = None
    if do_auto and tumor_xy_m is not None:
        result = auto_calibrate_reconstruction(
            ds.s_parameters,
            ds.frequencies,
            cfg,
            tumor_xy_m=tumor_xy_m,
            quick=True,
            top_k_full=4,
            baseline_prefer_off_center=False,
            baseline_tight_peak=False,
        )
        auto_dist = (
            None
            if result.best.tumor_gt_distance_m is None
            else result.best.tumor_gt_distance_m * 100.0
        )
        auto_label = f"{result.best.beamformer}|{result.best.geometry.label()}"
        if not result.improved and result.baseline is not None:
            auto_dist = (
                None
                if result.baseline.tumor_gt_distance_m is None
                else result.baseline.tumor_gt_distance_m * 100.0
            )
            auto_label = f"kept|{result.baseline.beamformer}"

    return {
        "index": scan_index,
        "phant_id": meta.get("bmid_phant_id"),
        "has_tumor": bool(meta.get("bmid_has_tumor")),
        "gt_cm": _gt_cm(meta),
        "peak_dist": _dist_cm(peak_c, tumor_xy_m),
        "tight_dist": _dist_cm(tight_c, tumor_xy_m),
        "peak_sus": peak_feat["suspicion"],
        "tight_sus": tight_feat["suspicion"],
        "peak_area": peak_roi.area,
        "tight_area": tight_roi.area,
        "auto_dist": auto_dist,
        "auto_label": auto_label,
    }


def main() -> None:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "fd_data_s21_adi.mat"))
    scans = list_bmid_scans(path)
    tumors = [s for s in scans if s.has_tumor]
    healthy = [s for s in scans if not s.has_tumor]

    chosen = []
    seen = set()
    for s in tumors:
        if s.index == 0 or s.phant_id not in seen:
            chosen.append(s)
            seen.add(s.phant_id)
        if len(chosen) >= 4:
            break
    if healthy:
        chosen.append(healthy[0])

    print("Baseline geometry locked; comparing Peak ROI vs Tight peak ROI")
    print("Plus quick Auto Tweak per tumor scan (keeps baseline if no gain)")
    print()
    print(
        f"{'idx':>4} {'type':>6} {'phant':>6} {'peak':>6} {'tight':>6} {'auto':>6} "
        f"{'Psus':>6} {'Tsus':>6} {'Parea':>6} {'Tarea':>6}"
    )

    rows = []
    for s in chosen:
        print(f"... scan {s.index}", flush=True)
        row = evaluate(path, s.index, do_auto=s.has_tumor)
        rows.append(row)
        kind = "TUMOR" if row["has_tumor"] else "HEALTH"
        peak = "—" if row["peak_dist"] is None else f"{row['peak_dist']:.2f}"
        tight = "—" if row["tight_dist"] is None else f"{row['tight_dist']:.2f}"
        auto = "—" if row["auto_dist"] is None else f"{row['auto_dist']:.2f}"
        print(
            f"{row['index']:4d} {kind:>6} {str(row['phant_id']):>6} {peak:>6} {tight:>6} {auto:>6} "
            f"{row['peak_sus']:6.2f} {row['tight_sus']:6.2f} {row['peak_area']:6d} {row['tight_area']:6d}"
        )
        if row["auto_label"]:
            print(f"     auto -> {row['auto_label']}")

    tumor_rows = [r for r in rows if r["has_tumor"]]
    if tumor_rows:
        peak_vals = [r["peak_dist"] for r in tumor_rows if r["peak_dist"] is not None]
        tight_vals = [r["tight_dist"] for r in tumor_rows if r["tight_dist"] is not None]
        auto_vals = [r["auto_dist"] for r in tumor_rows if r["auto_dist"] is not None]
        print()
        print(
            f"Tumor means: peak={np.mean(peak_vals):.2f} cm, "
            f"tight={np.mean(tight_vals):.2f} cm, "
            f"auto={np.mean(auto_vals):.2f} cm"
        )
    healthy_rows = [r for r in rows if not r["has_tumor"]]
    if healthy_rows and tumor_rows:
        h = healthy_rows[0]
        t0 = tumor_rows[0]
        print(
            f"Suspicion (peak ROI): healthy={h['peak_sus']:.2f}, "
            f"tumor0={t0['peak_sus']:.2f} "
            f"(higher => more tumor-like)"
        )


if __name__ == "__main__":
    main()

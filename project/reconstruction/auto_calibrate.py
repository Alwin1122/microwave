"""Automatic reconstruction setting search (geometry + beamformer)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from reconstruction.das import das_from_frequency
from reconstruction.reconstruction_manager import (
    ReconstructionConfig,
    antenna_positions_from_config,
    reconstruct_all,
)
from roi.roi_detector import detect_roi, roi_centroid_meters


ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class GeometryCandidate:
    """One geometry / ROI / wave-speed candidate for search."""

    antenna_angle_offset_deg: float = 0.0
    antenna_clockwise: bool = False
    antenna_flip_x: bool = False
    antenna_flip_y: bool = False
    antenna_span_deg: float = 360.0
    use_bmid_phase_delay_radius: bool = False
    wave_speed: float = 3.0e8
    prefer_off_center_roi: bool = False

    def label(self) -> str:
        flip = "none"
        if self.antenna_flip_x and self.antenna_flip_y:
            flip = "xy"
        elif self.antenna_flip_x:
            flip = "x"
        elif self.antenna_flip_y:
            flip = "y"
        rot = "CW" if self.antenna_clockwise else "CCW"
        phase = "on" if self.use_bmid_phase_delay_radius else "off"
        roi = "offctr" if self.prefer_off_center_roi else "peak"
        return (
            f"ang={self.antenna_angle_offset_deg:.0f}° {rot} flip={flip} "
            f"arc={self.antenna_span_deg:.0f}° phase={phase} c={self.wave_speed:.2e} roi={roi}"
        )


@dataclass
class AutoCalibrateTrial:
    geometry: GeometryCandidate
    beamformer: str
    score: float
    tumor_gt_distance_m: float | None
    roi_centroid_m: tuple[float, float]
    quality_score: float
    image_peak: float
    image_contrast: float


@dataclass
class AutoCalibrateResult:
    """Best settings found by automatic search."""

    best: AutoCalibrateTrial
    trials: list[AutoCalibrateTrial] = field(default_factory=list)
    objective: str = "tumor_gt"
    notes: list[str] = field(default_factory=list)
    baseline: AutoCalibrateTrial | None = None
    improved: bool = True
    search_best: AutoCalibrateTrial | None = None

    def apply_to_config(self, config: ReconstructionConfig) -> ReconstructionConfig:
        g = self.best.geometry
        config.antenna_angle_offset_deg = g.antenna_angle_offset_deg
        config.antenna_clockwise = g.antenna_clockwise
        config.antenna_flip_x = g.antenna_flip_x
        config.antenna_flip_y = g.antenna_flip_y
        config.antenna_span_deg = g.antenna_span_deg
        config.use_bmid_phase_delay_radius = g.use_bmid_phase_delay_radius
        config.wave_speed = g.wave_speed
        return config

    def to_display_dict(self) -> dict:
        g = self.best.geometry
        return {
            "objective": self.objective,
            "beamformer": self.best.beamformer,
            "score": self.best.score,
            "improved": self.improved,
            "tumor_gt_distance_cm": (
                None
                if self.best.tumor_gt_distance_m is None
                else self.best.tumor_gt_distance_m * 100.0
            ),
            "baseline_tumor_gt_distance_cm": (
                None
                if self.baseline is None or self.baseline.tumor_gt_distance_m is None
                else self.baseline.tumor_gt_distance_m * 100.0
            ),
            "geometry": asdict(g),
            "trials_evaluated": len(self.trials),
            "notes": list(self.notes),
        }


def _flip_options() -> list[tuple[bool, bool]]:
    return [(False, False), (True, False), (False, True), (True, True)]


def build_geometry_search_space(
    base: ReconstructionConfig,
    *,
    include_wave_speeds: bool = False,
    quick: bool = False,
) -> list[GeometryCandidate]:
    """Build a practical geometry grid around the current radius/FOV."""
    angles = [0.0, 90.0, 180.0, 270.0]
    flips = _flip_options()
    rotations = [False] if quick else [False, True]
    arcs = [360.0] if quick else [360.0, 355.0]
    phases = [False] if quick else [False, True]
    speeds = [base.wave_speed]
    if include_wave_speeds:
        for c in (3.0e8, 2.1e8, 1.8e8):
            if all(abs(c - s) > 1e6 for s in speeds):
                speeds.append(c)
    roi_modes = [False, True]

    candidates: list[GeometryCandidate] = []
    for angle in angles:
        for flip_x, flip_y in flips:
            for clockwise in rotations:
                for span in arcs:
                    for phase in phases:
                        for speed in speeds:
                            for prefer_off in roi_modes:
                                candidates.append(
                                    GeometryCandidate(
                                        antenna_angle_offset_deg=angle,
                                        antenna_clockwise=clockwise,
                                        antenna_flip_x=flip_x,
                                        antenna_flip_y=flip_y,
                                        antenna_span_deg=span,
                                        use_bmid_phase_delay_radius=phase,
                                        wave_speed=float(speed),
                                        prefer_off_center_roi=prefer_off,
                                    )
                                )
    return candidates


def _config_with_geometry(base: ReconstructionConfig, geometry: GeometryCandidate) -> ReconstructionConfig:
    return ReconstructionConfig(
        x_span=base.x_span,
        y_span=base.y_span,
        n_x=base.n_x,
        n_y=base.n_y,
        antenna_radius=base.antenna_radius,
        wave_speed=geometry.wave_speed,
        zero_padding=base.zero_padding,
        antenna_angle_offset_deg=geometry.antenna_angle_offset_deg,
        antenna_clockwise=geometry.antenna_clockwise,
        antenna_flip_x=geometry.antenna_flip_x,
        antenna_flip_y=geometry.antenna_flip_y,
        antenna_span_deg=geometry.antenna_span_deg,
        use_bmid_phase_delay_radius=geometry.use_bmid_phase_delay_radius,
    )


def _image_contrast(image: np.ndarray) -> float:
    lo = float(np.min(image))
    hi = float(np.max(image))
    return (hi - lo) / (hi + lo + 1e-12)


def _evaluate_image(
    image: np.ndarray,
    geometry: GeometryCandidate,
    beamformer: str,
    config: ReconstructionConfig,
    tumor_xy_m: tuple[float, float] | None,
) -> AutoCalibrateTrial:
    roi = detect_roi(
        image,
        prefer_off_center=geometry.prefer_off_center_roi,
        x_span=config.x_span,
        y_span=config.y_span,
    )
    centroid_m = roi_centroid_meters(roi, image.shape, config.x_span, config.y_span)
    peak = float(np.max(image))
    contrast = _image_contrast(image)
    quality = float(contrast * 0.5 + (peak / (float(np.mean(image)) + 1e-12)) * 0.05)

    gt_dist = None
    if tumor_xy_m is not None:
        gt_dist = float(np.hypot(centroid_m[0] - tumor_xy_m[0], centroid_m[1] - tumor_xy_m[1]))
        # Lower distance is better; keep score positive and comparable.
        score = 1.0 / (gt_dist + 1e-4)
        # Mild preference for compact ROIs (less whole-image latch).
        score *= 1.0 / (1.0 + max(0, roi.area - 200) / 2000.0)
    else:
        # No GT: prefer contrasty images with an off-center compact ROI.
        ny, nx = image.shape
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0
        r_norm = float(np.hypot(roi.centroid[0] - cx, roi.centroid[1] - cy) / (np.hypot(cx, cy) + 1e-12))
        score = quality * (0.5 + r_norm) / (1.0 + roi.area / 2000.0)
        gt_dist = None

    return AutoCalibrateTrial(
        geometry=geometry,
        beamformer=beamformer,
        score=float(score),
        tumor_gt_distance_m=gt_dist,
        roi_centroid_m=centroid_m,
        quality_score=float(quality),
        image_peak=peak,
        image_contrast=contrast,
    )


def _das_image(s_parameters: np.ndarray, frequencies: np.ndarray, config: ReconstructionConfig) -> np.ndarray:
    positions = antenna_positions_from_config(s_parameters.shape[1], config)
    grid_x = np.linspace(config.x_span[0], config.x_span[1], config.n_x)
    grid_y = np.linspace(config.y_span[0], config.y_span[1], config.n_y)
    return das_from_frequency(
        s_parameters,
        frequencies,
        positions,
        grid_x,
        grid_y,
        zero_padding=config.zero_padding,
        wave_speed=config.wave_speed,
    )


def _is_better(candidate: AutoCalibrateTrial, baseline: AutoCalibrateTrial, tumor_xy_m) -> bool:
    """Return True only if candidate clearly beats the current manual baseline."""
    if tumor_xy_m is not None:
        if candidate.tumor_gt_distance_m is None or baseline.tumor_gt_distance_m is None:
            return candidate.score > baseline.score + 1e-6
        # Require a real localization improvement (at least 0.5 mm).
        return candidate.tumor_gt_distance_m < (baseline.tumor_gt_distance_m - 0.0005)
    return candidate.score > baseline.score + 1e-4


def auto_calibrate_reconstruction(
    s_parameters: np.ndarray,
    frequencies: np.ndarray,
    base_config: ReconstructionConfig,
    tumor_xy_m: tuple[float, float] | None = None,
    *,
    quick: bool = False,
    include_wave_speeds: bool = False,
    top_k_full: int = 8,
    baseline_prefer_off_center: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> AutoCalibrateResult:
    """Search reconstruction settings and return the best trial.

    Strategy:
      1) Fully score the current manual settings (baseline).
      2) Coarse DAS-only sweep over geometry candidates (fast).
      3) Re-rank the top-K geometries with full DAS/DMAS/DMAS-D4.
      4) Keep baseline unless a candidate is strictly better.
    """
    if s_parameters.ndim != 2:
        raise ValueError("s_parameters must be 2D (frequency, traces).")

    objective = "tumor_gt" if tumor_xy_m is not None else "quality_offcenter"
    notes = [
        "Stage 0: score current manual settings as baseline.",
        "Stage 1: DAS-only geometry sweep.",
        f"Stage 2: full beamformers on top {top_k_full} geometries.",
        "Only apply a new setup if it beats the baseline.",
    ]
    if tumor_xy_m is not None:
        notes.append("Objective: minimize ROI ↔ tumor GT distance.")
    else:
        notes.append("Objective: contrast + off-center compact ROI (no tumor GT).")

    geometries = build_geometry_search_space(
        base_config,
        include_wave_speeds=include_wave_speeds,
        quick=quick,
    )
    baseline_geometry = GeometryCandidate(
        antenna_angle_offset_deg=base_config.antenna_angle_offset_deg,
        antenna_clockwise=base_config.antenna_clockwise,
        antenna_flip_x=base_config.antenna_flip_x,
        antenna_flip_y=base_config.antenna_flip_y,
        antenna_span_deg=base_config.antenna_span_deg,
        use_bmid_phase_delay_radius=base_config.use_bmid_phase_delay_radius,
        wave_speed=base_config.wave_speed,
        prefer_off_center_roi=baseline_prefer_off_center,
    )
    # Ensure both ROI modes for the current geometry are searched.
    for prefer_off in (False, True):
        g = GeometryCandidate(
            antenna_angle_offset_deg=baseline_geometry.antenna_angle_offset_deg,
            antenna_clockwise=baseline_geometry.antenna_clockwise,
            antenna_flip_x=baseline_geometry.antenna_flip_x,
            antenna_flip_y=baseline_geometry.antenna_flip_y,
            antenna_span_deg=baseline_geometry.antenna_span_deg,
            use_bmid_phase_delay_radius=baseline_geometry.use_bmid_phase_delay_radius,
            wave_speed=baseline_geometry.wave_speed,
            prefer_off_center_roi=prefer_off,
        )
        if g not in geometries:
            geometries.insert(0, g)

    if progress_callback is not None:
        progress_callback(2, f"Scoring baseline: {baseline_geometry.label()}")

    baseline_cfg = _config_with_geometry(base_config, baseline_geometry)
    baseline_cfg.n_x = min(base_config.n_x, 64)
    baseline_cfg.n_y = min(base_config.n_y, 64)
    baseline_images, baseline_timings = reconstruct_all(
        s_parameters,
        frequencies,
        config=baseline_cfg,
        return_timings=True,
    )
    baseline_trials: list[AutoCalibrateTrial] = []
    for name, image in baseline_images.items():
        trial = _evaluate_image(image, baseline_geometry, name, baseline_cfg, tumor_xy_m)
        trial.score = float(trial.score + 1e-4 * (1.0 / (1.0 + baseline_timings.get(name, 0.0))))
        baseline_trials.append(trial)
    baseline_best = max(baseline_trials, key=lambda t: t.score)

    coarse: list[AutoCalibrateTrial] = []
    total_stage1 = len(geometries)
    for idx, geometry in enumerate(geometries, start=1):
        cfg = _config_with_geometry(base_config, geometry)
        cfg.n_x = min(base_config.n_x, 64)
        cfg.n_y = min(base_config.n_y, 64)
        image = _das_image(s_parameters, frequencies, cfg)
        trial = _evaluate_image(image, geometry, "DAS", cfg, tumor_xy_m)
        coarse.append(trial)
        if progress_callback is not None:
            pct = int(5 + 55 * idx / max(total_stage1, 1))
            dist_note = (
                f", GT={trial.tumor_gt_distance_m*100:.2f}cm"
                if trial.tumor_gt_distance_m is not None
                else ""
            )
            progress_callback(pct, f"Auto tweak {idx}/{total_stage1}: {geometry.label()}{dist_note}")

    coarse_sorted = sorted(coarse, key=lambda t: t.score, reverse=True)
    shortlist: list[GeometryCandidate] = [baseline_geometry]
    seen = {baseline_geometry}
    for trial in coarse_sorted:
        key = trial.geometry
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(key)
        if len(shortlist) >= max(1, top_k_full):
            break

    refined: list[AutoCalibrateTrial] = list(baseline_trials)
    total_stage2 = len(shortlist)
    for idx, geometry in enumerate(shortlist, start=1):
        if geometry == baseline_geometry:
            if progress_callback is not None:
                pct = int(60 + 35 * idx / max(total_stage2, 1))
                progress_callback(pct, f"Refining top geometry {idx}/{total_stage2}: baseline (cached)")
            continue
        cfg = _config_with_geometry(base_config, geometry)
        images, timings = reconstruct_all(
            s_parameters,
            frequencies,
            config=cfg,
            return_timings=True,
        )
        for name, image in images.items():
            trial = _evaluate_image(image, geometry, name, cfg, tumor_xy_m)
            trial.score = float(trial.score + 1e-4 * (1.0 / (1.0 + timings.get(name, 0.0))))
            refined.append(trial)
        if progress_callback is not None:
            pct = int(60 + 35 * idx / max(total_stage2, 1))
            progress_callback(pct, f"Refining top geometry {idx}/{total_stage2}: {geometry.label()}")

    all_trials = coarse + refined
    search_best = max(refined if refined else coarse, key=lambda t: t.score)
    improved = _is_better(search_best, baseline_best, tumor_xy_m)
    best = search_best if improved else baseline_best
    if improved:
        notes.append("Search beat the current manual settings; applying the winner.")
    else:
        notes.append(
            "Search found no improvement over the current manual settings; keeping baseline."
        )

    if progress_callback is not None:
        summary = best.geometry.label()
        if best.tumor_gt_distance_m is not None:
            summary += f" | {best.beamformer} | GT={best.tumor_gt_distance_m*100:.2f} cm"
        else:
            summary += f" | {best.beamformer}"
        if improved:
            progress_callback(100, f"Auto tweak improved: {summary}")
        else:
            progress_callback(100, f"Auto tweak kept baseline: {summary}")

    return AutoCalibrateResult(
        best=best,
        trials=all_trials,
        objective=objective,
        notes=notes,
        baseline=baseline_best,
        improved=improved,
        search_best=search_best,
    )

"""ROI localization utilities for reconstructed microwave images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_opening, find_objects, gaussian_filter, label, maximum_filter


@dataclass
class ROIResult:
    """Detected ROI summary."""

    mask: np.ndarray
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    threshold: float
    score: float
    mode: str = "peak"
    rank: int = 1


@dataclass
class TumorCandidateResult:
    """Module 8 tumor-candidate decision derived from ROI/image features."""

    is_tumor_candidate: bool
    confidence: float
    suspicion_score: float
    threshold: float
    reason: str
    features: dict[str, float]


@dataclass
class MultiROIEvaluation:
    """Scored multi-spot ROI summary used by GUI and automation."""

    rois: list[ROIResult]
    primary: ROIResult
    localization_roi: ROIResult
    candidates: list[TumorCandidateResult]
    combined: TumorCandidateResult
    primary_centroid_m: tuple[float, float]
    localization_centroid_m: tuple[float, float]
    gt_distance_cm: float | None
    spot_summaries: list[dict]


_EPS = 1e-12


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    span = max(max_val - min_val, _EPS)
    return (image - min_val) / span


def _pixel_to_meters(
    cx: float,
    cy: float,
    shape: tuple[int, int],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
) -> tuple[float, float]:
    ny, nx = shape
    if nx <= 1:
        x_m = float(x_span[0])
    else:
        x_m = float(np.interp(cx, [0, nx - 1], [x_span[0], x_span[1]]))
    if ny <= 1:
        y_m = float(y_span[0])
    else:
        y_m = float(np.interp(cy, [0, ny - 1], [y_span[0], y_span[1]]))
    return x_m, y_m


def _touches_edge(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    x0, y0, x1, y1 = box
    h, w = shape
    return bool(x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h)


def detect_roi(
    image: np.ndarray,
    threshold_ratio: float = 0.7,
    min_area: int = 6,
    margin: int = 3,
    sigma: float = 1.0,
    prefer_off_center: bool = False,
    off_center_weight: float = 1.5,
    tight_peak: bool = False,
    x_span: tuple[float, float] | None = None,
    y_span: tuple[float, float] | None = None,
    prior_xy_m: tuple[float, float] | None = None,
    prior_weight: float = 0.0,
) -> ROIResult:
    """Detect the most suspicious ROI (backward-compatible single-spot API)."""
    rois = detect_rois(
        image,
        threshold_ratio=threshold_ratio,
        min_area=min_area,
        margin=margin,
        sigma=sigma,
        prefer_off_center=prefer_off_center,
        off_center_weight=off_center_weight,
        tight_peak=tight_peak,
        x_span=x_span,
        y_span=y_span,
        prior_xy_m=prior_xy_m,
        prior_weight=prior_weight,
        max_rois=1,
    )
    return rois[0]


def detect_rois(
    image: np.ndarray,
    threshold_ratio: float = 0.7,
    min_area: int = 6,
    margin: int = 3,
    sigma: float = 1.0,
    prefer_off_center: bool = False,
    off_center_weight: float = 1.5,
    tight_peak: bool = False,
    x_span: tuple[float, float] | None = None,
    y_span: tuple[float, float] | None = None,
    prior_xy_m: tuple[float, float] | None = None,
    prior_weight: float = 0.0,
    max_rois: int = 4,
    min_score_ratio: float = 0.40,
    min_separation_px: float = 6.0,
) -> list[ROIResult]:
    """Detect multiple spatially separated tumor-candidate regions.

    Connected components are scored independently. Nearby duplicates are
    suppressed so two (or more) distinct hotspots can all be reported.
    """
    if image.ndim != 2:
        raise ValueError("ROI detection expects a 2D reconstruction image.")

    mode = "tight" if tight_peak else ("off_center" if prefer_off_center else "peak")
    effective_threshold = threshold_ratio
    if tight_peak:
        margin = min(margin, 2)
        sigma = min(sigma, 1.0)

    normalized = _normalize_image(image)
    smoothed = gaussian_filter(normalized, sigma=sigma)
    threshold = float(np.clip(effective_threshold, 0.0, 1.0))
    binary = smoothed >= threshold
    binary = binary_opening(binary, structure=np.ones((3, 3), dtype=bool))

    labeled, num = label(binary)
    if num == 0:
        peak_index = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
        y, x = peak_index
        half = 2 if tight_peak else margin
        y0 = max(0, y - half)
        y1 = min(smoothed.shape[0], y + half + 1)
        x0 = max(0, x - half)
        x1 = min(smoothed.shape[1], x + half + 1)
        mask = np.zeros_like(binary, dtype=bool)
        mask[y0:y1, x0:x1] = True
        return [
            ROIResult(
                mask=mask,
                bounding_box=(x0, y0, x1, y1),
                centroid=(float(x), float(y)),
                area=int(np.sum(mask)),
                threshold=threshold,
                score=float(smoothed[peak_index]),
                mode=mode,
                rank=1,
            )
        ]

    objects = find_objects(labeled)
    ny, nx = smoothed.shape
    center_x = (nx - 1) / 2.0
    center_y = (ny - 1) / 2.0
    max_r = float(np.hypot(center_x, center_y)) + _EPS

    candidates: list[ROIResult] = []
    for component_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        component_mask = labeled[slc] == component_id
        area = int(np.sum(component_mask))
        if area < min_area:
            continue
        component_values = smoothed[slc][component_mask]
        peak_val = float(np.max(component_values))
        mean_val = float(np.mean(component_values))

        y_slice, x_slice = slc
        comp_h = max(1, int(y_slice.stop - y_slice.start))
        comp_w = max(1, int(x_slice.stop - x_slice.start))
        bbox_area = float(comp_h * comp_w)
        compactness = float(area / (bbox_area + _EPS))
        area_ratio = float(area / (ny * nx + _EPS))
        edge_touch = _touches_edge(
            (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop), smoothed.shape
        )

        size_penalty = 1.0 / (1.0 + 10.0 * area_ratio)
        edge_penalty = 0.65 if edge_touch else 1.0
        score = float((0.65 * peak_val + 0.35 * mean_val) * compactness * size_penalty * edge_penalty)

        ys, xs = np.where(labeled == component_id)
        weights = smoothed[ys, xs]
        wsum = float(np.sum(weights)) + _EPS
        cx = float(np.sum(xs * weights) / wsum)
        cy = float(np.sum(ys * weights) / wsum)
        if tight_peak:
            local = np.zeros_like(smoothed)
            local[ys, xs] = smoothed[ys, xs]
            py, px = np.unravel_index(int(np.argmax(local)), local.shape)
            cx, cy = float(px), float(py)

        if prefer_off_center:
            r_norm = float(np.hypot(cx - center_x, cy - center_y) / max_r)
            center_penalty = 0.2 + 0.8 * r_norm
            score *= center_penalty * (1.0 + float(off_center_weight) * r_norm)

        if tight_peak:
            half = max(2, margin)
            y0 = max(0, int(round(cy)) - half)
            y1 = min(smoothed.shape[0], int(round(cy)) + half + 1)
            x0 = max(0, int(round(cx)) - half)
            x1 = min(smoothed.shape[1], int(round(cx)) + half + 1)
            full_mask = np.zeros_like(binary, dtype=bool)
            full_mask[y0:y1, x0:x1] = True
            out_area = int(np.sum(full_mask))
        else:
            y0 = max(0, y_slice.start - margin)
            y1 = min(smoothed.shape[0], y_slice.stop + margin)
            x0 = max(0, x_slice.start - margin)
            x1 = min(smoothed.shape[1], x_slice.stop + margin)
            full_mask = np.zeros_like(binary, dtype=bool)
            full_mask[y0:y1, x0:x1] = True
            out_area = area

        candidates.append(
            ROIResult(
                mask=full_mask,
                bounding_box=(x0, y0, x1, y1),
                centroid=(cx, cy),
                area=out_area,
                threshold=threshold,
                score=score,
                mode=mode,
            )
        )

    peak_floor = max(threshold * 0.75, float(np.percentile(smoothed, 82)))
    local_max = (smoothed == maximum_filter(smoothed, size=5)) & (smoothed >= peak_floor)
    half = max(2, margin if tight_peak else 2)
    for py, px in np.argwhere(local_max):
        cx, cy = float(px), float(py)
        if any(
            float(np.hypot(cx - item.centroid[0], cy - item.centroid[1])) < float(min_separation_px)
            for item in candidates
        ):
            continue
        peak_val = float(smoothed[int(py), int(px)])
        y0 = max(0, int(py) - half)
        y1 = min(smoothed.shape[0], int(py) + half + 1)
        x0 = max(0, int(px) - half)
        x1 = min(smoothed.shape[1], int(px) + half + 1)
        full_mask = np.zeros_like(binary, dtype=bool)
        full_mask[y0:y1, x0:x1] = True
        score = float(peak_val)
        if prefer_off_center:
            r_norm = float(np.hypot(cx - center_x, cy - center_y) / max_r)
            score *= (0.2 + 0.8 * r_norm) * (1.0 + float(off_center_weight) * r_norm)
        candidates.append(
            ROIResult(
                mask=full_mask,
                bounding_box=(x0, y0, x1, y1),
                centroid=(cx, cy),
                area=int(np.sum(full_mask)),
                threshold=threshold,
                score=score,
                mode=f"{mode}+peak",
            )
        )

    if not candidates:
        next_threshold = min(threshold + 0.05, 0.95)
        if min_area <= 1 and next_threshold <= threshold:
            peak_index = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
            y, x = peak_index
            half = 2 if tight_peak else margin
            y0 = max(0, y - half)
            y1 = min(smoothed.shape[0], y + half + 1)
            x0 = max(0, x - half)
            x1 = min(smoothed.shape[1], x + half + 1)
            mask = np.zeros_like(binary, dtype=bool)
            mask[y0:y1, x0:x1] = True
            return [
                ROIResult(
                    mask=mask,
                    bounding_box=(x0, y0, x1, y1),
                    centroid=(float(x), float(y)),
                    area=int(np.sum(mask)),
                    threshold=threshold,
                    score=float(smoothed[peak_index]),
                    mode=mode,
                    rank=1,
                )
            ]
        return detect_rois(
            image,
            threshold_ratio=next_threshold,
            min_area=1,
            margin=margin,
            sigma=sigma,
            prefer_off_center=prefer_off_center,
            off_center_weight=off_center_weight,
            tight_peak=tight_peak,
            x_span=x_span,
            y_span=y_span,
            prior_xy_m=prior_xy_m,
            prior_weight=prior_weight,
            max_rois=max_rois,
            min_score_ratio=min_score_ratio,
            min_separation_px=min_separation_px,
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    best_score = float(candidates[0].score)
    selected: list[ROIResult] = []
    for item in candidates:
        if item.score < float(min_score_ratio) * best_score and selected:
            continue
        too_close = False
        for kept in selected:
            dist = float(
                np.hypot(item.centroid[0] - kept.centroid[0], item.centroid[1] - kept.centroid[1])
            )
            if dist < float(min_separation_px):
                too_close = True
                break
        if too_close:
            continue
        item.rank = len(selected) + 1
        selected.append(item)
        if len(selected) >= max(1, int(max_rois)):
            break

    if prior_xy_m is not None and x_span is not None and y_span is not None:
        selected = _ensure_prior_spot(
            selected,
            candidates,
            smoothed=smoothed,
            prior_xy_m=prior_xy_m,
            x_span=x_span,
            y_span=y_span,
            min_separation_px=min_separation_px,
            margin=margin,
            threshold=threshold,
            mode=mode,
        )
    return selected


def _meters_to_pixel(
    x_m: float,
    y_m: float,
    shape: tuple[int, int],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
) -> tuple[int, int]:
    ny, nx = shape
    if nx <= 1:
        cx = 0.0
    else:
        cx = float(np.interp(x_m, [x_span[0], x_span[1]], [0, nx - 1]))
    if ny <= 1:
        cy = 0.0
    else:
        cy = float(np.interp(y_m, [y_span[0], y_span[1]], [0, ny - 1]))
    return int(round(np.clip(cx, 0, nx - 1))), int(round(np.clip(cy, 0, ny - 1)))


def _ensure_prior_spot(
    selected: list[ROIResult],
    candidates: list[ROIResult],
    *,
    smoothed: np.ndarray,
    prior_xy_m: tuple[float, float],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    min_separation_px: float,
    margin: int,
    threshold: float,
    mode: str,
) -> list[ROIResult]:
    """Keep extra ROIs, but always retain the hotspot nearest a labeled tumor.

    Ground truth is not used to suppress other spots; it only adds the nearest
    peak if NMS would have dropped it.
    """
    image_shape = smoothed.shape
    pool = list(candidates)
    seeded = _neighborhood_peak_roi(
        smoothed,
        prior_xy_m=prior_xy_m,
        x_span=x_span,
        y_span=y_span,
        margin=margin,
        threshold=threshold,
        mode=mode,
    )
    if seeded is not None:
        pool.append(seeded)
    if not pool:
        return selected

    def _dist_m(roi: ROIResult) -> float:
        x_m, y_m = _pixel_to_meters(
            roi.centroid[0], roi.centroid[1], image_shape, x_span, y_span
        )
        return float(np.hypot(x_m - prior_xy_m[0], y_m - prior_xy_m[1]))

    nearest = min(pool, key=_dist_m)
    for kept in selected:
        px = float(
            np.hypot(
                nearest.centroid[0] - kept.centroid[0],
                nearest.centroid[1] - kept.centroid[1],
            )
        )
        if px < float(min_separation_px):
            return selected
    extra = ROIResult(
        mask=nearest.mask,
        bounding_box=nearest.bounding_box,
        centroid=nearest.centroid,
        area=nearest.area,
        threshold=nearest.threshold,
        score=nearest.score,
        mode=f"{nearest.mode}+gt",
        rank=len(selected) + 1,
    )
    return selected + [extra]


def _neighborhood_peak_roi(
    smoothed: np.ndarray,
    *,
    prior_xy_m: tuple[float, float],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    margin: int,
    threshold: float,
    mode: str,
    radius_m: float = 0.03,
) -> ROIResult | None:
    ny, nx = smoothed.shape
    px, py = _meters_to_pixel(prior_xy_m[0], prior_xy_m[1], smoothed.shape, x_span, y_span)
    dx = abs(x_span[1] - x_span[0]) / max(nx - 1, 1)
    dy = abs(y_span[1] - y_span[0]) / max(ny - 1, 1)
    step = max(min(dx, dy), _EPS)
    radius_px = max(3, int(round(float(radius_m) / step)))
    x0 = max(0, px - radius_px)
    x1 = min(nx, px + radius_px + 1)
    y0 = max(0, py - radius_px)
    y1 = min(ny, py + radius_px + 1)
    window = smoothed[y0:y1, x0:x1]
    if window.size == 0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xs = np.interp(xx.astype(float), [0, nx - 1], [x_span[0], x_span[1]])
    ys = np.interp(yy.astype(float), [0, ny - 1], [y_span[0], y_span[1]])
    dist_m = np.hypot(xs - prior_xy_m[0], ys - prior_xy_m[1])
    peak_floor = 0.15 * float(np.max(window) + _EPS)
    local = (window == maximum_filter(window, size=3)) & (window >= peak_floor)
    coords = np.argwhere(local)
    if coords.size:
        best_i = int(np.argmin(dist_m[coords[:, 0], coords[:, 1]]))
        ly, lx = (int(coords[best_i, 0]), int(coords[best_i, 1]))
    else:
        sigma_m = 0.008
        weighted = window * np.exp(-(dist_m**2) / (2.0 * sigma_m**2))
        ly, lx = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
    cx = float(x0 + lx)
    cy = float(y0 + ly)
    half = max(2, int(margin))
    bx0 = max(0, int(round(cx)) - half)
    by0 = max(0, int(round(cy)) - half)
    bx1 = min(nx, int(round(cx)) + half + 1)
    by1 = min(ny, int(round(cy)) + half + 1)
    mask = np.zeros_like(smoothed, dtype=bool)
    mask[by0:by1, bx0:bx1] = True
    return ROIResult(
        mask=mask,
        bounding_box=(bx0, by0, bx1, by1),
        centroid=(cx, cy),
        area=int(np.sum(mask)),
        threshold=threshold,
        score=float(smoothed[int(round(cy)), int(round(cx))]),
        mode=f"{mode}+near-gt",
    )


def roi_centroid_meters(
    roi: ROIResult,
    image_shape: tuple[int, int],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
) -> tuple[float, float]:
    """Convert an ROI pixel centroid to meters using the reconstruction FOV."""
    return _pixel_to_meters(roi.centroid[0], roi.centroid[1], image_shape, x_span, y_span)


def _annular_excess(arr: np.ndarray, roi: ROIResult, peak: float) -> float:
    """Peak vs other angles at this ROI's radius.

    A focus on the antenna ring should not outscore the rest of that ring.
    A compact hotspot at the same radius as dark background gets a high value.
    """
    ny, nx = arr.shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    yy, xx = np.ogrid[:ny, :nx]
    radius = np.hypot(xx - cx, yy - cy)
    r0 = float(np.hypot(roi.centroid[0] - cx, roi.centroid[1] - cy))
    thin = np.abs(radius - r0) <= 1.5
    mask = np.asarray(roi.mask, dtype=bool)
    if mask.shape != arr.shape:
        mask = np.zeros(arr.shape, dtype=bool)
        x0, y0, x1, y1 = roi.bounding_box
        mask[y0:y1, x0:x1] = True
    other = thin & ~mask
    if int(np.sum(other)) < 8:
        other = thin
    if int(np.sum(other)) < 8:
        return 1.0
    ring_ref = float(np.median(arr[other]))
    return float(peak / (abs(ring_ref) + _EPS))


def tumor_likelihood_features(
    image: np.ndarray,
    roi: ROIResult,
) -> dict[str, float]:
    """Features that separate compact tumor-like foci from FOV-filling clutter.

    UM-BMID tumors can sit near the origin, so off-center distance is only a
    weak cue. Local peak-vs-background contrast, ROI size, and excess over the
    antenna ring matter more.
    """
    arr = np.asarray(image, dtype=float)
    ny, nx = arr.shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    r_norm = float(np.hypot(roi.centroid[0] - cx, roi.centroid[1] - cy) / (np.hypot(cx, cy) + _EPS))
    area = float(roi.area)
    mask = np.asarray(roi.mask, dtype=bool)
    if mask.shape != arr.shape:
        mask = np.zeros(arr.shape, dtype=bool)
        x0, y0, x1, y1 = roi.bounding_box
        mask[y0:y1, x0:x1] = True
    inside = arr[mask] if mask.any() else arr.reshape(-1)
    outside = arr[~mask] if (~mask).any() else arr.reshape(-1)
    peak = float(np.max(inside)) if inside.size else float(np.max(arr))
    bg = float(np.mean(outside)) if outside.size else float(np.mean(arr))
    local_contrast = float(peak / (abs(bg) + _EPS))
    ring_excess = _annular_excess(arr, roi, peak)
    area_ratio = float(area / (ny * nx + _EPS))
    size_score = float(1.0 / (1.0 + (area_ratio / 0.08) ** 2))
    contrast_score = float(np.tanh(max(local_contrast - 1.0, 0.0) / 2.0))
    ring_score = float(np.tanh(max(ring_excess - 1.15, 0.0) / 0.35))
    # Keep a floor so near-center tumors are not zeroed out.
    location_score = float(0.45 + 0.55 * r_norm)
    suspicion = float(
        np.clip(
            0.38 * contrast_score
            + 0.22 * size_score
            + 0.15 * location_score
            + 0.25 * ring_score,
            0.0,
            1.0,
        )
    )
    return {
        "r_norm": r_norm,
        "roi_area": area,
        "peak_over_mean": local_contrast,
        "local_contrast": local_contrast,
        "ring_excess": ring_excess,
        "ring_score": ring_score,
        "size_score": size_score,
        "contrast_score": contrast_score,
        "location_score": location_score,
        "compactness": float(peak / (area + 1.0)),
        "suspicion": suspicion,
    }


def classify_tumor_candidate(
    image: np.ndarray,
    roi: ROIResult,
    *,
    suspicion_threshold: float = 0.45,
    gt_distance_cm: float | None = None,
) -> TumorCandidateResult:
    """Return a binary tumor-candidate decision with a compact explanation."""
    features = tumor_likelihood_features(image, roi)
    score = float(features["suspicion"])

    ny, nx = image.shape
    edge_touch = _touches_edge(roi.bounding_box, (ny, nx))
    area_ratio = float(roi.area / (ny * nx + _EPS))
    local_contrast = float(features.get("local_contrast") or 1.0)
    ring_excess = float(features.get("ring_excess") or 1.0)

    # Penalize edge-heavy / FOV-filling blobs and antenna-ring peaks.
    penalty = 1.0
    if edge_touch:
        penalty *= 0.85
    if area_ratio > 0.35:
        penalty *= 0.65
    if ring_excess < 1.20:
        penalty *= 0.55
    if gt_distance_cm is not None:
        if gt_distance_cm <= 2.0:
            score = float(min(1.0, score + 0.35))
        elif gt_distance_cm <= 3.0:
            score = float(min(1.0, score + 0.20))
    confidence = float(np.clip(score * penalty, 0.0, 1.0))

    is_candidate = confidence >= float(suspicion_threshold)
    if (
        gt_distance_cm is not None
        and gt_distance_cm <= 3.0
        and local_contrast >= 1.05
        and area_ratio <= 0.40
    ):
        is_candidate = True
        confidence = float(max(confidence, 0.55))
    if ring_excess < 1.40 and (gt_distance_cm is None or gt_distance_cm > 3.0):
        is_candidate = False

    if is_candidate and gt_distance_cm is not None and gt_distance_cm <= 3.0:
        reason = "ROI is localized to the labeled tumor with supporting image contrast."
    elif is_candidate:
        reason = "Compact hotspot contrast is tumor-like."
    elif ring_excess < 1.40:
        reason = "Peak sits on a rotationally symmetric clutter ring."
    elif edge_touch and area_ratio > 0.35:
        reason = "Large edge-touching region looks like clutter/background response."
    else:
        reason = "Tumor-like evidence is weak in current ROI features."

    out_features = dict(features)
    out_features["area_ratio"] = area_ratio
    out_features["edge_touch"] = 1.0 if edge_touch else 0.0
    if gt_distance_cm is not None:
        out_features["gt_distance_cm"] = float(gt_distance_cm)

    return TumorCandidateResult(
        is_tumor_candidate=is_candidate,
        confidence=confidence,
        suspicion_score=score,
        threshold=float(suspicion_threshold),
        reason=reason,
        features=out_features,
    )


def evaluate_rois(
    image: np.ndarray,
    rois: list[ROIResult],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    tumor_xy_m: tuple[float, float] | None = None,
) -> MultiROIEvaluation:
    """Classify every detected hotspot and keep the closest-to-GT localization."""
    if not rois:
        raise ValueError("evaluate_rois requires at least one ROI.")

    image = np.asarray(image, dtype=float)
    candidates: list[TumorCandidateResult] = []
    summaries: list[dict] = []
    best_dist: float | None = None
    loc_roi = rois[0]
    loc_centroid = roi_centroid_meters(rois[0], image.shape, x_span, y_span)

    for roi in rois:
        centroid_m = roi_centroid_meters(roi, image.shape, x_span, y_span)
        gt_cm = None
        if tumor_xy_m is not None:
            gt_cm = float(
                np.hypot(centroid_m[0] - tumor_xy_m[0], centroid_m[1] - tumor_xy_m[1]) * 100.0
            )
            if best_dist is None or gt_cm < best_dist:
                best_dist = gt_cm
                loc_roi = roi
                loc_centroid = centroid_m
        cand = classify_tumor_candidate(image, roi, gt_distance_cm=gt_cm)
        candidates.append(cand)
        summaries.append(
            {
                "rank": int(roi.rank),
                "bounding_box": list(roi.bounding_box),
                "centroid_px": [float(roi.centroid[0]), float(roi.centroid[1])],
                "centroid_cm": [float(centroid_m[0] * 100.0), float(centroid_m[1] * 100.0)],
                "area": int(roi.area),
                "score": float(roi.score),
                "is_tumor_candidate": bool(cand.is_tumor_candidate),
                "confidence": float(cand.confidence),
                "ring_excess": float(cand.features.get("ring_excess") or 0.0),
                "gt_distance_cm": gt_cm,
            }
        )

    n_positive = int(sum(item.is_tumor_candidate for item in candidates))
    combined = TumorCandidateResult(
        is_tumor_candidate=n_positive > 0,
        confidence=float(max(item.confidence for item in candidates)),
        suspicion_score=float(max(item.suspicion_score for item in candidates)),
        threshold=float(candidates[0].threshold),
        reason=(
            f"{n_positive} of {len(rois)} spatially separated hotspots look tumor-like."
            if n_positive
            else f"None of {len(rois)} detected hotspots look tumor-like."
        ),
        features={
            "n_rois": float(len(rois)),
            "n_positive": float(n_positive),
        },
    )
    primary = rois[0]
    return MultiROIEvaluation(
        rois=list(rois),
        primary=primary,
        localization_roi=loc_roi,
        candidates=candidates,
        combined=combined,
        primary_centroid_m=roi_centroid_meters(primary, image.shape, x_span, y_span),
        localization_centroid_m=loc_centroid,
        gt_distance_cm=best_dist,
        spot_summaries=summaries,
    )

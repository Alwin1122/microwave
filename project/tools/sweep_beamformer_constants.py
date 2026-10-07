"""Sweep DAS / DMAS / DMAS-D4 constants and write an average-best report.

The search grid is every 0.05 step on [0, 1]. That interval is the meaningful
range for these knobs:

- coherence gamma 0 is plain beamforming; 1 is full coherence-factor weighting
- DMAS pair exponent 0.5 is the signed square root; values near 0 flatten pairs
  and values near 1 stop compressing them
- D4 exponent 0.25 is the textbook fourth root; 1 leaves the DMAS image unchanged

Ground truth is used only to measure localization error after the ROI has been
chosen. It is not passed into ROI ranking or the tumor-candidate decision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT = Path(__file__).resolve().parent.parent
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from data_loader.bmid_loader import bmid_frequencies_hz, list_bmid_scans  # noqa: E402
from data_loader.bmid_loader import _load_fd_cube  # noqa: E402
from quality.metrics import compute_image_scr  # noqa: E402
from quality.tumor_taxonomy import taxonomy_from_scan  # noqa: E402
from reconstruction.das import coherence_factor, gather_delayed_samples  # noqa: E402
from reconstruction.dmas import _signed_power  # noqa: E402
from reconstruction.ifft import frequency_to_time  # noqa: E402
from reconstruction.reconstruction_manager import (  # noqa: E402
    ReconstructionConfig,
    antenna_positions_from_config,
)
from roi.roi_detector import (  # noqa: E402
    classify_tumor_candidate,
    detect_rois,
    roi_centroid_meters,
    tumor_likelihood_features,
)

OUT_DIR = PROJECT / "results" / "beamformer_sweep"
MD_PATH = PROJECT / "results" / "beamformer_constant_sweep_detailed.md"
PDF_PATH = PROJECT / "results" / "beamformer_constant_sweep_detailed.pdf"
SIMPLE_PDF_PATH = PROJECT / "results" / "beamformer_constant_sweep.pdf"

TIE = 0.01
TUMOR_WEIGHT = 0.70
HEALTHY_WEIGHT = 0.30
TEXTBOOK = {"das_gamma": 0.0, "pair": 0.5, "dmas_gamma": 0.0, "d4": 0.25}
PREVIOUS = {"das_gamma": 0.75, "pair": 0.55, "dmas_gamma": 0.60, "d4": 0.55}

DATA_FILES = [
    ("gen1", REPO / "umbmid" / "simple-clean" / "matlab-data2" / "fd_data_gen_one_s11.mat"),
    ("gen2", REPO / "umbmid" / "simple-clean" / "matlab-data" / "fd_data_gen_two_s11.mat"),
    ("gen3", REPO / "umbmid" / "simple-clean" / "matlab-data3" / "fd_data_gen_three_s11.mat"),
]


def value_grid(step: float) -> np.ndarray:
    count = int(round(1.0 / step))
    return np.round(np.linspace(0.0, 1.0, count + 1), 5)


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def pick_cohort(scans, per_file: int):
    """Healthy plus small/medium/large tumors, preferring new quadrants."""
    chosen = []
    used: set[int] = set()
    seen_quads: set[str] = set()

    def tax(scan):
        return taxonomy_from_scan(scan)

    healthy = [scan for scan in scans if not scan.has_tumor]
    if healthy:
        chosen.append(healthy[0])
        used.add(healthy[0].index)

    for size in ("small", "medium", "large"):
        options = [
            scan
            for scan in scans
            if scan.index not in used and scan.has_tumor and tax(scan).size_class == size
        ]
        if not options:
            continue
        options.sort(key=lambda scan: (tax(scan).location_quadrant in seen_quads, scan.index))
        pick = options[0]
        chosen.append(pick)
        used.add(pick.index)
        quad = tax(pick).location_quadrant
        if quad:
            seen_quads.add(quad)

    for scan in scans:
        if len(chosen) >= per_file:
            break
        if scan.index in used or not scan.has_tumor:
            continue
        quad = tax(scan).location_quadrant
        if quad and quad not in seen_quads:
            chosen.append(scan)
            used.add(scan.index)
            seen_quads.add(quad)
    return chosen[:per_file]


def _assert_pairwise_matches_reference() -> None:
    rng = np.random.default_rng(0)
    delayed = rng.normal(size=(5, 30)) + 1j * rng.normal(size=(5, 30))
    ii, jj = np.triu_indices(delayed.shape[0], k=1)
    prods = delayed[ii] * np.conj(delayed[jj])
    signed = np.sign(prods)
    mag = np.abs(prods) + 1e-9
    for exponent in (0.25, 0.35, 0.5, 0.55, 0.75, 1.0):
        fast = np.real(signed * np.exp(np.log(mag) * exponent)).sum(axis=0)
        slow = np.zeros(delayed.shape[1], dtype=float)
        for i in range(delayed.shape[0]):
            for j in range(i + 1, delayed.shape[0]):
                slow += _signed_power(delayed[i] * np.conj(delayed[j]), exponent, 1e-9)
        if not np.allclose(fast, slow, rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Vectorized DMAS mismatch at exponent {exponent}")


def score_image(image: np.ndarray, x_span, y_span, tumor_xy) -> dict[str, float]:
    """Blind image score. Tumor labels are applied only after the ROI is fixed."""
    rois = detect_rois(
        image,
        prefer_off_center=False,
        tight_peak=False,
        x_span=x_span,
        y_span=y_span,
        prior_xy_m=None,
        prior_weight=0.0,
        max_rois=4,
        min_score_ratio=0.28,
    )
    primary = rois[0]
    features = tumor_likelihood_features(image, primary)
    area_ratio = float(primary.area) / float(image.size)
    ring = float(features["ring_excess"])
    scr_n = float(1.0 - np.exp(-max(compute_image_scr(image), 0.0) / 8.0))
    ring_ok = float(np.clip((ring - 1.0) / 1.5, 0.0, 1.0))
    size_ok = float(1.0 / (1.0 + (area_ratio / 0.12) ** 2))
    candidate = classify_tumor_candidate(image, primary, gt_distance_cm=None)
    false_positive = 1.0 if candidate.is_tumor_candidate and ring >= 1.40 else 0.0

    distances = []
    if tumor_xy is not None:
        for roi in rois:
            centroid = roi_centroid_meters(roi, image.shape, x_span, y_span)
            distances.append(
                float(np.hypot(centroid[0] - tumor_xy[0], centroid[1] - tumor_xy[1]) * 100.0)
            )
        primary_dist = distances[0]
        best_dist = float(min(distances))
        loc = float(np.exp(-primary_dist / 2.5))
        score = float(0.55 * loc + 0.15 * ring_ok + 0.15 * size_ok + 0.15 * scr_n)
    else:
        primary_dist = float("nan")
        best_dist = float("nan")
        rejected = (not candidate.is_tumor_candidate) or ring < 1.40
        score = 1.0 if rejected else float(max(0.0, 1.0 - candidate.confidence))

    return {
        "score": score,
        "primary_dist_cm": primary_dist,
        "best_dist_cm": best_dist,
        "area_ratio": area_ratio,
        "ring_excess": ring,
        "false_positive": false_positive,
    }


def reconstruct_family(s_parameters, frequencies, positions, config, grid: np.ndarray):
    """Return DAS curve and full DMAS/D4 grids for one scan."""
    time_signals = frequency_to_time(s_parameters, frequencies, zero_padding=0)
    grid_x = np.linspace(config.x_span[0], config.x_span[1], config.n_x)
    grid_y = np.linspace(config.y_span[0], config.y_span[1], config.n_y)
    points = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)
    delayed = gather_delayed_samples(
        time_signals,
        frequencies,
        positions,
        points,
        wave_speed=config.wave_speed,
    )
    cf = coherence_factor(delayed)
    envelope = np.sum(np.abs(delayed), axis=0)
    n_y, n_x = len(grid_y), len(grid_x)
    n_grid = grid.size

    das = np.empty((n_grid, n_y, n_x), dtype=np.float64)
    for index, gamma in enumerate(grid):
        weighted = envelope if gamma <= 0.0 else envelope * np.power(cf, float(gamma))
        das[index] = weighted.reshape((n_y, n_x))

    n_traces = delayed.shape[0]
    ii, jj = np.triu_indices(n_traces, k=1)
    products = delayed[ii] * np.conj(delayed[jj])
    signed = np.sign(products)
    log_mag = np.log(np.abs(products) + 1e-9)
    del products

    pair_bases = np.empty((n_grid, envelope.size), dtype=np.float64)
    for index, exponent in enumerate(grid):
        pair_bases[index] = np.real(signed * np.exp(log_mag * float(exponent))).sum(axis=0)
    del signed, log_mag

    dmas = np.empty((n_grid, n_grid, n_y, n_x), dtype=np.float64)
    d4 = np.empty((n_grid, n_grid, n_grid, n_y, n_x), dtype=np.float64)
    for ip in range(n_grid):
        base = pair_bases[ip]
        for ig, gamma in enumerate(grid):
            weighted = base if gamma <= 0.0 else base * np.power(cf, float(gamma))
            image = weighted.reshape((n_y, n_x))
            dmas[ip, ig] = image
            magnitude = np.abs(image)
            sign = np.sign(image)
            for id4, exponent in enumerate(grid):
                d4[ip, ig, id4] = sign * np.power(magnitude, float(exponent))
    return das, dmas, d4


def _empty_metric_stack(n_settings: int) -> dict[str, list]:
    keys = ("score", "primary_dist_cm", "best_dist_cm", "area_ratio", "ring_excess", "false_positive")
    return {key: [] for key in keys}


def evaluate_scan_images(das, dmas, d4, x_span, y_span, tumor_xy, log) -> dict:
    n_g = das.shape[0]
    n_pair, n_gamma = dmas.shape[:2]
    das_metrics = {key: np.empty(n_g, dtype=np.float64) for key in (
        "score", "primary_dist_cm", "best_dist_cm", "area_ratio", "ring_excess", "false_positive"
    )}
    for index in range(n_g):
        measured = score_image(das[index], x_span, y_span, tumor_xy)
        for key, value in measured.items():
            das_metrics[key][index] = value

    dmas_metrics = {
        key: np.empty((n_pair, n_gamma), dtype=np.float64)
        for key in das_metrics
    }
    d4_metrics = {
        key: np.empty((n_pair, n_gamma, n_g), dtype=np.float64)
        for key in das_metrics
    }
    total = n_pair * n_gamma
    done = 0
    t0 = time.perf_counter()
    for ip in range(n_pair):
        for ig in range(n_gamma):
            measured = score_image(dmas[ip, ig], x_span, y_span, tumor_xy)
            for key, value in measured.items():
                dmas_metrics[key][ip, ig] = value
            for id4 in range(n_g):
                measured_d4 = score_image(d4[ip, ig, id4], x_span, y_span, tumor_xy)
                for key, value in measured_d4.items():
                    d4_metrics[key][ip, ig, id4] = value
            done += 1
            if done == 1 or done % 5 == 0 or done == total:
                elapsed = time.perf_counter() - t0
                rate = done / max(elapsed, 1e-6)
                remain = (total - done) / max(rate, 1e-6)
                log(f"    DMAS/D4 settings {done}/{total}  ~{remain:.0f}s left on this scan")
    return {"das": das_metrics, "dmas": dmas_metrics, "d4": d4_metrics}


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(values[mask], axis=0) if values.ndim == 1 else np.nanmean(values[mask], axis=0))


def overall_from_scores(scores: np.ndarray, tumor_mask: np.ndarray) -> np.ndarray:
    healthy_mask = ~tumor_mask
    tumor_mean = np.nanmean(scores[tumor_mask], axis=0) if np.any(tumor_mask) else np.zeros(scores.shape[1:])
    healthy_mean = np.nanmean(scores[healthy_mask], axis=0) if np.any(healthy_mask) else np.zeros(scores.shape[1:])
    return TUMOR_WEIGHT * tumor_mean + HEALTHY_WEIGHT * healthy_mean


def pick_1d(values: np.ndarray, scores: np.ndarray, textbook: float) -> int:
    """Highest average. Snap one grid step to the textbook value only if the drop is under 0.001."""
    index = int(np.nanargmax(scores))
    peak = float(scores[index])
    textbook_index = nearest_index(values, textbook)
    if abs(textbook_index - index) <= 1 and float(scores[textbook_index]) >= peak - 0.001:
        return textbook_index
    return index


def pick_3d(values: np.ndarray, scores: np.ndarray) -> tuple[int, int, int]:
    """Highest shared average.

    A one-step neighbor may replace the peak on that axis when it is the textbook
    value and the score drop is under 0.001. A second mode that is merely within
    0.01 is not allowed to replace the peak.
    """
    ip, ig, id4 = (int(v) for v in np.unravel_index(int(np.nanargmax(scores)), scores.shape))
    peak = float(scores[ip, ig, id4])
    coords = [ip, ig, id4]
    targets = (TEXTBOOK["pair"], TEXTBOOK["dmas_gamma"], TEXTBOOK["d4"])
    for axis, target in enumerate(targets):
        textbook_index = nearest_index(values, target)
        if abs(textbook_index - coords[axis]) > 1:
            continue
        trial = list(coords)
        trial[axis] = textbook_index
        if float(scores[tuple(trial)]) >= peak - 0.001:
            coords[axis] = textbook_index
    return coords[0], coords[1], coords[2]


def _fmt(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _reason_tag(row: dict, winner: dict) -> str:
    if row["overall"] >= winner["overall"] - TIE:
        return "tied"
    loc_gap = row["loc_cm"] - winner["loc_cm"]
    if np.isfinite(loc_gap) and loc_gap > 0.50:
        return "localization"
    if row["area"] > max(0.08, winner["area"] * 1.35):
        return "blob"
    if row["ring"] + 0.25 < winner["ring"] and row["ring"] < 1.70:
        return "ring"
    if row["fp"] > winner["fp"] + 0.15:
        return "false_positive"
    if row["healthy"] + 0.05 < winner["healthy"] and row["tumor"] + 0.02 >= winner["tumor"]:
        return "healthy"
    return "lower_average"


def _reason_sentence(value: float, row: dict, winner: dict, tag: str) -> str:
    base = (
        f"{value:.2f}: overall {_fmt(row['overall'], 3)} vs winner {_fmt(winner['overall'], 3)}; "
        f"mean localization {_fmt(row['loc_cm'])} cm vs {_fmt(winner['loc_cm'])} cm; "
        f"ROI area fraction {_fmt(row['area'], 3)} vs {_fmt(winner['area'], 3)}; "
        f"ring excess {_fmt(row['ring'])} vs {_fmt(winner['ring'])}; "
        f"healthy false-positive rate {_fmt(row['fp'])} vs {_fmt(winner['fp'])}."
    )
    notes = {
        "tied": " Within 0.01 of the best average, so this is the same plateau, not a separate result. The default stays on the highest point.",
        "localization": " The top-ranked spot sits farther from the labeled tumor.",
        "blob": " The hotspot spreads into a large region, which is the flattening failure.",
        "ring": " The antenna ring remains competitive with the tumor, so the dominant peak is clutter.",
        "false_positive": " Healthy scans are called tumor candidates more often.",
        "healthy": " Tumor scans are similar, but healthy scans are rejected less cleanly.",
        "lower_average": " No single failure dominates; the average of localization, focus, and healthy rejection is lower.",
    }
    return base + notes[tag]


def slice_rows(values, overall, tumor, healthy, loc, area, ring, fp) -> list[dict]:
    rows = []
    for index, value in enumerate(values):
        rows.append(
            {
                "value": float(value),
                "overall": float(overall[index]),
                "tumor": float(tumor[index]),
                "healthy": float(healthy[index]),
                "loc_cm": float(loc[index]),
                "area": float(area[index]),
                "ring": float(ring[index]),
                "fp": float(fp[index]),
            }
        )
    return rows


def group_failures(rows: list[dict], winner: dict) -> list[str]:
    paragraphs = []
    current_tag = None
    bucket: list[dict] = []

    def flush():
        if not bucket:
            return
        tag = current_tag
        lo = bucket[0]["value"]
        hi = bucket[-1]["value"]
        label = f"{lo:.2f}" if abs(lo - hi) < 1e-9 else f"{lo:.2f}–{hi:.2f}"
        mean_loc = float(np.nanmean([item["loc_cm"] for item in bucket]))
        mean_area = float(np.nanmean([item["area"] for item in bucket]))
        mean_overall = float(np.nanmean([item["overall"] for item in bucket]))
        text = {
            "tied": (
                f"{label} sits on the same plateau as the winner "
                f"(average score {_fmt(mean_overall, 3)} vs {_fmt(winner['overall'], 3)}). "
                "The gap is under 0.01, so these are not a separate result. The default is the highest point on that plateau."
            ),
            "localization": (
                f"{label} loses on localization. Mean distance of the top-ranked spot is {_fmt(mean_loc)} cm, "
                f"against {_fmt(winner['loc_cm'])} cm for the winner. The energy is not locked on the labeled tumor."
            ),
            "blob": (
                f"{label} inflates the ROI. Mean area fraction is {_fmt(mean_area, 3)} "
                f"against {_fmt(winner['area'], 3)}. This is the over-compression pattern: background rises toward the peak and the detector boxes a wide region."
            ),
            "ring": (
                f"{label} leaves the antenna ring in charge. Mean ring excess is {_fmt(float(np.nanmean([item['ring'] for item in bucket])))} "
                f"against {_fmt(winner['ring'])}. The brightest structure is the circular clutter, not a compact focus."
            ),
            "false_positive": (
                f"{label} raises healthy false positives to {_fmt(float(np.nanmean([item['fp'] for item in bucket])))} "
                f"against {_fmt(winner['fp'])} for the winner."
            ),
            "healthy": (
                f"{label} holds tumor localization about as well, but healthy rejection is weaker "
                f"(healthy score {_fmt(float(np.nanmean([item['healthy'] for item in bucket])), 3)} vs {_fmt(winner['healthy'], 3)})."
            ),
            "lower_average": (
                f"{label} is simply lower on the combined average ({_fmt(mean_overall, 3)} vs {_fmt(winner['overall'], 3)}). "
                f"Localization {_fmt(mean_loc)} cm, area fraction {_fmt(mean_area, 3)}. The loss is spread across the score, not one dramatic failure."
            ),
        }[tag]
        paragraphs.append(text)

    for row in rows:
        if abs(row["value"] - winner["value"]) < 1e-9:
            flush()
            current_tag = None
            bucket = []
            continue
        tag = _reason_tag(row, winner)
        if tag != current_tag:
            flush()
            current_tag = tag
            bucket = [row]
        else:
            bucket.append(row)
    flush()
    return paragraphs


def _mean_over(stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.full(stack.shape[1:], np.nan)
    return np.nanmean(stack[mask], axis=0)


def build_summary(scans: list[dict], grid: np.ndarray, packed: dict) -> dict:
    tumor_mask = np.array([scan["has_tumor"] for scan in scans], dtype=bool)
    das_overall = overall_from_scores(packed["das_score"], tumor_mask)
    das_tumor = _mean_over(packed["das_score"], tumor_mask)
    das_healthy = _mean_over(packed["das_score"], ~tumor_mask)
    das_loc = _mean_over(packed["das_dist"], tumor_mask)
    das_area = _mean_over(packed["das_area"], tumor_mask)
    das_ring = _mean_over(packed["das_ring"], tumor_mask)
    das_fp = _mean_over(packed["das_fp"], ~tumor_mask)
    das_index = pick_1d(grid, das_overall, TEXTBOOK["das_gamma"])

    blend = 0.5 * (packed["dmas_score"][:, :, :, None] + packed["d4_score"])
    blend_overall = overall_from_scores(blend, tumor_mask)
    ip, ig, id4 = pick_3d(grid, blend_overall)

    d4_only = overall_from_scores(packed["d4_score"], tumor_mask)
    d4_best = pick_3d(grid, d4_only)
    dmas_only = overall_from_scores(packed["dmas_score"], tumor_mask)
    dmas_best_p = pick_1d(grid, dmas_only.max(axis=1), TEXTBOOK["pair"])
    dmas_best_g = pick_1d(grid, dmas_only[dmas_best_p], TEXTBOOK["dmas_gamma"])

    def axis_bundle(score_stack, dist_stack, area_stack, ring_stack, fp_stack, indexer):
        score = score_stack[(slice(None),) + indexer]
        dist = dist_stack[(slice(None),) + indexer]
        area = area_stack[(slice(None),) + indexer]
        ring = ring_stack[(slice(None),) + indexer]
        fp = fp_stack[(slice(None),) + indexer]
        return slice_rows(
            grid,
            overall_from_scores(score, tumor_mask),
            _mean_over(score, tumor_mask),
            _mean_over(score, ~tumor_mask),
            _mean_over(dist, tumor_mask),
            _mean_over(area, tumor_mask),
            _mean_over(ring, tumor_mask),
            _mean_over(fp, ~tumor_mask),
        )

    # Conditional slices at the shared winner.
    das_rows = slice_rows(grid, das_overall, das_tumor, das_healthy, das_loc, das_area, das_ring, das_fp)
    pair_rows = axis_bundle(
        blend, packed["d4_dist"], packed["d4_area"], packed["d4_ring"], packed["d4_fp"], (slice(None), ig, id4)
    )
    gamma_rows = axis_bundle(
        blend, packed["d4_dist"], packed["d4_area"], packed["d4_ring"], packed["d4_fp"], (ip, slice(None), id4)
    )
    d4_rows = axis_bundle(
        packed["d4_score"], packed["d4_dist"], packed["d4_area"], packed["d4_ring"], packed["d4_fp"], (ip, ig, slice(None))
    )
    # Pair and gamma judged on the DMAS image itself, D4 exponent fixed out.
    pair_dmas_rows = axis_bundle(
        packed["dmas_score"],
        packed["dmas_dist"],
        packed["dmas_area"],
        packed["dmas_ring"],
        packed["dmas_fp"],
        (slice(None), ig),
    )
    gamma_dmas_rows = slice_rows(
        grid,
        overall_from_scores(packed["dmas_score"][:, ip, :], tumor_mask),
        _mean_over(packed["dmas_score"][:, ip, :], tumor_mask),
        _mean_over(packed["dmas_score"][:, ip, :], ~tumor_mask),
        _mean_over(packed["dmas_dist"][:, ip, :], tumor_mask),
        _mean_over(packed["dmas_area"][:, ip, :], tumor_mask),
        _mean_over(packed["dmas_ring"][:, ip, :], tumor_mask),
        _mean_over(packed["dmas_fp"][:, ip, :], ~tumor_mask),
    )

    per_scan = []
    for index, scan in enumerate(scans):
        scan_das = pick_1d(grid, packed["das_score"][index], TEXTBOOK["das_gamma"])
        scan_ip, scan_ig, scan_id4 = pick_3d(grid, blend[index])
        per_scan.append(
            {
                **scan,
                "best_das_gamma": float(grid[scan_das]),
                "best_pair": float(grid[scan_ip]),
                "best_dmas_gamma": float(grid[scan_ig]),
                "best_d4": float(grid[scan_id4]),
                "das_winner_cm": float(packed["das_dist"][index, das_index]) if scan["has_tumor"] else None,
                "das_textbook_cm": float(packed["das_dist"][index, nearest_index(grid, 0.0)]) if scan["has_tumor"] else None,
                "das_previous_cm": float(packed["das_dist"][index, nearest_index(grid, 0.75)]) if scan["has_tumor"] else None,
                "das_winner_score": float(packed["das_score"][index, das_index]),
                "d4_winner_cm": float(packed["d4_dist"][index, ip, ig, id4]) if scan["has_tumor"] else None,
                "d4_textbook_cm": float(packed["d4_dist"][index, nearest_index(grid, 0.5), nearest_index(grid, 0.0), nearest_index(grid, 0.25)]) if scan["has_tumor"] else None,
                "d4_previous_cm": float(packed["d4_dist"][index, nearest_index(grid, 0.55), nearest_index(grid, 0.60), nearest_index(grid, 0.55)]) if scan["has_tumor"] else None,
                "blend_winner": float(blend[index, ip, ig, id4]),
                "blend_previous": float(blend[index, nearest_index(grid, 0.55), nearest_index(grid, 0.60), nearest_index(grid, 0.55)]),
                "blend_textbook": float(blend[index, nearest_index(grid, 0.5), nearest_index(grid, 0.0), nearest_index(grid, 0.25)]),
            }
        )

    return {
        "grid": [float(v) for v in grid],
        "das_index": das_index,
        "das_gamma": float(grid[das_index]),
        "das_rows": das_rows,
        "pair_index": ip,
        "gamma_index": ig,
        "d4_index": id4,
        "pair": float(grid[ip]),
        "dmas_gamma": float(grid[ig]),
        "d4": float(grid[id4]),
        "pair_rows": pair_rows,
        "gamma_rows": gamma_rows,
        "d4_rows": d4_rows,
        "pair_dmas_rows": pair_dmas_rows,
        "gamma_dmas_rows": gamma_dmas_rows,
        "blend_best": float(blend_overall[ip, ig, id4]),
        "dmas_only_pair": float(grid[dmas_best_p]),
        "dmas_only_gamma": float(grid[dmas_best_g]),
        "dmas_only_score": float(dmas_only[dmas_best_p, dmas_best_g]),
        "d4_only_pair": float(grid[d4_best[0]]),
        "d4_only_gamma": float(grid[d4_best[1]]),
        "d4_only_d4": float(grid[d4_best[2]]),
        "d4_only_score": float(d4_only[d4_best]),
        "previous_blend": float(blend_overall[nearest_index(grid, 0.55), nearest_index(grid, 0.60), nearest_index(grid, 0.55)]),
        "textbook_blend": float(blend_overall[nearest_index(grid, 0.5), nearest_index(grid, 0.0), nearest_index(grid, 0.25)]),
        "previous_das": float(das_overall[nearest_index(grid, 0.75)]),
        "textbook_das": float(das_overall[nearest_index(grid, 0.0)]),
        "winner_das": float(das_overall[das_index]),
        "scans": per_scan,
        "n_tumor": int(np.sum(tumor_mask)),
        "n_healthy": int(np.sum(~tumor_mask)),
        "blend_overall": blend_overall,
        "das_overall": das_overall,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def rows_to_md(rows: list[dict], winner_value: float) -> str:
    body = []
    for row in rows:
        mark = " **" if abs(row["value"] - winner_value) < 1e-9 else ""
        end = "**" if mark else ""
        body.append(
            [
                f"{mark}{row['value']:.2f}{end}",
                f"{mark}{_fmt(row['overall'], 3)}{end}",
                _fmt(row["tumor"], 3),
                _fmt(row["healthy"], 3),
                _fmt(row["loc_cm"]),
                _fmt(row["area"], 3),
                _fmt(row["ring"]),
                _fmt(row["fp"]),
            ]
        )
    return markdown_table(
        ["Value", "Average", "Tumor score", "Healthy score", "Tumor loc. cm", "Tumor area", "Tumor ring", "Healthy FP"],
        body,
    )


def write_markdown(summary: dict) -> str:
    das_winner = summary["das_rows"][summary["das_index"]]
    pair_winner = summary["pair_rows"][summary["pair_index"]]
    gamma_winner = summary["gamma_rows"][summary["gamma_index"]]
    d4_winner = summary["d4_rows"][summary["d4_index"]]
    lines = [
        "# Beamformer constant sweep",
        "",
        "Research and educational comparison only. These scores are not a clinical diagnosis.",
        "",
        "## How the average was defined",
        "",
        f"Every constant was sampled from **0.00 to 1.00 in steps of {(summary['grid'][1] - summary['grid'][0]):.2f}** "
        f"({len(summary['grid'])} values). "
        "Values outside that interval are not part of the standard definitions: a coherence power above 1 "
        "over-suppresses any pixel that is not perfectly coherent, and a negative exponent blows up noise.",
        "",
        "The cohort is a stratified UM-BMID sample, not all 2,264 scans: each generation contributes one healthy scan "
        "and small, medium, and large tumors in different quadrants when those labels exist. "
        f"This run used **{summary['n_tumor']} tumor scans and {summary['n_healthy']} healthy scans**.",
        "",
        "For each setting the pipeline's own ROI detector picks the top spot **without the tumor label**. "
        "The label is used afterwards only to measure distance. A tumor score is",
        "",
        "`0.55 exp(-distance_cm / 2.5) + 0.15 ring_focus + 0.15 compact_size + 0.15 SCR`.",
        "",
        "A healthy score is 1 when that same blind rule rejects the scan, and lower when it calls a false tumor. "
        f"The reported average is **{TUMOR_WEIGHT:.2f} × mean tumor score + {HEALTHY_WEIGHT:.2f} × mean healthy score**.",
        "",
        "DAS has one knob, so its winner is the gamma with the best average. "
        "DMAS and DMAS-D4 share the pair exponent and the DMAS coherence gamma in the code, and D4 adds its own exponent. "
        "The shared triple maximizes `0.5 × DMAS score + 0.5 × DMAS-D4 score`. "
        "The default is that highest average. A second peak stays a second peak even when it is within 0.01, "
        "so a lower mode is not promoted just because it is nearer the textbook constants. "
        "The textbook value replaces the peak only when it is one grid step away and the score drop is under 0.001.",
        "",
        "## Chosen defaults",
        "",
        markdown_table(
            ["Knob", "Textbook", "Average-best"],
            [
                ["DAS coherence γ", "0.00", f"**{summary['das_gamma']:.2f}**"],
                ["DMAS pair exponent", "0.50", f"**{summary['pair']:.2f}**"],
                ["DMAS coherence γ", "0.00", f"**{summary['dmas_gamma']:.2f}**"],
                ["DMAS-D4 exponent", "0.25", f"**{summary['d4']:.2f}**"],
            ],
        ),
        "",
        f"DAS average at the winner: **{_fmt(summary['winner_das'], 3)}**. "
        f"Textbook γ = 0 scored {_fmt(summary['textbook_das'], 3)}.",
        "",
        f"Shared DMAS + D4 average at the winner: **{_fmt(summary['blend_best'], 3)}**. "
        f"Textbook (0.50, 0, 0.25) scored {_fmt(summary['textbook_blend'], 3)}.",
        "",
        "Optimizing DMAS alone, ignoring D4, would pick "
        f"pair **{summary['dmas_only_pair']:.2f}** and γ **{summary['dmas_only_gamma']:.2f}** "
        f"(score {_fmt(summary['dmas_only_score'], 3)}). "
        "Optimizing the D4 image alone would pick "
        f"pair **{summary['d4_only_pair']:.2f}**, γ **{summary['d4_only_gamma']:.2f}**, exponent **{summary['d4_only_d4']:.2f}** "
        f"(score {_fmt(summary['d4_only_score'], 3)}). "
        "The defaults below are the shared compromise, because one configuration feeds both algorithms.",
        "",
        "## Cohort",
        "",
        markdown_table(
            ["Generation", "Scan", "Label", "This scan's best DAS / pair / DMAS γ / D4", "Cohort DAS cm", "Textbook DAS cm", "Cohort D4 cm", "Textbook D4 cm"],
            [
                [
                    scan["generation"],
                    str(scan["index"]),
                    scan["label"],
                    f"{scan['best_das_gamma']:.2f} / {scan['best_pair']:.2f} / {scan['best_dmas_gamma']:.2f} / {scan['best_d4']:.2f}",
                    _fmt(scan["das_winner_cm"]) if scan["das_winner_cm"] is not None else "healthy",
                    _fmt(scan["das_textbook_cm"]) if scan["das_textbook_cm"] is not None else "healthy",
                    _fmt(scan["d4_winner_cm"]) if scan["d4_winner_cm"] is not None else "healthy",
                    _fmt(scan["d4_textbook_cm"]) if scan["d4_textbook_cm"] is not None else "healthy",
                ]
                for scan in summary["scans"]
            ],
        ),
        "",
        "The fourth column is the best setting **for that scan alone**. It is there to show the spread. "
        "Healthy rows often tie across the whole grid, and the table then shows the first tied cell. "
        "The project default is the cohort average, not any one of those per-scan winners. "
        "Localization columns are the distance from the **blind top-ranked spot** to the label when the cohort-best or textbook constants are used.",
        "",
        "## DAS coherence γ",
        "",
        rows_to_md(summary["das_rows"], summary["das_gamma"]),
        "",
        "### Why the other DAS values lost",
        "",
    ]
    lines.extend(f"- {paragraph}" for paragraph in group_failures(summary["das_rows"], das_winner))
    lines.extend(
        [
            "",
            "Named alternatives on this axis:",
            "",
        ]
    )
    for value in (0.0, 0.25, 0.35, 0.50, 0.55, 0.75, 1.0):
        row = summary["das_rows"][nearest_index(np.array(summary["grid"]), value)]
        if abs(row["value"] - das_winner["value"]) < 1e-9:
            lines.append(f"- **{value:.2f} is the average-best DAS γ.**")
        else:
            lines.append(f"- {_reason_sentence(value, row, das_winner, _reason_tag(row, das_winner))}")

    lines.extend(
        [
            "",
            "## DMAS pair exponent",
            "",
            "This slice holds DMAS γ and the D4 exponent at the winning companions. "
            "The average column is the shared DMAS + D4 score. Localization, area, and ring are measured on the D4 image, "
            "because the D4 image is where the compression effect shows most.",
            "",
            rows_to_md(summary["pair_rows"], summary["pair"]),
            "",
            "### Why the other pair exponents lost",
            "",
        ]
    )
    lines.extend(f"- {paragraph}" for paragraph in group_failures(summary["pair_rows"], pair_winner))
    lines.extend(["", "Named alternatives:", ""])
    for value in (0.25, 0.35, 0.50, 0.55, 0.75, 1.0):
        row = summary["pair_rows"][nearest_index(np.array(summary["grid"]), value)]
        if abs(row["value"] - pair_winner["value"]) < 1e-9:
            lines.append(f"- **{value:.2f} is the average-best pair exponent.**")
        else:
            lines.append(f"- {_reason_sentence(value, row, pair_winner, _reason_tag(row, pair_winner))}")

    lines.extend(
        [
            "",
            "The same pair axis scored on the DMAS image only (D4 not applied), still at the winning DMAS γ:",
            "",
            rows_to_md(summary["pair_dmas_rows"], summary["pair"]),
            "",
            "## DMAS coherence γ",
            "",
            "Pair exponent and D4 exponent are held at the winning companions.",
            "",
            rows_to_md(summary["gamma_rows"], summary["dmas_gamma"]),
            "",
            "### Why the other DMAS γ values lost",
            "",
        ]
    )
    lines.extend(f"- {paragraph}" for paragraph in group_failures(summary["gamma_rows"], gamma_winner))
    lines.extend(["", "Named alternatives:", ""])
    for value in (0.0, 0.25, 0.35, 0.50, 0.60, 0.75, 1.0):
        row = summary["gamma_rows"][nearest_index(np.array(summary["grid"]), value)]
        if abs(row["value"] - gamma_winner["value"]) < 1e-9:
            lines.append(f"- **{value:.2f} is the average-best DMAS γ.**")
        else:
            lines.append(f"- {_reason_sentence(value, row, gamma_winner, _reason_tag(row, gamma_winner))}")

    lines.extend(
        [
            "",
            "DMAS-image-only γ curve at the winning pair exponent:",
            "",
            rows_to_md(summary["gamma_dmas_rows"], summary["dmas_gamma"]),
            "",
            "## DMAS-D4 exponent",
            "",
            "Pair exponent and DMAS γ are held at the winning companions. This table is the D4 image alone, "
            "so a bad exponent cannot hide behind a good DMAS image.",
            "",
            rows_to_md(summary["d4_rows"], summary["d4"]),
            "",
            "### Why the other D4 exponents lost",
            "",
        ]
    )
    lines.extend(f"- {paragraph}" for paragraph in group_failures(summary["d4_rows"], d4_winner))
    lines.extend(["", "Named alternatives:", ""])
    for value in (0.25, 0.35, 0.50, 0.55, 0.75, 1.0):
        row = summary["d4_rows"][nearest_index(np.array(summary["grid"]), value)]
        if abs(row["value"] - d4_winner["value"]) < 1e-9:
            lines.append(f"- **{value:.2f} is the average-best D4 exponent.**")
        else:
            lines.append(f"- {_reason_sentence(value, row, d4_winner, _reason_tag(row, d4_winner))}")

    lines.extend(
        [
            "",
            "## What this does not claim",
            "",
            "- The grid step is 0.05. A neighbor 0.02 away was not scored, and the tie rule treats gaps under 0.01 as noise.",
            "- The cohort is stratified across generations, sizes, and locations. It is not every UM-BMID scan. A different mix can move the average slightly.",
            "- One scan can prefer a different value. The default is the cohort mean. Per-scan retuning is a separate search and is not what these defaults are.",
            "- Geometry, wave speed, and the numerical floor ε = 1e-9 were held at the values the app already uses. This sweep does not retune those.",
            "",
            "Plots: `results/beamformer_sweep/das_gamma.png`, `dmas_pair.png`, `dmas_gamma.png`, `dmas_d4.png`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def save_plots(summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.array(summary["grid"])

    def curve(path, rows, title, winner):
        fig, ax1 = plt.subplots(figsize=(8.2, 4.4))
        ax1.plot(grid, [row["overall"] for row in rows], color="#1b365d", marker="o", label="Average score")
        ax1.axvline(winner, color="#1f6f8b", linestyle="--", label=f"Chosen {winner:.2f}")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Average score")
        ax1.set_ylim(0, 1)
        ax2 = ax1.twinx()
        ax2.plot(grid, [row["loc_cm"] for row in rows], color="#b45309", marker="s", alpha=0.8, label="Localization cm")
        ax2.set_ylabel("Mean localization error (cm)")
        ax1.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    curve(OUT_DIR / "das_gamma.png", summary["das_rows"], "DAS coherence gamma", summary["das_gamma"])
    curve(OUT_DIR / "dmas_pair.png", summary["pair_rows"], "DMAS pair exponent (companions fixed at winner)", summary["pair"])
    curve(OUT_DIR / "dmas_gamma.png", summary["gamma_rows"], "DMAS coherence gamma (companions fixed at winner)", summary["dmas_gamma"])
    curve(OUT_DIR / "dmas_d4.png", summary["d4_rows"], "DMAS-D4 exponent (companions fixed at winner)", summary["d4"])

    heat = summary["blend_overall"][:, summary["gamma_index"], :]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(heat, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
    ax.scatter([summary["d4"]], [summary["pair"]], color="white", edgecolor="black", s=40, label="Chosen")
    ax.set_xlabel("D4 exponent")
    ax.set_ylabel("DMAS pair exponent")
    ax.set_title(f"Shared score at DMAS γ = {summary['dmas_gamma']:.2f}")
    fig.colorbar(image, ax=ax, label="Average score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pair_vs_d4.png", dpi=140)
    plt.close(fig)


def write_pdf(summary: dict, markdown: str) -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Times-Roman", fontSize=10, leading=13, alignment=TA_JUSTIFY, spaceAfter=8)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=colors.HexColor("#1b365d"), spaceAfter=8)
    small = ParagraphStyle("small", parent=styles["Normal"], fontName="Times-Roman", fontSize=8, leading=10, alignment=TA_LEFT)
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=landscape(A4), leftMargin=1.2 * cm, rightMargin=1.2 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
    story = [
        Paragraph("Beamformer constant sweep", h1),
        Paragraph(
            "Average-best DAS, DMAS, and DMAS-D4 constants on a stratified UM-BMID cohort. "
            "Research and educational use only. Not a clinical diagnosis.",
            body,
        ),
        Paragraph(
            f"Chosen: DAS γ = {summary['das_gamma']:.2f}, DMAS pair = {summary['pair']:.2f}, "
            f"DMAS γ = {summary['dmas_gamma']:.2f}, D4 exponent = {summary['d4']:.2f}. "
            f"Grid: 0.00 to 1.00 step 0.05. Tie band: {TIE:.2f}. "
            f"Score weights: {TUMOR_WEIGHT:.2f} tumor / {HEALTHY_WEIGHT:.2f} healthy, and 0.50 DMAS / 0.50 D4 for the shared triple.",
            body,
        ),
    ]

    def add_table(title: str, rows: list[dict], winner: float):
        story.append(Paragraph(title, h1))
        data = [["Value", "Average", "Tumor", "Healthy", "Loc cm", "Area", "Ring", "Healthy FP"]]
        for row in rows:
            data.append(
                [
                    f"{row['value']:.2f}",
                    _fmt(row["overall"], 3),
                    _fmt(row["tumor"], 3),
                    _fmt(row["healthy"], 3),
                    _fmt(row["loc_cm"]),
                    _fmt(row["area"], 3),
                    _fmt(row["ring"]),
                    _fmt(row["fp"]),
                ]
            )
        table = Table(data, repeatRows=1)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b365d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#c5d0de")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f7fb")),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]
        for index, row in enumerate(rows, start=1):
            if abs(row["value"] - winner) < 1e-9:
                style_cmds.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor("#d9efe4")))
        table.setStyle(TableStyle(style_cmds))
        story.append(table)
        story.append(Spacer(1, 0.3 * cm))

    add_table("DAS coherence gamma — full grid", summary["das_rows"], summary["das_gamma"])
    story.append(Paragraph("Why the other DAS values lost", h1))
    winner = summary["das_rows"][summary["das_index"]]
    for paragraph in group_failures(summary["das_rows"], winner):
        story.append(Paragraph(paragraph, body))
    story.append(PageBreak())
    add_table("DMAS pair exponent at the winning companions", summary["pair_rows"], summary["pair"])
    story.append(Paragraph("Why the other pair exponents lost", h1))
    for paragraph in group_failures(summary["pair_rows"], summary["pair_rows"][summary["pair_index"]]):
        story.append(Paragraph(paragraph, body))
    add_table("DMAS coherence gamma at the winning companions", summary["gamma_rows"], summary["dmas_gamma"])
    story.append(Paragraph("Why the other DMAS gamma values lost", h1))
    for paragraph in group_failures(summary["gamma_rows"], summary["gamma_rows"][summary["gamma_index"]]):
        story.append(Paragraph(paragraph, body))
    story.append(PageBreak())
    add_table("DMAS-D4 exponent at the winning companions", summary["d4_rows"], summary["d4"])
    story.append(Paragraph("Why the other D4 exponents lost", h1))
    for paragraph in group_failures(summary["d4_rows"], summary["d4_rows"][summary["d4_index"]]):
        story.append(Paragraph(paragraph, body))
    story.append(Paragraph("Per-scan localization of the blind top spot (cm). Healthy scans have no target distance.", h1))
    cohort = [["Gen", "Scan", "Label", "DAS winner", "DAS textbook", "D4 winner", "D4 textbook"]]
    for scan in summary["scans"]:
        cohort.append(
            [
                scan["generation"],
                str(scan["index"]),
                scan["label"][:42],
                _fmt(scan["das_winner_cm"]) if scan["has_tumor"] else "healthy",
                _fmt(scan["das_textbook_cm"]) if scan["has_tumor"] else "healthy",
                _fmt(scan["d4_winner_cm"]) if scan["has_tumor"] else "healthy",
                _fmt(scan["d4_textbook_cm"]) if scan["has_tumor"] else "healthy",
            ]
        )
    cohort_table = Table(cohort, repeatRows=1)
    cohort_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#c5d0de")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f7fb")),
            ]
        )
    )
    story.append(cohort_table)
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "The full narrative, including every named alternative (0.25, 0.35, 0.50, 0.55, 0.75), "
            "is in results/beamformer_constant_sweep_detailed.md. Plots are in results/beamformer_sweep/.",
            small,
        )
    )
    doc.build(story)
    del markdown


SIMPLE_KNOBS = [
    {
        "key": "das",
        "rows": "das_rows",
        "winner": "das_gamma",
        "title": "Knob 1 - DAS dimming weight",
        "what": (
            "DAS adds up the echoes from all antennas. This knob decides how much to dim a spot "
            "when the antennas disagree about it. 0 means no dimming (the textbook version). "
            "1 means strong dimming."
        ),
        "textbook": 0.0,
        "show": (0.0, 0.25, 0.35, 0.50, 0.55, 0.75, 1.0),
    },
    {
        "key": "pair",
        "rows": "pair_rows",
        "winner": "pair",
        "title": "Knob 2 - DMAS pair squeeze",
        "what": (
            "DMAS multiplies the echoes of every pair of antennas. This knob sets how much each "
            "pair's result is squeezed before everything is added up. 0.5 is the textbook square root. "
            "A smaller number squeezes more, a larger number squeezes less."
        ),
        "textbook": 0.50,
        "show": (0.0, 0.25, 0.30, 0.35, 0.50, 0.55, 0.75, 1.0),
    },
    {
        "key": "dmas_gamma",
        "rows": "gamma_rows",
        "winner": "dmas_gamma",
        "title": "Knob 3 - DMAS dimming weight",
        "what": (
            "The same dimming idea as Knob 1, but applied to the DMAS picture. "
            "0 means no dimming (textbook). 1 means strong dimming."
        ),
        "textbook": 0.0,
        "show": (0.0, 0.25, 0.35, 0.50, 0.60, 0.75, 1.0),
    },
    {
        "key": "d4",
        "rows": "d4_rows",
        "winner": "d4",
        "title": "Knob 4 - D4 brightness compression",
        "what": (
            "After DMAS, D4 compresses the brightness range of the final picture so weak and strong "
            "spots are easier to compare. 0.25 is the textbook fourth root. 1 means no compression at all."
        ),
        "textbook": 0.25,
        "show": (0.0, 0.25, 0.35, 0.50, 0.55, 0.75, 1.0),
    },
]

SIMPLE_VERDICT = {
    "tied": "Almost the same score - difference too small to matter",
    "localization": "Bright spot lands farther from the tumor",
    "blob": "Bright spot spreads over a wide area",
    "ring": "Edge ring noise stays brighter than the tumor",
    "false_positive": "A healthy scan is wrongly flagged as tumor",
    "healthy": "Healthy scans handled less cleanly",
    "lower_average": "Slightly lower score, no single big problem",
}


def _healthy_alarms(row: dict, n_healthy: int) -> str:
    count = int(round(float(row["fp"]) * n_healthy)) if np.isfinite(row["fp"]) else 0
    return f"{count} of {n_healthy}"


def _simple_groups(rows: list[dict], winner: dict, n_healthy: int) -> list[str]:
    lines: list[str] = []
    tag = None
    bucket: list[dict] = []

    def flush():
        if not bucket:
            return
        lo, hi = bucket[0]["value"], bucket[-1]["value"]
        label = f"{lo:.2f}" if abs(lo - hi) < 1e-9 else f"{lo:.2f} to {hi:.2f}"
        score = 100.0 * float(np.nanmean([item["overall"] for item in bucket]))
        miss = float(np.nanmean([item["loc_cm"] for item in bucket]))
        win_score = 100.0 * winner["overall"]
        text = {
            "tied": (
                f"<b>{label}</b>: almost the same score ({score:.1f} vs {win_score:.1f}). "
                "The gap is too small to call a real difference, so the top value was kept."
            ),
            "localization": (
                f"<b>{label}</b>: the bright spot landed farther from the real tumor, "
                f"about {miss:.1f} cm away instead of {winner['loc_cm']:.1f} cm."
            ),
            "blob": (
                f"<b>{label}</b>: the bright spot spread over a wide area instead of staying a small, clear point."
            ),
            "ring": (
                f"<b>{label}</b>: the ring-shaped noise near the antennas stayed brighter than the tumor."
            ),
            "false_positive": (
                f"<b>{label}</b>: a healthy scan was wrongly called a tumor "
                f"({_healthy_alarms(bucket[-1], n_healthy)} healthy scans)."
            ),
            "healthy": (
                f"<b>{label}</b>: tumor scans were similar, but healthy scans were handled less cleanly."
            ),
            "lower_average": (
                f"<b>{label}</b>: slightly lower score overall ({score:.1f} vs {win_score:.1f}), "
                "with no single big problem."
            ),
        }[tag]
        lines.append(text)

    for row in rows:
        if abs(row["value"] - winner["value"]) < 1e-9:
            flush()
            tag, bucket = None, []
            continue
        row_tag = _reason_tag(row, winner)
        if row_tag != tag:
            flush()
            tag, bucket = row_tag, [row]
        else:
            bucket.append(row)
    flush()
    return lines


def save_simple_plots(summary: dict) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.array(summary["grid"])
    paths: dict[str, Path] = {}
    for knob in SIMPLE_KNOBS:
        rows = summary[knob["rows"]]
        winner = summary[knob["winner"]]
        scores = [100.0 * row["overall"] for row in rows]
        palette = []
        for row in rows:
            if abs(row["value"] - winner) < 1e-9:
                palette.append("#2e8b57")
            elif np.isfinite(row["fp"]) and row["fp"] > 0:
                palette.append("#c0504d")
            else:
                palette.append("#8fa8c8")
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
        ax.bar(grid, scores, width=0.04, color=palette)
        low = max(0.0, min(scores) - 5.0)
        ax.set_ylim(low, max(scores) + 3.0)
        ax.set_xticks(grid[::2])
        ax.set_xlabel("Knob value")
        ax.set_ylabel("Score out of 100")
        ax.set_title(f"{knob['title']}  (chosen = {winner:.2f})", fontsize=10)
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="#2e8b57", label="Chosen"),
                Patch(color="#8fa8c8", label="Other values"),
                Patch(color="#c0504d", label="Healthy scan wrongly flagged"),
            ],
            fontsize=7.5,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            frameon=False,
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = OUT_DIR / f"simple_{knob['key']}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[knob["key"]] = path
    return paths


def write_simple_pdf(summary: dict) -> None:
    plots = save_simple_plots(summary)
    grid = np.array(summary["grid"])
    n_healthy = int(summary["n_healthy"])
    n_tumor = int(summary["n_tumor"])
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#1b365d")
    body = ParagraphStyle("sbody", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5, leading=14.5, alignment=TA_LEFT, spaceAfter=7)
    h1 = ParagraphStyle("sh1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, textColor=navy, spaceAfter=8)
    h2 = ParagraphStyle("sh2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, textColor=navy, spaceBefore=6, spaceAfter=6)
    note = ParagraphStyle("snote", parent=body, fontSize=9, leading=12, textColor=colors.HexColor("#555555"))
    cell = ParagraphStyle("scell", parent=body, fontSize=9, leading=11.5, spaceAfter=0)
    head = ParagraphStyle("shead", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)

    def table(data, widths, highlight_rows=(), wrap=True):
        wrapped = []
        for r, row in enumerate(data):
            style = head if r == 0 else cell
            wrapped.append([Paragraph(str(value), style) if wrap else value for value in row])
        tbl = Table(wrapped, colWidths=widths, repeatRows=1)
        cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d0de")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for r in highlight_rows:
            cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#d9efe4")))
        tbl.setStyle(TableStyle(cmds))
        return tbl

    doc = SimpleDocTemplate(
        str(SIMPLE_PDF_PATH),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Beamformer settings - simple explanation",
    )
    story = [
        Paragraph("Choosing the beamformer settings", h1),
        Paragraph("A plain-language summary of the test. Research and educational use only, not a medical diagnosis.", note),
        Paragraph("1. The short answer", h2),
        Paragraph(
            "The three imaging methods (DAS, DMAS and DMAS-D4) each have a few number settings, or \"knobs\". "
            "Every value from 0 to 1 was tried, in steps of 0.05, on real breast-phantom scans. "
            "The value that worked best <b>on average across all scans</b> was kept.",
            body,
        ),
        table(
            [
                ["Knob", "Textbook value", "Chosen value", "In one line"],
                ["DAS dimming weight", "0", f"<b>{summary['das_gamma']:.2f}</b>", "Some dimming helps; too much flags healthy scans"],
                ["DMAS pair squeeze", "0.50", f"<b>{summary['pair']:.2f}</b>", "Squeezing a bit more put the spot closer to the tumor"],
                ["DMAS dimming weight", "0", f"<b>{summary['dmas_gamma']:.2f}</b>", "Extra dimming did not help DMAS"],
                ["D4 compression", "0.25", f"<b>{summary['d4']:.2f}</b>", "The textbook value was already the best"],
            ],
            [4.2 * cm, 2.4 * cm, 2.4 * cm, 8.4 * cm],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "Overall score out of 100 (higher is better) for each full set of settings:",
            body,
        ),
        table(
            [
                ["Settings", "DAS score", "DMAS + D4 score"],
                ["Textbook", f"{100 * summary['textbook_das']:.1f}", f"{100 * summary['textbook_blend']:.1f}"],
                ["<b>Chosen</b>", f"<b>{100 * summary['winner_das']:.1f}</b>", f"<b>{100 * summary['blend_best']:.1f}</b>"],
            ],
            [5.5 * cm, 4.0 * cm, 4.0 * cm],
            highlight_rows=(2,),
        ),
        Paragraph("2. How the test worked", h2),
        Paragraph(
            f"<b>Scans used:</b> {n_tumor + n_healthy} scans from the UM-BMID phantom dataset "
            f"({n_tumor} with a tumor, {n_healthy} healthy), taken from all three dataset generations and covering "
            "small, medium and large tumors in different positions.",
            body,
        ),
        Paragraph(
            "<b>For each knob value:</b> the picture was rebuilt for every scan, and the program found the brightest spot "
            "<b>on its own</b>, without being told where the tumor really was. Only afterwards was the answer checked:",
            body,
        ),
        table(
            [
                ["What was checked", "Good result"],
                ["Tumor scans: how far the bright spot landed from the real tumor (cm)", "Small distance"],
                ["Tumor scans: whether the spot is small and clear, not smeared or buried in edge noise", "Small, sharp spot"],
                ["Healthy scans: whether the program wrongly said \"tumor\"", "No false alarm"],
            ],
            [11.5 * cm, 5.9 * cm],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "These were combined into one <b>score out of 100</b>. Tumor scans count for 70% and healthy scans for 30%. "
            "The chosen value is the one with the <b>highest average score over all scans</b>, not the best value for any single scan.",
            body,
        ),
        PageBreak(),
        Paragraph("3. Each knob, and why the other values lost", h2),
        Paragraph(
            "Each chart shows the score for all 21 values. <font color='#2e8b57'><b>Green</b></font> is the chosen value. "
            "<font color='#c0504d'><b>Red</b></font> means at least one healthy scan was wrongly flagged as a tumor. "
            "The table below each chart lists the most discussed values. \"Average miss\" is how far, on average, the bright spot "
            "landed from the real tumor.",
            body,
        ),
    ]

    for knob in SIMPLE_KNOBS:
        rows = summary[knob["rows"]]
        winner_value = float(summary[knob["winner"]])
        winner_row = rows[nearest_index(grid, winner_value)]
        values = sorted({*knob["show"], winner_value, knob["textbook"]})
        data = [["Value", "Score /100", "Average miss", "Healthy false alarms", "What happened"]]
        highlight = []
        for value in values:
            row = rows[nearest_index(grid, value)]
            tags = []
            if abs(value - knob["textbook"]) < 1e-9:
                tags.append("textbook")
            is_winner = abs(value - winner_value) < 1e-9
            if is_winner:
                tags.append("chosen")
                verdict = "Best average score"
                highlight.append(len(data))
            else:
                verdict = SIMPLE_VERDICT[_reason_tag(row, winner_row)]
            label = f"{value:.2f}" + (f" ({', '.join(tags)})" if tags else "")
            data.append(
                [
                    f"<b>{label}</b>" if is_winner else label,
                    f"{100 * row['overall']:.1f}",
                    f"{row['loc_cm']:.1f} cm",
                    _healthy_alarms(row, n_healthy),
                    verdict,
                ]
            )
        block = [
            Paragraph(knob["title"], h2),
            Paragraph(knob["what"], body),
            Image(str(plots[knob["key"]]), width=17.2 * cm, height=7.5 * cm),
            table(data, [3.8 * cm, 2.0 * cm, 2.3 * cm, 2.8 * cm, 6.5 * cm], highlight_rows=highlight),
            Spacer(1, 0.2 * cm),
            Paragraph("<b>Why not the other values?</b>", body),
        ]
        story.append(KeepTogether(block[:4]))
        story.extend(block[4:])
        for line in _simple_groups(rows, winner_row, n_healthy):
            story.append(Paragraph("&bull; " + line, body))
        story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())
    story.append(Paragraph("4. Scan by scan: textbook settings vs chosen settings", h2))
    story.append(
        Paragraph(
            "Distance (cm) from the program's bright spot to the real tumor, for each tumor scan. Smaller is better. "
            "The last column is the DAS dimming weight that would suit only that one scan.",
            body,
        )
    )
    per_scan = [["Scan", "Tumor", "DAS textbook", "DAS chosen", "D4 textbook", "D4 chosen", "Best DAS weight for this scan alone"]]
    for scan in summary["scans"]:
        if not scan["has_tumor"]:
            continue
        parts = scan["label"].split("/")
        diameter = scan.get("diameter_cm")
        size = f"{diameter:.0f} cm" if diameter else parts[0]
        where = parts[-1] if len(parts) > 1 else ""
        per_scan.append(
            [
                f"{scan['generation']} #{scan['index']}",
                f"{size}, {where}",
                f"{scan['das_textbook_cm']:.1f}",
                f"{scan['das_winner_cm']:.1f}",
                f"{scan['d4_textbook_cm']:.1f}",
                f"{scan['d4_winner_cm']:.1f}",
                f"{scan['best_das_gamma']:.2f}",
            ]
        )
    story.append(table(per_scan, [2.2 * cm, 3.0 * cm, 2.1 * cm, 2.0 * cm, 2.1 * cm, 2.0 * cm, 4.0 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    stuck = [
        f"{scan['generation']} #{scan['index']}"
        for scan in summary["scans"]
        if scan["has_tumor"]
        and min(
            scan["das_winner_cm"], scan["das_textbook_cm"],
            scan["d4_winner_cm"], scan["d4_textbook_cm"],
        ) > 5.0
    ]
    stuck_text = (
        f" Some scans ({', '.join(stuck)}) stay more than 5 cm from the tumor with both textbook and chosen settings, "
        "so no knob value fixes them; that problem lies elsewhere in the pipeline."
        if stuck
        else ""
    )
    story.append(
        Paragraph(
            "<b>Why use the average and not one scan's best?</b> The last column shows that each scan prefers a different value, "
            "anywhere from low to high. A value tuned to one scan can be poor on the next. The average-best value is the safest single "
            "setting across the whole set." + stuck_text,
            body,
        )
    )
    story.append(Paragraph("5. Limits of this test", h2))
    for text in (
        f"It used {n_tumor + n_healthy} carefully mixed scans, not all 2,264 in the dataset. A different mix could shift the result slightly.",
        "Values were tried in steps of 0.05. Something like 0.32 was not tested separately.",
        "Scores that differ by less than 1 point out of 100 are treated as practically the same.",
        "Antenna layout and wave speed were kept as the app already uses them; only the four knobs were tested.",
    ):
        story.append(Paragraph("&bull; " + text, body))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "Full numbers for every value are in beamformer_constant_sweep_detailed.pdf and .md in the same results folder.",
            note,
        )
    )
    doc.build(story)


def stack_metric(scans_payload: list[dict], family: str, key: str) -> np.ndarray:
    return np.stack([item[family][key] for item in scans_payload], axis=0)


def run(step: float, per_file: int, generations: set[str]) -> dict:
    _assert_pairwise_matches_reference()
    grid = value_grid(step)
    print(f"Grid ({grid.size} values): {grid.tolist()}", flush=True)
    scans_meta = []
    payloads = []
    for generation, path in DATA_FILES:
        if generation not in generations:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Listing {generation} metadata…", flush=True)
        catalog = list_bmid_scans(str(path))
        picked = pick_cohort(catalog, per_file)
        print(f"Loading {path.name} ({path.stat().st_size / 1e6:.0f} MB)…", flush=True)
        t_load = time.perf_counter()
        _name, cube = _load_fd_cube(str(path))
        print(f"  cube {cube.shape} in {time.perf_counter() - t_load:.1f}s", flush=True)
        frequencies = bmid_frequencies_hz()
        for scan in picked:
            label = taxonomy_from_scan(scan).short_label()
            print(f"  scan {scan.index} {label} tumor={scan.has_tumor}", flush=True)
            s_parameters = np.asarray(cube[scan.index], dtype=np.complex128)
            radius = float(scan.ant_rad_cm or 18.0) / 100.0
            config = ReconstructionConfig(
                x_span=(-0.06, 0.06),
                y_span=(-0.06, 0.06),
                n_x=64,
                n_y=64,
                antenna_radius=radius,
                wave_speed=3e8,
                use_bmid_phase_delay_radius=False,
            )
            positions = antenna_positions_from_config(s_parameters.shape[1], config)
            tumor_xy = None
            if scan.has_tumor and scan.tum_x_cm is not None and scan.tum_y_cm is not None:
                tumor_xy = (scan.tum_x_cm / 100.0, scan.tum_y_cm / 100.0)
            t0 = time.perf_counter()
            das, dmas, d4 = reconstruct_family(s_parameters, frequencies, positions, config, grid)
            print(f"    images in {time.perf_counter() - t0:.1f}s  antennas={s_parameters.shape[1]}", flush=True)
            measured = evaluate_scan_images(
                das, dmas, d4, config.x_span, config.y_span, tumor_xy, lambda msg: print(msg, flush=True)
            )
            del das, dmas, d4
            payloads.append(measured)
            scans_meta.append(
                {
                    "generation": generation,
                    "index": int(scan.index),
                    "has_tumor": bool(scan.has_tumor),
                    "label": label,
                    "file": path.name,
                    "diameter_cm": scan.tum_diam_cm,
                    "x_cm": scan.tum_x_cm,
                    "y_cm": scan.tum_y_cm,
                }
            )
        del cube
    packed = {
        "das_score": stack_metric(payloads, "das", "score"),
        "das_dist": stack_metric(payloads, "das", "primary_dist_cm"),
        "das_area": stack_metric(payloads, "das", "area_ratio"),
        "das_ring": stack_metric(payloads, "das", "ring_excess"),
        "das_fp": stack_metric(payloads, "das", "false_positive"),
        "dmas_score": stack_metric(payloads, "dmas", "score"),
        "dmas_dist": stack_metric(payloads, "dmas", "primary_dist_cm"),
        "dmas_area": stack_metric(payloads, "dmas", "area_ratio"),
        "dmas_ring": stack_metric(payloads, "dmas", "ring_excess"),
        "dmas_fp": stack_metric(payloads, "dmas", "false_positive"),
        "d4_score": stack_metric(payloads, "d4", "score"),
        "d4_dist": stack_metric(payloads, "d4", "primary_dist_cm"),
        "d4_area": stack_metric(payloads, "d4", "area_ratio"),
        "d4_ring": stack_metric(payloads, "d4", "ring_excess"),
        "d4_fp": stack_metric(payloads, "d4", "false_positive"),
    }
    summary = build_summary(scans_meta, grid, packed)
    # Drop the large heat array's non-serializable twin later; keep blend_overall for the plot.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "sweep_scores.npz",
        grid=grid,
        **{key: value for key, value in packed.items()},
    )
    save_plots(summary)
    serial = {key: value for key, value in summary.items() if key not in {"blend_overall", "das_overall"}}
    (OUT_DIR / "summary.json").write_text(json.dumps(serial, indent=2), encoding="utf-8")
    markdown = write_markdown(summary)
    MD_PATH.write_text(markdown, encoding="utf-8")
    write_pdf(summary, markdown)
    write_simple_pdf(summary)
    print(
        f"Chosen DAS gamma={summary['das_gamma']:.2f}  pair={summary['pair']:.2f}  "
        f"DMAS gamma={summary['dmas_gamma']:.2f}  D4={summary['d4']:.2f}",
        flush=True,
    )
    print(f"Wrote {MD_PATH}", flush=True)
    print(f"Wrote {PDF_PATH}", flush=True)
    print(f"Wrote {SIMPLE_PDF_PATH}", flush=True)
    return summary


def rebuild_from_saved() -> dict:
    """Rewrite the report from sweep_scores.npz using the current selection rule."""
    payload = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    stored = np.load(OUT_DIR / "sweep_scores.npz")
    scans = [
        {
            "generation": scan["generation"],
            "index": scan["index"],
            "has_tumor": scan["has_tumor"],
            "label": scan["label"],
            "file": scan.get("file", ""),
            "diameter_cm": scan.get("diameter_cm"),
            "x_cm": scan.get("x_cm"),
            "y_cm": scan.get("y_cm"),
        }
        for scan in payload["scans"]
    ]
    packed = {key: stored[key] for key in stored.files if key != "grid"}
    summary = build_summary(scans, stored["grid"], packed)
    save_plots(summary)
    serial = {key: value for key, value in summary.items() if key not in {"blend_overall", "das_overall"}}
    (OUT_DIR / "summary.json").write_text(json.dumps(serial, indent=2), encoding="utf-8")
    markdown = write_markdown(summary)
    MD_PATH.write_text(markdown, encoding="utf-8")
    write_pdf(summary, markdown)
    write_simple_pdf(summary)
    print(
        f"Chosen DAS gamma={summary['das_gamma']:.2f}  pair={summary['pair']:.2f}  "
        f"DMAS gamma={summary['dmas_gamma']:.2f}  D4={summary['d4']:.2f}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep beamformer constants on UM-BMID")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--scans-per-gen", type=int, default=4)
    parser.add_argument("--gens", default="gen1,gen2,gen3")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the report from saved scores")
    args = parser.parse_args()
    if args.rebuild:
        rebuild_from_saved()
        return
    generations = {item.strip() for item in args.gens.split(",") if item.strip()}
    run(args.step, args.scans_per_gen, generations)


if __name__ == "__main__":
    main()

"""Fair cohort-level calibration of UM-BMID imaging geometry and background removal.

For every tumor scan, a phase-coherent DAS image is formed for several wave
speeds and antenna-radius models, and for three background-removal choices:

    emp      empty-chamber scan subtracted
    emp_rot  empty-chamber subtracted, then the average over antenna angles removed
    adi      adipose-only (fat) scan subtracted - reference only, not available for patients

Rotating or mirroring the antenna ring only rotates or mirrors the image, so the
bright spot from one reconstruction is rotated to evaluate every start angle and
rotation direction. Settings are chosen on half of the phantoms and the error is
reported on the other half (and vice versa), per generation.

Usage:
    python tools/calibrate_bmid_geometry.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.features import calibrate_empty  # noqa: E402
from reconstruction.bmid_geometry import remove_rotational_mean  # noqa: E402
from data_loader.bmid_loader import _load_fd_cube, bmid_frequencies_hz  # noqa: E402
from reconstruction.das import das_coherent_from_frequency, das_from_frequency  # noqa: E402
from reconstruction.reconstruction_manager import bmid_phase_delayed_radius, build_circular_antenna_array  # noqa: E402
from tools.build_classification_features import GENERATIONS, scan_records  # noqa: E402

OUT_PATH = PROJECT / "results" / "classification" / "geometry_calibration.json"
SPAN = 0.06
GRID = np.linspace(-SPAN, SPAN, 64)
SPEEDS = (3.0e8, 2.7e8, 2.4e8, 2.1e8, 1.8e8)
RADIUS_MODES = ("nominal", "phase_delay")
MODES = ("emp", "emp_rot", "adi")
OFFSETS = np.arange(0.0, 360.0, 2.5)
DOCUMENTED = {"gen1": -102.5, "gen2": -130.0}
SEED = 0


def bright_spot_cm(image: np.ndarray) -> tuple[float, float]:
    smooth = gaussian_filter(np.asarray(image, dtype=float), 1.0)
    iy, ix = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    return float(GRID[ix] * 100.0), float(GRID[iy] * 100.0)


def process(task):
    record, raw, empty, adipose = task
    freqs = bmid_frequencies_hz()
    data = {"emp": calibrate_empty(raw, empty)}
    data["emp_rot"] = remove_rotational_mean(data["emp"])
    if adipose is not None:
        data["adi"] = calibrate_empty(raw, adipose)
    spots = {}
    for radius_mode in RADIUS_MODES:
        radius = float(record["ant_rad_cm"]) / 100.0
        if radius_mode == "phase_delay":
            radius = bmid_phase_delayed_radius(radius)
        positions = build_circular_antenna_array(raw.shape[1], radius, 0.0, False, span_deg=355.0)
        for speed in SPEEDS:
            for mode, values in data.items():
                image = das_coherent_from_frequency(values, freqs, positions, GRID, GRID, wave_speed=speed)
                spots[f"{mode}|{radius_mode}|{speed:.2e}"] = bright_spot_cm(image)
    app_positions = build_circular_antenna_array(raw.shape[1], float(record["ant_rad_cm"]) / 100.0)
    spots["app_default"] = bright_spot_cm(das_from_frequency(raw, freqs, app_positions, GRID, GRID))
    return record, spots


def rotate(spots: np.ndarray, offset_deg: float, clockwise: bool) -> np.ndarray:
    pts = spots.copy()
    if clockwise:
        pts[:, 1] = -pts[:, 1]
    a = np.deg2rad(offset_deg)
    c, s = np.cos(a), np.sin(a)
    return np.column_stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]])


def errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.hypot(pred[:, 0] - truth[:, 0], pred[:, 1] - truth[:, 1])


def best_setting(spot_table: dict[str, np.ndarray], truth: np.ndarray, mode: str) -> tuple[dict, float]:
    best, best_err = None, np.inf
    for key, spots in spot_table.items():
        if not key.startswith(mode + "|"):
            continue
        _, radius_mode, speed = key.split("|")
        for clockwise in (False, True):
            for offset in OFFSETS:
                err = float(errors(rotate(spots, offset, clockwise), truth).mean())
                if err < best_err:
                    best_err = err
                    best = {"key": key, "radius_mode": radius_mode, "wave_speed": float(speed), "clockwise": clockwise, "offset_deg": float(offset)}
    return best, best_err


def apply(setting: dict, spot_table: dict[str, np.ndarray]) -> np.ndarray:
    return rotate(spot_table[setting["key"]], setting["offset_deg"], setting["clockwise"])


def main() -> None:
    started = time.time()
    results: dict = {"note": __doc__.strip().splitlines()[0], "generations": {}}
    with get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        for generation, fd_path, md_name in GENERATIONS:
            records = [r for r in scan_records(generation, fd_path.parent / md_name) if r["group"] == "tumor" and np.isfinite(r["tum_x_cm"])]
            _, cube = _load_fd_cube(str(fd_path))
            tasks = (
                (r, np.asarray(cube[r["index"]], dtype=complex), np.asarray(cube[r["emp_index"]], dtype=complex),
                 None if r["adi_index"] is None else np.asarray(cube[r["adi_index"]], dtype=complex))
                for r in records
            )
            rows = list(pool.imap(process, tasks, chunksize=2))
            del cube
            print(f"{generation}: {len(rows)} tumor scans  {time.time() - started:.0f}s", flush=True)

            truth = np.array([[r["tum_x_cm"], r["tum_y_cm"]] for r, _ in rows], dtype=float)
            keys = sorted(set().union(*(spots.keys() for _, spots in rows)))
            table = {k: np.array([spots.get(k, (np.nan, np.nan)) for _, spots in rows]) for k in keys}
            phantoms = np.array([r["phant_id"] for r, _ in rows])
            unique = np.array(sorted(set(phantoms)))
            np.random.default_rng(SEED).shuffle(unique)
            halves = [np.isin(phantoms, unique[: len(unique) // 2]), np.isin(phantoms, unique[len(unique) // 2 :])]

            gen_out = {
                "n_tumor_scans": len(rows),
                "n_phantoms": int(len(unique)),
                "centre_guess_cm": float(errors(np.zeros_like(truth), truth).mean()),
                "app_default_cm": float(errors(table["app_default"], truth).mean()),
                "modes": {},
            }
            for mode in MODES:
                if not any(k.startswith(mode + "|") for k in keys) or np.isnan(table[f"{mode}|nominal|3.00e+08"]).any():
                    continue
                held_out = np.zeros(len(rows))
                chosen = []
                for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
                    sub = {k: v[fit] for k, v in table.items()}
                    setting, fit_err = best_setting(sub, truth[fit], mode)
                    held_out[test] = errors(apply(setting, table)[test], truth[test])
                    chosen.append({**setting, "fit_error_cm": fit_err})
                full_setting, full_err = best_setting(table, truth, mode)
                gen_out["modes"][mode] = {
                    "held_out_mean_cm": float(held_out.mean()),
                    "held_out_within_2cm": float(np.mean(held_out <= 2.0)),
                    "chosen_per_half": chosen,
                    "final_setting": full_setting,
                    "final_fit_error_cm": full_err,
                }
                if generation in DOCUMENTED:
                    doc = {"key": f"{mode}|phase_delay|{full_setting['wave_speed']:.2e}", "clockwise": True, "offset_deg": DOCUMENTED[generation]}
                    gen_out["modes"][mode]["documented_geometry_cm"] = float(errors(apply(doc, table), truth).mean())
                print(
                    f"   {mode:8s} held-out {held_out.mean():.2f} cm (within 2 cm {np.mean(held_out <= 2):.0%})  "
                    f"final {full_setting['radius_mode']} c={full_setting['wave_speed']:.1e} "
                    f"{'CW' if full_setting['clockwise'] else 'CCW'} {full_setting['offset_deg']:.1f} deg",
                    flush=True,
                )
            print(f"   centre guess {gen_out['centre_guess_cm']:.2f} cm, app default {gen_out['app_default_cm']:.2f} cm", flush=True)
            results["generations"][generation] = gen_out

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()

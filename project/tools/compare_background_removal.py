"""Compare single-scan background-removal variants by blind tumour localisation.

Every variant starts from the empty-chamber-subtracted scan and is imaged with
coherent DAS using the calibrated per-generation geometry. Tumour positions are
used only to score the image spot; the variant is chosen on one half of the
phantoms and checked on the other.

Usage:
    python tools/compare_background_removal.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.features import calibrate_empty  # noqa: E402
from classification.predict import recon_config  # noqa: E402
from data_loader.bmid_loader import _load_fd_cube, bmid_frequencies_hz  # noqa: E402
from reconstruction.bmid_geometry import remove_low_rank, remove_rotational_mean, time_gate  # noqa: E402
from reconstruction.das import das_coherent_from_frequency  # noqa: E402
from reconstruction.reconstruction_manager import _build_grid, antenna_positions_from_config  # noqa: E402
from tools.build_classification_features import GENERATIONS, scan_records  # noqa: E402

OUT_PATH = PROJECT / "results" / "classification" / "background_removal_comparison.json"
GATES_NS = (0.0, 0.5, 1.0, 1.5)
RANKS = (1, 2, 3)
BANDS_GHZ = ((1.0, 8.0), (1.0, 4.0), (2.0, 6.0), (4.0, 8.0))
SEED = 0


def variants(cal: np.ndarray, freqs: np.ndarray):
    rot = remove_rotational_mean(cal)
    yield "rot", rot, freqs
    for rank in RANKS:
        yield f"svd{rank}", remove_low_rank(cal, rank), freqs
    for gate in GATES_NS[1:]:
        yield f"rot_gate{gate:g}", time_gate(rot, freqs, gate * 1e-9), freqs
    for lo, hi in BANDS_GHZ[1:]:
        sel = (freqs >= lo * 1e9) & (freqs <= hi * 1e9)
        yield f"rot_band{lo:g}-{hi:g}", rot[sel], freqs[sel]


def spot_cm(image: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[float, float]:
    smooth = gaussian_filter(np.asarray(image, dtype=float), 1.0)
    iy, ix = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    return float(grid_x[ix] * 100.0), float(grid_y[iy] * 100.0)


def process(task):
    record, raw, empty = task
    freqs = bmid_frequencies_hz()
    config = recon_config(record["generation"], record["ant_rad_cm"])
    positions = antenna_positions_from_config(raw.shape[1], config)
    grid_x, grid_y = _build_grid(config.x_span, config.y_span, n_x=config.n_x, n_y=config.n_y)
    cal = calibrate_empty(raw, empty)
    spots = {}
    for name, data, f in variants(cal, freqs):
        image = das_coherent_from_frequency(data, f, positions, grid_x, grid_y, wave_speed=config.wave_speed)
        spots[name] = spot_cm(image, grid_x, grid_y)
    return record, spots


def tasks(generation, fd_path, records):
    _, cube = _load_fd_cube(str(fd_path))
    for record in records:
        yield record, np.asarray(cube[record["index"]], dtype=complex), np.asarray(cube[record["emp_index"]], dtype=complex)


def main() -> None:
    results = {}
    with get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        for generation, fd_path, md_name in GENERATIONS:
            records = [
                r for r in scan_records(generation, fd_path.parent / md_name)
                if r["group"] == "tumor" and np.isfinite(r["tum_x_cm"]) and np.isfinite(r["tum_y_cm"])
            ]
            rows = list(pool.imap(process, tasks(generation, fd_path, records), chunksize=2))
            truth = np.array([[r["tum_x_cm"], r["tum_y_cm"]] for r, _ in rows], dtype=float)
            names = list(rows[0][1])
            err = {n: np.hypot(*(np.array([s[n] for _, s in rows]) - truth).T) for n in names}
            phantoms = np.array([r["phant_id"] for r, _ in rows])
            unique = np.array(sorted(set(phantoms)))
            rng = np.random.default_rng(SEED)
            half = np.isin(phantoms, rng.choice(unique, len(unique) // 2, replace=False))
            held_out = []
            for fit, test in ((half, ~half), (~half, half)):
                best = min(names, key=lambda n: err[n][fit].mean())
                held_out.append((best, err[best][test]))
            results[generation] = {
                "n_tumor_scans": len(rows),
                "centre_guess_cm": float(np.hypot(*truth.T).mean()),
                "mean_miss_cm": {n: float(err[n].mean()) for n in names},
                "within_2cm": {n: float(np.mean(err[n] <= 2.0)) for n in names},
                "chosen_per_half": [b for b, _ in held_out],
                "held_out_mean_cm": float(np.concatenate([e for _, e in held_out]).mean()),
            }
            print(generation, json.dumps({n: round(v, 2) for n, v in results[generation]["mean_miss_cm"].items()}),
                  "| centre", round(results[generation]["centre_guess_cm"], 2),
                  "| chosen", results[generation]["chosen_per_half"], "held-out", round(results[generation]["held_out_mean_cm"], 2), flush=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()

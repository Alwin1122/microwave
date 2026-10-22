"""Build the cached per-scan feature table for tumor classification.

Every UM-BMID phantom scan (tumor, healthy fibroglandular, adipose-only) gets
signal features from the empty-chamber-subtracted data (``sig_``), the same
after rotational-mean removal (``rsig_``), and image features from coherent
DAS (``img_``) and DMAS-D4 (``d4img_``) reconstructions of the
rotational-mean-removed data using the calibrated per-generation geometry,
plus coherent DAS of the 1-4 GHz band only (``imgl_``).

Usage:
    python tools/build_classification_features.py [--limit N] [--workers W]
    python tools/build_classification_features.py --reference adi --gens gen3
    python tools/build_classification_features.py --reference twin --gens gen1 gen3
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
REPO = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.features import calibrate_empty, signal_features  # noqa: E402
from classification.scattering_math import scattering_descriptors  # noqa: E402
from classification.predict import scan_features  # noqa: E402
from data_loader.bmid_loader import bmid_frequencies_hz  # noqa: E402
from reconstruction.bmid_geometry import remove_rotational_mean  # noqa: E402
from data_loader.bmid_loader import _load_fd_cube, _load_metadata_structs, _safe_float, _safe_str  # noqa: E402

OUT_DIR = PROJECT / "results" / "classification"
TWIN_K = 3
GENERATIONS = (
    ("gen1", REPO / "umbmid" / "simple-clean" / "matlab-data2" / "fd_data_gen_one_s11.mat", "metadata_gen_one.mat"),
    ("gen2", REPO / "umbmid" / "simple-clean" / "matlab-data" / "fd_data_gen_two_s11.mat", "metadata_gen_two.mat"),
    ("gen3", REPO / "umbmid" / "simple-clean" / "matlab-data3" / "fd_data_gen_three_s11.mat", "metadata_gen_three.mat"),
)
META_COLUMNS = (
    "generation",
    "index",
    "scan_id",
    "phant_id",
    "group",
    "has_tumor",
    "tum_diam_cm",
    "tum_x_cm",
    "tum_y_cm",
    "birads",
    "ant_rad_cm",
    "n_session",
    "tum_shape",
)


def scan_records(generation: str, md_path: Path) -> list[dict]:
    """Phantom scans of one generation with labels and their empty-chamber index."""
    items = _load_metadata_structs(str(md_path))
    id_to_index = {}
    for index, item in enumerate(items):
        scan_id = _safe_float(getattr(item, "id", np.nan))
        if scan_id is not None:
            id_to_index[int(scan_id)] = index

    records = []
    for index, item in enumerate(items):
        phant_id = _safe_str(getattr(item, "phant_id", ""))
        if not phant_id:
            continue
        diam = _safe_float(getattr(item, "tum_diam", np.nan))
        rad = _safe_float(getattr(item, "tum_rad", np.nan))
        if diam is None and rad is not None and rad > 0:
            diam = 2.0 * rad
        tum_x = _safe_float(getattr(item, "tum_x", np.nan))
        tum_y = _safe_float(getattr(item, "tum_y", np.nan))
        has_tumor = bool((diam is not None and diam > 0) or (tum_x is not None and tum_y is not None))
        emp_id = _safe_float(getattr(item, "emp_ref_id", np.nan))
        if emp_id is None or int(emp_id) not in id_to_index:
            continue
        adi_id = _safe_float(getattr(item, "adi_ref_id", np.nan))
        adi_index = id_to_index.get(int(adi_id)) if adi_id is not None else None
        group = "tumor" if has_tumor else ("healthy_fib" if "F" in phant_id.upper() else "adipose_only")
        records.append(
            {
                "generation": generation,
                "index": index,
                "scan_id": int(_safe_float(getattr(item, "id", index + 1)) or index + 1),
                "phant_id": phant_id,
                "group": group,
                "has_tumor": int(has_tumor),
                "tum_diam_cm": diam if has_tumor else np.nan,
                "tum_x_cm": tum_x if has_tumor and tum_x is not None else np.nan,
                "tum_y_cm": tum_y if has_tumor and tum_y is not None else np.nan,
                "birads": _safe_float(getattr(item, "birads", np.nan)) or np.nan,
                "ant_rad_cm": _safe_float(getattr(item, "ant_rad", np.nan)) or np.nan,
                "n_session": _safe_float(getattr(item, "n_session", np.nan)) or np.nan,
                "tum_shape": _safe_str(getattr(item, "tum_shape", "")) if has_tumor else "",
                "emp_index": id_to_index[int(emp_id)],
                "adi_index": adi_index,
            }
        )
    return records


def signal_pair_features(raw: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Signal features of the calibrated scan, and again after rotational-mean removal."""
    freqs = bmid_frequencies_hz()
    raw = np.asarray(raw, dtype=complex)
    cal = calibrate_empty(raw, reference)
    feats = signal_features(raw, cal, freqs)
    feats.update(signal_features(raw, remove_rotational_mean(cal), freqs, prefix="rsig"))
    feats.update(scattering_descriptors(cal, freqs))
    return feats


def process_scan(task: tuple[dict, np.ndarray, np.ndarray]) -> tuple[dict, dict[str, float]]:
    record, raw, reference = task
    return record, scan_features(record["generation"], record["ant_rad_cm"], raw, reference)


def process_signal(task: tuple[dict, np.ndarray, np.ndarray]) -> tuple[dict, dict[str, float]]:
    record, raw, reference = task
    return record, signal_pair_features(raw, reference)


def fair_twins(records: list[dict], cap: int = TWIN_K) -> list[dict]:
    """Same-sitting healthy baseline with one shared reference count per sitting.

    Every kept scan in a sitting is averaged against the same number of other
    healthy scans: the fewest any of them has, capped at ``cap``. A tumour scan
    is never given a cleaner baseline than the healthy scans from that sitting,
    so the count cannot give away the label. Sittings where a scan has no other
    healthy scan are dropped.
    """
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        if record["group"] in ("tumor", "healthy_fib"):
            grouped.setdefault((record["phant_id"], record["n_session"]), []).append(record)
    kept = []
    for group in grouped.values():
        healthy = [record["index"] for record in group if record["group"] == "healthy_fib"]
        usable = []
        for record in group:
            twins = [index for index in healthy if index != record["index"]]
            if twins:
                usable.append((record, twins))
        if not usable:
            continue
        n_use = min(cap, min(len(twins) for _, twins in usable))
        for record, twins in usable:
            rng = np.random.default_rng(record["scan_id"])
            chosen = sorted(int(i) for i in rng.choice(np.asarray(twins), n_use, replace=False))
            kept.append({**record, "twin_indices": chosen})
    return kept


def attach_twins(records: list[dict], k: int = TWIN_K, min_k: int | None = None) -> list[dict]:
    """Give every tumor/healthy scan other healthy scans of the same phantom and session.

    Uses up to ``k`` of them (at least ``min_k``). The baseline is chosen the same
    way for both classes (random, seeded by scan id), so its choice carries no
    label information. Scans without ``min_k`` such twins are dropped.
    """
    need = k if min_k is None else min_k
    kept = []
    for record in records:
        if record["group"] not in ("tumor", "healthy_fib"):
            continue
        twins = [
            other["index"]
            for other in records
            if other["group"] == "healthy_fib"
            and other["phant_id"] == record["phant_id"]
            and other["n_session"] == record["n_session"]
            and other["index"] != record["index"]
        ]
        if len(twins) < need:
            continue
        rng = np.random.default_rng(record["scan_id"])
        take = min(k, len(twins))
        kept.append({**record, "twin_indices": sorted(int(i) for i in rng.choice(twins, take, replace=False))})
    return kept


def iter_tasks(generation: str, fd_path: Path, records: list[dict], reference: str):
    _, cube = _load_fd_cube(str(fd_path))
    for record in records:
        if reference == "twin":
            ref = np.mean([cube[i] for i in record["twin_indices"]], axis=0)
        else:
            ref = cube[record[f"{reference}_index"]]
        yield record, np.asarray(cube[record["index"]], dtype=complex), np.asarray(ref, dtype=complex)
    del cube


def refresh_signal_columns(features_path: Path, generation_specs, reference: str, workers: int) -> None:
    """Recompute sig_/rsig_ columns and keep the existing image columns."""
    with np.load(features_path, allow_pickle=False) as old:
        old_names = [str(name) for name in old["feature_names"]]
        image_names = [name for name in old_names if name.startswith(("img_", "imgl_", "d4img_"))]
        image_index = [old_names.index(name) for name in image_names]
        old_ids = [int(v) for v in old["scan_id"]]
        old_generation = np.array([str(v) for v in old["generation"]])
        old_scan_id = np.array([int(v) for v in old["scan_id"]])
        old_image = np.array(old["X"][:, image_index], dtype=float) if image_index else np.zeros((len(old_ids), 0))
        meta = {key: np.array(old[key]) for key in old.files if key not in ("X", "feature_names")}
    by_generation: dict[str, list[dict]] = {}
    for generation, fd_path, md_name in generation_specs:
        records = scan_records(generation, fd_path.parent / md_name)
        if reference == "adi":
            records = [r for r in records if r["adi_index"] is not None]
        elif reference == "twin":
            records = fair_twins(records)
        by_generation[generation] = records

    got: dict[tuple[str, int], dict[str, float]] = {}
    started = time.time()
    with get_context("spawn").Pool(workers) as pool:
        for generation, fd_path, _md_name in generation_specs:
            records = by_generation[generation]
            wanted = {old_ids[i] for i in range(len(old_ids)) if old_generation[i] == generation}
            records = [r for r in records if r["scan_id"] in wanted]
            print(f"{generation}: refreshing {len(records)} signal vectors", flush=True)
            found: dict[int, dict[str, float]] = {}
            for record, feats in pool.imap(process_signal, iter_tasks(generation, fd_path, records, reference), chunksize=4):
                found[record["scan_id"]] = feats
            missing = wanted - set(found)
            if missing:
                raise RuntimeError(f"{generation} refresh missed {len(missing)} scans, e.g. {sorted(missing)[:5]}")
            for scan_id, feats in found.items():
                got[(generation, scan_id)] = feats
            print(f"  {generation} done  {time.time() - started:.0f}s", flush=True)

    signal_rows = [got[(gen, scan_id)] for gen, scan_id in zip(old_generation, old_scan_id)]
    signal_names = sorted(signal_rows[0])
    names = sorted(signal_names + image_names)
    matrix = np.empty((len(signal_rows), len(names)), dtype=float)
    name_at = {name: i for i, name in enumerate(names)}
    for row_i, feats in enumerate(signal_rows):
        for name in signal_names:
            matrix[row_i, name_at[name]] = feats.get(name, np.nan)
        for name_i, name in enumerate(image_names):
            matrix[row_i, name_at[name]] = old_image[row_i, name_i]
    np.savez_compressed(features_path, X=matrix, feature_names=np.array(names), **meta)
    print(json.dumps({"path": str(features_path), "n_scans": len(signal_rows), "n_features": len(names), "seconds": round(time.time() - started, 1)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Scans per generation (0 = all).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--refresh-signal", action="store_true", help="Recompute signal columns in an existing feature file; keep image columns.")
    parser.add_argument(
        "--reference",
        choices=("emp", "adi", "twin"),
        default="emp",
        help="Subtract the empty-chamber scan (emp), the adipose-only scan (adi; skips scans without one), "
        "or the mean of other healthy scans of the same phantom and session (twin).",
    )
    parser.add_argument("--gens", nargs="+", default=[g for g, _, _ in GENERATIONS])
    args = parser.parse_args()

    suffix = "" if args.reference == "emp" else f"_{args.reference}_{'_'.join(args.gens)}"
    features_path = OUT_DIR / f"features{suffix}.npz"
    if args.refresh_signal:
        specs = [item for item in GENERATIONS if item[0] in args.gens]
        refresh_signal_columns(features_path, specs, args.reference, args.workers)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    feature_rows: list[dict[str, float]] = []
    started = time.time()

    with get_context("spawn").Pool(args.workers) as pool:
        for generation, fd_path, md_name in GENERATIONS:
            if generation not in args.gens:
                continue
            records = scan_records(generation, fd_path.parent / md_name)
            if args.reference == "adi":
                records = [r for r in records if r["adi_index"] is not None]
            elif args.reference == "twin":
                records = fair_twins(records)
            if args.limit:
                by_group: dict[str, list[dict]] = {}
                for record in records:
                    by_group.setdefault(record["group"], []).append(record)
                records = [r for group in by_group.values() for r in group[: args.limit]]
            print(f"{generation}: {len(records)} scans", flush=True)
            done = 0
            for record, feats in pool.imap(process_scan, iter_tasks(generation, fd_path, records, args.reference), chunksize=2):
                rows.append(record)
                feature_rows.append(feats)
                done += 1
                if done % 50 == 0 or done == len(records):
                    print(f"  {generation} {done}/{len(records)}  {time.time() - started:.0f}s", flush=True)

    names = sorted(feature_rows[0])
    matrix = np.array([[row.get(name, np.nan) for name in names] for row in feature_rows], dtype=float)
    meta = {column: np.array([row[column] for row in rows]) for column in META_COLUMNS}
    np.savez_compressed(features_path, X=matrix, feature_names=np.array(names), **meta)
    summary = {
        "n_scans": len(rows),
        "n_features": len(names),
        "groups": {group: int(sum(r["group"] == group for r in rows)) for group in ("tumor", "healthy_fib", "adipose_only")},
        "reference": args.reference,
        "generations": args.gens,
        "seconds": round(time.time() - started, 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

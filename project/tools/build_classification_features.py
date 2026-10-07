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

from classification.predict import scan_features  # noqa: E402
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


def process_scan(task: tuple[dict, np.ndarray, np.ndarray]) -> tuple[dict, dict[str, float]]:
    record, raw, reference = task
    return record, scan_features(record["generation"], record["ant_rad_cm"], raw, reference)


def attach_twins(records: list[dict], k: int = TWIN_K) -> list[dict]:
    """Give every tumor/healthy scan k other healthy scans of the same phantom and session.

    The baseline is chosen the same way for both classes (random, seeded by scan id),
    so its choice carries no label information. Scans without k such twins are dropped.
    """
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
        if len(twins) < k:
            continue
        rng = np.random.default_rng(record["scan_id"])
        kept.append({**record, "twin_indices": sorted(int(i) for i in rng.choice(twins, k, replace=False))})
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Scans per generation (0 = all).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
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
                records = attach_twins(records)
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

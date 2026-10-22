"""Cache breast-window sinograms for the first-scan convolutional test.

One generation is written at a time so a large cube can be released before
the next is opened. Re-running skips a generation whose part file is already
present.

Usage:
    python tools/build_sinograms.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.sinogram import sinogram_pair  # noqa: E402
from data_loader.bmid_loader import bmid_frequencies_hz  # noqa: E402
from tools.build_classification_features import GENERATIONS, iter_tasks, scan_records  # noqa: E402

OUT_DIR = PROJECT / "results" / "classification"
PARTS = OUT_DIR / "sinogram_parts"


def build_generation(generation: str, fd_path: Path, md_name: str) -> Path:
    part = PARTS / f"{generation}.npz"
    if part.exists():
        print(f"{generation}: part exists, skipped", flush=True)
        return part
    records = [record for record in scan_records(generation, fd_path.parent / md_name) if record["group"] in ("tumor", "healthy_fib")]
    freqs = bmid_frequencies_hz()
    images = []
    kept = []
    started = time.time()
    print(f"{generation}: {len(records)} tumour or healthy scans", flush=True)
    for done, (record, raw, reference) in enumerate(iter_tasks(generation, fd_path, records, "emp"), start=1):
        images.append(sinogram_pair(raw, reference, freqs))
        kept.append(record)
        if done % 100 == 0 or done == len(records):
            print(f"  {generation} {done}/{len(records)}  {time.time() - started:.0f}s", flush=True)
    stack = np.stack(images, axis=0)
    meta = {key: np.array([record[key] for record in kept]) for key in ("generation", "scan_id", "phant_id", "group", "has_tumor", "n_session")}
    PARTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(part, X=stack, **meta)
    print(f"{generation}: wrote {stack.shape}  {time.time() - started:.0f}s", flush=True)
    return part


def merge_parts(parts: list[Path]) -> Path:
    arrays = [np.load(path, allow_pickle=False) for path in parts]
    X = np.concatenate([item["X"] for item in arrays], axis=0)
    meta = {key: np.concatenate([item[key] for item in arrays]) for key in ("generation", "scan_id", "phant_id", "group", "has_tumor", "n_session")}
    path = OUT_DIR / "sinograms_emp.npz"
    np.savez_compressed(path, X=X, **meta)
    for item in arrays:
        item.close()
    print(f"merged {X.shape} -> {path}", flush=True)
    return path


def main() -> None:
    parts = []
    for generation, fd_path, md_name in GENERATIONS:
        parts.append(build_generation(generation, fd_path, md_name))
    merge_parts(parts)


if __name__ == "__main__":
    main()

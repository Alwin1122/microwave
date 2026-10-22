"""Follow-up with one earlier healthy scan instead of three.

Generation 1 has 89 tumour scans with one same-sitting healthy scan, but only
24 with three. This builds that larger set (gen1 and gen3) and scores it with
the same grouped phantom split used by the published model. Gen2 is omitted:
only one of its tumour scans has a same-sitting healthy scan.
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.models import classifiers, select_columns  # noqa: E402
from tools.build_classification_features import (  # noqa: E402
    GENERATIONS,
    META_COLUMNS,
    iter_tasks,
    process_signal,
    scan_records,
)
from tools.train_tumor_models import balanced_fair_mask, folds_for, load  # noqa: E402

OUT_NPZ = PROJECT / "results" / "classification" / "features_twin_upto3_signal.npz"
OUT_JSON = PROJECT / "results" / "classification" / "followup_upto3.json"
CURRENT = PROJECT / "results" / "classification" / "features_twin_gen1_gen3.npz"
SMALL_CM = 2.0


def fair_twins(records: list[dict], cap: int = 3) -> list[dict]:
    """Same-sitting healthy baseline. Every kept scan in a sitting uses the same count.

    The count is the smallest number of other healthy scans available to any kept
    scan, capped at ``cap``. A tumour is never averaged against more references
    than a healthy scan from that sitting, so the count cannot give away the label.
    """
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        if record["group"] in ("tumor", "healthy_fib"):
            grouped.setdefault((record["phant_id"], record["n_session"]), []).append(record)
    kept = []
    for group in grouped.values():
        healthy = [r["index"] for r in group if r["group"] == "healthy_fib"]
        usable = []
        for record in group:
            twins = [i for i in healthy if i != record["index"]]
            if twins:
                usable.append((record, twins))
        if not usable:
            continue
        n_use = min(cap, min(len(twins) for _, twins in usable))
        for record, twins in usable:
            rng = np.random.default_rng(record["scan_id"])
            chosen = sorted(int(i) for i in rng.choice(np.array(twins), n_use, replace=False))
            kept.append({**record, "twin_indices": chosen})
    return kept


def build(k: int, workers: int, min_k: int = 1) -> None:
    rows, feature_rows = [], []
    started = time.time()
    specs = [item for item in GENERATIONS if item[0] in ("gen1", "gen3")]
    with get_context("spawn").Pool(workers) as pool:
        for generation, fd_path, md_name in specs:
            records = fair_twins(scan_records(generation, fd_path.parent / md_name), cap=k)
            print(f"{generation}: {len(records)} scans, same reference count within each sitting", flush=True)
            done = 0
            tasks = iter_tasks(generation, fd_path, records, "twin")
            for record, feats in pool.imap(process_signal, tasks, chunksize=4):
                rows.append(record)
                feature_rows.append(feats)
                done += 1
                if done % 40 == 0 or done == len(records):
                    print(f"  {generation} {done}/{len(records)}  {time.time() - started:.0f}s", flush=True)
    names = sorted(feature_rows[0])
    matrix = np.array([[row.get(name, np.nan) for name in names] for row in feature_rows], dtype=float)
    meta = {column: np.array([row[column] for row in rows]) for column in META_COLUMNS}
    np.savez_compressed(OUT_NPZ, X=matrix, feature_names=np.array(names), **meta)
    print(f"saved {OUT_NPZ.name} {matrix.shape} in {time.time() - started:.0f}s", flush=True)


def _auc(y, score, mask):
    sel = np.asarray(mask)
    if int(sel.sum()) < 8 or len(np.unique(y[sel])) < 2:
        return None
    return round(float(roc_auc_score(y[sel], score[sel])), 3)


def cross_val(table: dict, prefixes: tuple[str, ...], model_name: str, pool_name: str) -> dict:
    fair = balanced_fair_mask(table)
    y = table["has_tumor"].astype(int)
    cols = select_columns(table["feature_names"], prefixes)
    X = table["X"][:, cols]
    groups = table["groups"]
    gen = np.array([str(g) for g in table["generation"]])
    diam = table["tum_diam_cm"].astype(float)
    idx = np.flatnonzero(fair)
    strat = table["generation_code"] * 2 + y
    model = classifiers()[model_name]
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=-1)
    score = np.zeros(len(y))
    count = np.zeros(len(y))
    for rep in range(3):
        for test in folds_for(idx, strat, groups, 5, rep):
            excluded = set(groups[test])
            base = fair if pool_name == "fair" else np.ones(len(y), dtype=bool)
            train = np.flatnonzero(base & ~np.isin(groups, list(excluded)))
            fit = clone(model).fit(X[train], y[train])
            pred = fit.predict_proba(X[test])[:, list(fit.classes_).index(1)]
            score[test] += pred
            count[test] += 1
    got = count > 0
    score[got] /= count[got]
    small = (y == 0) | ((y == 1) & np.isfinite(diam) & (diam <= SMALL_CM))
    large = (y == 0) | ((y == 1) & np.isfinite(diam) & (diam > SMALL_CM))
    return {
        "model": f"{'+'.join(prefixes)}|{model_name}|{pool_name}",
        "n": int(got.sum()),
        "phantoms": int(len(set(groups[got]))),
        "by_generation_n": {g: int((got & (gen == g)).sum()) for g in ("gen1", "gen3")},
        "auc": _auc(y, score, got),
        "small": _auc(y, score, got & small),
        "large": _auc(y, score, got & large),
        "gen1": _auc(y, score, got & (gen == "gen1")),
        "gen3": _auc(y, score, got & (gen == "gen3")),
        "n_small_tumors": int((got & (y == 1) & np.isfinite(diam) & (diam <= SMALL_CM)).sum()),
    }


def report(path: Path) -> list[dict]:
    table = load(path)
    specs = (
        (("rsig_",), "logistic", "fair"),
        (("rsig_",), "random_forest", "fair"),
        (("sig_", "rsig_"), "random_forest", "fair_plus_other"),
    )
    rows = []
    for prefixes, model_name, pool_name in specs:
        row = cross_val(table, prefixes, model_name, pool_name)
        row["file"] = path.name
        rows.append(row)
        print(path.name, row["model"], "auc", row["auc"], "small", row["small"], "gen1", row["gen1"], "gen3", row["gen3"], "n", row["n"], row["by_generation_n"], flush=True)
    return rows


def main() -> None:
    build(k=3, workers=6, min_k=1)
    built = report(OUT_NPZ)
    payload = {"one_to_three_healthy_scans": built}
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

"""Check two fair ways to use more breasts.

1. Count how many scans have 1, 2, or 3 earlier healthy scans in the same sitting.
2. Score every breast that has both a tumour scan and a healthy scan, even when
   those scans were taken in different sittings. A setup-only model (antenna
   radius, sitting number, generation) is the leak check: if it scores well,
   the sitting itself is giving away the answer and the signal score is not real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.models import classifiers, select_columns  # noqa: E402
from tools.build_classification_features import GENERATIONS, scan_records  # noqa: E402
from tools.train_tumor_models import balanced_fair_mask, load, setup_matrix  # noqa: E402

OUT = PROJECT / "results" / "classification" / "more_breasts_probe.json"
SMALL_CM = 2.0


def twin_counts() -> dict:
    out = {}
    for gen, fd_path, md_name in GENERATIONS:
        records = [r for r in scan_records(gen, fd_path.parent / md_name) if r["group"] in ("tumor", "healthy_fib")]
        counts = {1: 0, 2: 0, 3: 0}
        by_class = {"tumor": {1: 0, 2: 0, 3: 0}, "healthy_fib": {1: 0, 2: 0, 3: 0}}
        for record in records:
            n = sum(
                1
                for other in records
                if other["group"] == "healthy_fib"
                and other["phant_id"] == record["phant_id"]
                and other["n_session"] == record["n_session"]
                and other["index"] != record["index"]
            )
            for k in (1, 2, 3):
                if n >= k:
                    counts[k] += 1
                    by_class[record["group"]][k] += 1
        out[gen] = {"scans": len(records), "with_at_least": counts, "by_class": by_class}
        print(gen, "scans", len(records), "twins>=", counts, flush=True)
    return out


def _auc(y, score, mask) -> float | None:
    sel = np.asarray(mask)
    if int(sel.sum()) < 8 or len(np.unique(y[sel])) < 2:
        return None
    return float(roc_auc_score(y[sel], score[sel]))


def score_paired_phantoms(table: dict) -> dict:
    y = table["has_tumor"].astype(int)
    groups = np.array([str(g) for g in table["groups"]])
    gen = np.array([str(g) for g in table["generation"]])
    diam = table["tum_diam_cm"].astype(float)
    paired = np.zeros(len(y), dtype=bool)
    for phantom in sorted(set(groups)):
        rows = groups == phantom
        if len(np.unique(y[rows])) == 2:
            paired[rows] = True
    names = table["feature_names"]
    X = table["X"][:, select_columns(names, ("sig_", "rsig_"))]
    S = setup_matrix(table)
    idx = np.flatnonzero(paired)
    strat = np.array([int(g[-1]) for g in gen]) * 2 + y
    signal_model = classifiers()["random_forest"]
    signal_model.set_params(n_estimators=160, n_jobs=-1)
    setup_model = classifiers()["random_forest"]
    setup_model.set_params(n_estimators=160, n_jobs=-1)
    sig_score = np.zeros(len(y))
    set_score = np.zeros(len(y))
    count = np.zeros(len(y))
    for rep in range(3):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
        for tr, te in splitter.split(np.zeros(len(idx)), strat[idx], groups[idx]):
            train, test = idx[tr], idx[te]
            held = set(groups[test])
            extra = np.flatnonzero(~np.isin(groups, list(held)))
            train = np.unique(np.concatenate([train, extra]))
            sig = clone(signal_model).fit(X[train], y[train])
            stp = clone(setup_model).fit(S[train], y[train])
            sig_score[test] += sig.predict_proba(X[test])[:, list(sig.classes_).index(1)]
            set_score[test] += stp.predict_proba(S[test])[:, list(stp.classes_).index(1)]
            count[test] += 1
        print(f"paired repeat {rep + 1}/3", flush=True)
    got = count > 0
    sig_score[got] /= count[got]
    set_score[got] /= count[got]
    small = (y == 0) | ((y == 1) & (diam <= SMALL_CM))
    large = (y == 0) | ((y == 1) & (diam > SMALL_CM))

    def pack(score):
        return {
            "auc": _auc(y, score, got),
            "small": _auc(y, score, got & small),
            "large": _auc(y, score, got & large),
            "by_generation": {g: _auc(y, score, got & (gen == g)) for g in ("gen1", "gen2", "gen3")},
        }

    session_gap = {}
    for g in ("gen1", "gen2", "gen3"):
        sel = paired & (gen == g)
        session_gap[g] = {
            "tumor_session_mean": float(table["n_session"][sel & (y == 1)].mean()) if np.any(sel & (y == 1)) else None,
            "healthy_session_mean": float(table["n_session"][sel & (y == 0)].mean()) if np.any(sel & (y == 0)) else None,
            "n": int(sel.sum()),
            "phantoms": int(len(set(groups[sel]))),
        }
    current = balanced_fair_mask(table)
    return {
        "paired_scans": int(paired.sum()),
        "paired_phantoms": int(len(set(groups[paired]))),
        "current_fair_scans": int(current.sum()),
        "signal": pack(sig_score),
        "setup_only": pack(set_score),
        "session_means": session_gap,
    }


def main() -> None:
    twins = twin_counts()
    table = load(PROJECT / "results" / "classification" / "features.npz")
    paired = score_paired_phantoms(table)
    payload = {"twins": twins, "paired_phantoms": paired}
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(paired, indent=2), flush=True)


if __name__ == "__main__":
    main()

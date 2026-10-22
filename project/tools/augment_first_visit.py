"""First-visit training with extra breasts built from real tumour echoes.

A first visit has no earlier healthy scan. The fair test stays real UM-BMID
scans only. Extra training examples are made by adding a tumour echo, measured
on a training breast, onto a different training breast of the same generation.
Test breasts never donate an echo and never receive one.

Usage:
    python tools/augment_first_visit.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.features import signal_features  # noqa: E402
from classification.models import classifiers  # noqa: E402
from data_loader.bmid_loader import bmid_frequencies_hz  # noqa: E402
from reconstruction.bmid_geometry import remove_rotational_mean  # noqa: E402
from tools.build_classification_features import GENERATIONS, scan_records  # noqa: E402
from tools.train_tumor_models import balanced_fair_mask, load  # noqa: E402
from data_loader.bmid_loader import _load_fd_cube  # noqa: E402

OUT_PATH = PROJECT / "results" / "classification" / "first_visit_augment.json"
SMALL_CM = 2.0
SEED = 0
OUTER = 5
REPEATS = 3
PASTES_SMALL = 4
PASTES_LARGE = 1


def feature_vector(raw: np.ndarray, calibrated: np.ndarray, names: list[str], freqs: np.ndarray) -> np.ndarray:
    feats = signal_features(raw, calibrated, freqs)
    feats.update(signal_features(raw, remove_rotational_mean(calibrated), freqs, prefix="rsig"))
    return np.nan_to_num(np.array([feats.get(name, 0.0) for name in names], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def choose_hosts(host_indexes: list[int], banned_phantom: str, phantoms: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """Other breasts of this generation. ``banned_phantom`` is the echo's source."""
    pool = [i for i in host_indexes if phantoms[i] != banned_phantom]
    if not pool:
        return []
    take = min(k, len(pool))
    return [int(i) for i in rng.choice(pool, size=take, replace=False)]


def paste_echo(host_cal: np.ndarray, echo: np.ndarray, scale: float, phase: float) -> np.ndarray:
    """Add a measured tumour echo onto another breast. Position labels are not used."""
    return host_cal + (scale * np.exp(1j * phase)) * echo


def _auc(y: np.ndarray, score: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) < 8 or len(np.unique(y[mask])) < 2:
        return None
    return float(roc_auc_score(y[mask], score[mask]))


def main() -> None:
    started = time.time()
    table = load(PROJECT / "results" / "classification" / "features.npz")
    names = [n for n in table["feature_names"] if n.startswith(("sig_", "rsig_"))]
    cols = [table["feature_names"].index(n) for n in names]
    real = table["X"][:, cols]
    fair = balanced_fair_mask(table)
    groups = table["groups"]
    y = table["has_tumor"].astype(int)
    diam = table["tum_diam_cm"].astype(float)
    generation = np.array([str(g) for g in table["generation"]])
    phantoms = np.array([str(p) for p in table["phant_id"]])
    sessions = table["n_session"].astype(float)
    scan_ids = table["scan_id"].astype(int)
    row_of = {(generation[i], int(scan_ids[i])): i for i in range(len(y))}

    freqs = bmid_frequencies_hz()
    calibrated: dict[tuple[str, int], np.ndarray] = {}
    raws: dict[tuple[str, int], np.ndarray] = {}
    echoes = []
    healthy_rows: dict[str, list[int]] = {g: [] for g, _, _ in GENERATIONS}

    for gen, fd_path, md_name in GENERATIONS:
        records = scan_records(gen, fd_path.parent / md_name)
        by_index = {r["index"]: r for r in records}
        _, cube = _load_fd_cube(str(fd_path))
        print(f"loaded {gen} cube {cube.shape}", flush=True)
        for record in records:
            if record["group"] not in ("tumor", "healthy_fib"):
                continue
            key = (gen, record["scan_id"])
            if key not in row_of:
                continue
            raw = np.asarray(cube[record["index"]], dtype=np.complex64)
            empty = np.asarray(cube[record["emp_index"]], dtype=np.complex64)
            calibrated[key] = raw - empty
            raws[key] = raw
            row = row_of[key]
            if record["group"] == "healthy_fib":
                healthy_rows[gen].append(row)
        for record in records:
            if record["group"] != "tumor":
                continue
            partners = [
                other["index"]
                for other in records
                if other["group"] == "healthy_fib"
                and other["phant_id"] == record["phant_id"]
                and other["n_session"] == record["n_session"]
            ]
            if len(partners) < 2:
                continue
            key = (gen, record["scan_id"])
            if key not in calibrated:
                continue
            partner_cal = [calibrated[(gen, by_index[i]["scan_id"])] for i in partners if (gen, by_index[i]["scan_id"]) in calibrated]
            if len(partner_cal) < 2:
                continue
            echo = calibrated[key] - np.mean(partner_cal, axis=0)
            echoes.append({"row": row_of[key], "gen": gen, "echo": np.asarray(echo, dtype=np.complex64), "diam": float(diam[row_of[key]])})
        del cube
        print(f"  {gen} echoes {sum(1 for e in echoes if e['gen'] == gen)} healthy {len(healthy_rows[gen])}", flush=True)

    eval_idx = np.flatnonzero(fair & np.isin(table["group"], ("tumor", "healthy_fib")))
    small_energy = {}
    for gen in healthy_rows:
        energies = [float(np.sum(np.abs(e["echo"]) ** 2)) for e in echoes if e["gen"] == gen and 0 < e["diam"] <= SMALL_CM]
        small_energy[gen] = float(np.median(energies)) if energies else None

    def synthetics(train_rows: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        allowed = set(int(i) for i in train_rows)
        made_x = []
        for echo in echoes:
            if echo["row"] not in allowed:
                continue
            hosts = [i for i in healthy_rows[echo["gen"]] if i in allowed and phantoms[i] != phantoms[echo["row"]]]
            k = PASTES_SMALL if echo["diam"] <= SMALL_CM else PASTES_LARGE
            chosen = choose_hosts(hosts, phantoms[echo["row"]], phantoms, k, rng)
            variants = [(1.0, echo["diam"])]
            energy = float(np.sum(np.abs(echo["echo"]) ** 2))
            if echo["diam"] > SMALL_CM and small_energy[echo["gen"]] and energy > 0:
                variants.append((float(np.sqrt(small_energy[echo["gen"]] / energy)), SMALL_CM))
            for host in chosen:
                key = (generation[host], int(scan_ids[host]))
                for scale, _diam in variants:
                    phase = float(rng.uniform(0, 2 * np.pi))
                    jitter = float(rng.uniform(0.85, 1.15))
                    synth = paste_echo(calibrated[key], echo["echo"], scale * jitter, phase)
                    made_x.append(feature_vector(raws[key], synth, names, freqs))
        if not made_x:
            return np.zeros((0, len(names))), np.zeros(0, dtype=int)
        return np.vstack(made_x), np.ones(len(made_x), dtype=int)

    def run(use_synth: bool, per_generation: bool) -> dict:
        scores = np.zeros(len(y))
        counts = np.zeros(len(y))
        forest = classifiers()["random_forest"]
        forest.set_params(n_estimators=160, n_jobs=-1)
        logistic = classifiers()["logistic"]
        collected = {name: (np.zeros(len(y)), np.zeros(len(y))) for name in ("logistic", "random_forest")}
        for rep in range(REPEATS):
            splitter = StratifiedGroupKFold(n_splits=OUTER, shuffle=True, random_state=SEED + rep)
            strat = generation_code_labels(generation, y)
            for fold_i, (train, test) in enumerate(splitter.split(np.zeros(len(eval_idx)), strat[eval_idx], groups[eval_idx]), start=1):
                train_rows = eval_idx[train]
                test_rows = eval_idx[test]
                extra_train = np.flatnonzero(~np.isin(groups, list(set(groups[test_rows]))) & np.isin(table["group"], ("tumor", "healthy_fib")))
                train_rows = np.unique(np.concatenate([train_rows, extra_train]))
                rng = np.random.default_rng(SEED + 1000 * rep + fold_i)
                x_extra, y_extra = synthetics(train_rows, rng) if use_synth else (np.zeros((0, len(names))), np.zeros(0, dtype=int))
                if len(y_extra) > 2 * int((y[train_rows] == 1).sum()):
                    keep = rng.choice(len(y_extra), size=2 * int((y[train_rows] == 1).sum()), replace=False)
                    x_extra, y_extra = x_extra[keep], y_extra[keep]
                x_tr = real[train_rows] if len(y_extra) == 0 else np.vstack([real[train_rows], x_extra])
                y_tr = y[train_rows] if len(y_extra) == 0 else np.concatenate([y[train_rows], y_extra])
                x_te = real[test_rows]
                for name, model in (("logistic", logistic), ("random_forest", forest)):
                    fitted = clone(model).fit(x_tr, y_tr)
                    pred = fitted.predict_proba(x_te)[:, list(fitted.classes_).index(1)]
                    collected[name][0][test_rows] += pred
                    collected[name][1][test_rows] += 1
                print(f"  {'synth' if use_synth else 'real'} rep {rep + 1} fold {fold_i} extra {len(y_extra)}", flush=True)
        out = {}
        for name, (scores, counts) in collected.items():
            scored = counts > 0
            mean_score = np.zeros(len(y))
            mean_score[scored] = scores[scored] / counts[scored]
            out[name] = metrics(y, mean_score, scored, diam, generation)
        return out

    # Per-generation synthetic path is handled inside run when the flag is set.
    # The placeholder branch above is replaced by predict_per_generation.
    print("baseline, real scans only", flush=True)
    baseline = run(False, False)
    print(json.dumps(baseline), flush=True)
    print("augmented, pooled", flush=True)
    augmented = run(True, False)
    print(json.dumps(augmented), flush=True)
    result = {
        "method": "Paste a tumour echo measured on a training breast onto other training breasts of the same generation. Test breasts are real and untouched.",
        "baseline_real_only": baseline,
        "augmented": augmented,
        "seconds": round(time.time() - started, 1),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH}", flush=True)


def generation_code_labels(generation: np.ndarray, y: np.ndarray) -> np.ndarray:
    code = np.array([int(str(g)[-1]) for g in generation])
    return code * 2 + y


def metrics(y, score, mask, diam, generation) -> dict:
    def pack(sel):
        auc = _auc(y, score, sel)
        return None if auc is None else round(auc, 3)

    small = mask & ((y == 0) | ((y == 1) & (diam <= SMALL_CM)))
    large = mask & ((y == 0) | ((y == 1) & (diam > SMALL_CM)))
    out = {
        "auc": pack(mask),
        "auc_small_tumours": pack(small),
        "auc_larger_tumours": pack(large),
        "n": int(mask.sum()),
        "n_small_tumours": int((mask & (y == 1) & (diam <= SMALL_CM)).sum()),
    }
    for gen in sorted(set(generation[mask])):
        out[f"auc_{gen}"] = pack(mask & (generation == gen))
    return out


if __name__ == "__main__":
    main()

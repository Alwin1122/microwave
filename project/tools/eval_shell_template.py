"""First visit: subtract the average healthy scan of the same fat shell.

A first patient has no earlier scan of their own breast. The template is the
mean empty-chamber-calibrated healthy scan of other breasts that share the
adipose shell and the generation. It is rebuilt inside each test fold from
training breasts only, so a test breast never enters its own background.
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
from tools.build_classification_features import GENERATIONS, process_signal, scan_records  # noqa: E402
from tools.train_tumor_models import balanced_fair_mask, folds_for, load  # noqa: E402

FEATURES = PROJECT / "results" / "classification" / "features.npz"
OUT_JSON = PROJECT / "results" / "classification" / "shell_template.json"
SMALL_CM = 2.0


def shell_name(phant_id: str) -> str:
    """Adipose shell id. A11F2 and A1F2 are different shells."""
    return str(phant_id).upper().split("F", 1)[0]


def load_calibrated(table: dict) -> tuple[np.ndarray, np.ndarray]:
    """Raw and empty-subtracted scans aligned to ``table`` rows."""
    keys = {(str(g), int(s)): i for i, (g, s) in enumerate(zip(table["generation"], table["scan_id"]))}
    raws = cals = None
    filled = np.zeros(len(keys), dtype=bool)
    for generation, fd_path, md_name in GENERATIONS:
        records = scan_records(generation, fd_path.parent / md_name)
        wanted = [r for r in records if (generation, r["scan_id"]) in keys]
        if not wanted:
            continue
        print(f"loading {generation} ({len(wanted)} scans)", flush=True)
        from data_loader.bmid_loader import _load_fd_cube

        _, cube = _load_fd_cube(str(fd_path))
        if raws is None:
            shape = np.asarray(cube[wanted[0]["index"]]).shape
            raws = np.empty((len(keys), *shape), dtype=np.complex64)
            cals = np.empty_like(raws)
        for record in wanted:
            row = keys[(generation, record["scan_id"])]
            raw = np.asarray(cube[record["index"]], dtype=np.complex64)
            empty = np.asarray(cube[record["emp_index"]], dtype=np.complex64)
            raws[row] = raw
            cals[row] = raw - empty
            filled[row] = True
        del cube
    if raws is None or not filled.all():
        raise RuntimeError(f"missing calibrated scans: {int((~filled).sum())}")
    return raws, cals


def template_bank(cals, healthy, generation, shells, groups, excluded: set):
    """One background per breast: other training breasts of the same shell.

    The breast itself is left out, and so is every breast in the test fold.
    """
    bank = {}
    fallback = {}
    for gen in sorted(set(generation)):
        for phantom in sorted(set(groups)):
            rows = np.flatnonzero(healthy & (generation == gen) & (groups != phantom) & ~np.isin(groups, list(excluded)))
            if rows.size == 0:
                continue
            fallback[(gen, phantom)] = cals[rows].mean(axis=0)
            for shell in sorted(set(shells[rows])):
                sel = rows[shells[rows] == shell]
                if sel.size:
                    bank[(gen, shell, phantom)] = cals[sel].mean(axis=0)
    return bank, fallback


def featurize(pool, raws, cals, names, bank, fallback, generation, shells, groups) -> tuple[np.ndarray, int]:
    matrix = np.zeros((len(raws), len(names)), dtype=float)
    pending = []
    covered = 0
    for i in range(len(raws)):
        key = (generation[i], shells[i], groups[i])
        template = bank.get(key)
        if template is not None:
            covered += 1
        else:
            template = fallback.get((generation[i], groups[i]))
        if template is None:
            continue
        reference = raws[i] - cals[i] + template
        pending.append(({"index": i}, raws[i], reference))
        if len(pending) == 48:
            for record, feats in pool.imap(process_signal, pending, chunksize=4):
                matrix[record["index"]] = [feats[name] for name in names]
            pending = []
    if pending:
        for record, feats in pool.imap(process_signal, pending, chunksize=4):
            matrix[record["index"]] = [feats[name] for name in names]
    return matrix, covered


def _auc(y, score, mask):
    sel = np.asarray(mask)
    if int(sel.sum()) < 8 or len(np.unique(y[sel])) < 2:
        return None
    return round(float(roc_auc_score(y[sel], score[sel])), 3)


def main() -> None:
    started = time.time()
    table = load(FEATURES)
    y = table["has_tumor"].astype(int)
    fair = balanced_fair_mask(table)
    groups = table["groups"]
    healthy = np.array([str(g) for g in table["group"]]) == "healthy_fib"
    generation = np.array([str(g) for g in table["generation"]])
    shells = np.array([shell_name(p) for p in table["phant_id"]])
    diam = table["tum_diam_cm"].astype(float)
    names = [n for n in table["feature_names"] if n.startswith(("sig_", "rsig_"))]
    raws, cals = load_calibrated(table)
    print("shells", {s: int((shells == s).sum()) for s in sorted(set(shells))}, flush=True)

    specs = (
        (("sig_", "rsig_"), "random_forest", "fair_plus_other"),
        (("rsig_",), "logistic", "fair"),
        (("sig_",), "logistic", "fair"),
    )
    acc = {
        key: {"base": np.zeros(len(y)), "shell": np.zeros(len(y)), "count": np.zeros(len(y))}
        for key in specs
    }
    strat = table["generation_code"] * 2 + y
    idx = np.flatnonzero(fair)
    sample = process_signal(({"index": 0}, raws[0], raws[0] - cals[0]))[1]
    feat_names = [n for n in names if n in sample]

    with get_context("spawn").Pool(6) as pool:
        for rep in range(3):
            for fold_i, test in enumerate(folds_for(idx, strat, groups, 5, rep), start=1):
                excluded = set(groups[test])
                bank, fallback = template_bank(cals, healthy, generation, shells, groups, excluded)
                matrix, covered = featurize(pool, raws, cals, feat_names, bank, fallback, generation, shells, groups)
                print(f"repeat {rep + 1} fold {fold_i} same-shell {covered}/{len(raws)}", flush=True)
                for prefixes, model_name, pool_name in specs:
                    cols = [feat_names.index(n) for n in feat_names if any(n.startswith(p) for p in prefixes)]
                    base_cols = select_columns(table["feature_names"], prefixes)
                    model = classifiers()[model_name]
                    if "n_jobs" in model.get_params():
                        model.set_params(n_jobs=-1)
                    base_mask = fair if pool_name == "fair" else np.ones(len(y), dtype=bool)
                    train = np.flatnonzero(base_mask & ~np.isin(groups, list(excluded)))
                    base_fit = clone(model).fit(table["X"][np.ix_(train, base_cols)], y[train])
                    shell_fit = clone(model).fit(matrix[np.ix_(train, cols)], y[train])
                    key = acc[(prefixes, model_name, pool_name)]
                    base_col = list(base_fit.classes_).index(1)
                    shell_col = list(shell_fit.classes_).index(1)
                    key["base"][test] += base_fit.predict_proba(table["X"][np.ix_(test, base_cols)])[:, base_col]
                    key["shell"][test] += shell_fit.predict_proba(matrix[np.ix_(test, cols)])[:, shell_col]
                    key["count"][test] += 1
            print(f"repeat {rep + 1}/3 done  {time.time() - started:.0f}s", flush=True)

    small = (y == 0) | ((y == 1) & np.isfinite(diam) & (diam <= SMALL_CM))
    large = (y == 0) | ((y == 1) & np.isfinite(diam) & (diam > SMALL_CM))
    rows = []
    for prefixes, model_name, pool_name in specs:
        key = acc[(prefixes, model_name, pool_name)]
        got = key["count"] > 0
        for kind in ("base", "shell"):
            score = np.zeros(len(y))
            score[got] = key[kind][got] / key["count"][got]
            row = {
                "kind": kind,
                "model": f"{'+'.join(prefixes)}|{model_name}|{pool_name}",
                "auc": _auc(y, score, got & fair),
                "small": _auc(y, score, got & fair & small),
                "large": _auc(y, score, got & fair & large),
                "gen1": _auc(y, score, got & fair & (generation == "gen1")),
                "gen2": _auc(y, score, got & fair & (generation == "gen2")),
                "gen3": _auc(y, score, got & fair & (generation == "gen3")),
                "n": int((got & fair).sum()),
            }
            rows.append(row)
            print(row["kind"], row["model"], "auc", row["auc"], "small", row["small"], "gen1", row["gen1"], "gen3", row["gen3"], flush=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT_JSON} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

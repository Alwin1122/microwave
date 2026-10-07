"""Fair, nested evaluation and final training of the tumor models.

Test scans: recording sessions that contain both tumor and healthy scans,
subsampled so every session holds as many tumor as healthy scans (setup and
session drift then carry no label information). Splits are grouped by
phantom name, so no phantom is ever in both training and test.

For every outer test fold the feature set, model type and training pool
(fair scans only, or fair plus every other scan of non-test phantoms) are
chosen by an inner grouped cross-validation on the training phantoms only.
The reported numbers are therefore never "the best of many tries".

Tasks: tumor detection, size class, diameter, position. Each is compared with
guessing, a setup-only model (antenna radius, session, generation), and for
detection/position the blind rule score / image bright spot. The detection
alarm threshold is also chosen on training phantoms only.

Sections: single scan (empty chamber removed), gen3 with a fat-only scan
(best-case reference), and same-phantom healthy baseline (follow-up scan
compared with earlier healthy scans of the same phantom and session).

Usage:
    python tools/train_tumor_models.py [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.models import SIZE_CLASSES, binary_metrics, classifiers, regressors, select_columns, size_class  # noqa: E402
from reconstruction.bmid_geometry import BMID_GEOMETRY  # noqa: E402

OUT_DIR = PROJECT / "results" / "classification"
SUMMARY_PATH = OUT_DIR / "final_summary.json"
MODEL_PATH = OUT_DIR / "tumor_models.joblib"
MODEL_TWIN_PATH = OUT_DIR / "tumor_models_twin.joblib"
TWIN_FEATURES = OUT_DIR / "features_twin_gen1_gen3.npz"
TWIN_KEY = "twin_baseline"
PARTIAL_PATH = OUT_DIR / "final_summary_partial.json"
# A setup-only model predicts position as well as the trained one (sessions share
# tumour positions), so no learned position model is saved; the untrained image
# spot is used instead.
NOT_SAVED = ("position",)
SEED = 0
OUTER_FOLDS = 5
INNER_FOLDS = 4
N_JOBS = -2
FEATURE_SETS = {
    "signal": ("sig_",),
    "signal_rot": ("rsig_",),
    "image": ("img_", "imgl_", "d4img_"),
    "all": ("sig_", "rsig_", "img_", "imgl_", "d4img_"),
}
POOLS = ("fair", "fair_plus_other")
SPOT = ("imgl_peak_x_cm", "imgl_peak_y_cm")
RULE = "img_rule_max_confidence"


# ---------------------------------------------------------------- data


def load(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    table = {key: data[key] for key in data.files}
    table["feature_names"] = [str(name) for name in table["feature_names"]]
    table["X"] = np.nan_to_num(table["X"].astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    table["generation_code"] = np.array([int(str(g)[-1]) for g in table["generation"]])
    table["groups"] = np.array([str(p) for p in table["phant_id"]])
    table["session"] = np.array([f"{g}:{s}" for g, s in zip(table["generation"], table["n_session"])])
    keep = np.isin(table["group"], ("tumor", "healthy_fib"))
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in table.items()}


def balanced_fair_mask(table: dict, seed: int = SEED) -> np.ndarray:
    """Sessions with both classes, subsampled to equal tumor/healthy counts per session."""
    rng = np.random.default_rng(seed)
    y = table["has_tumor"].astype(int)
    keep = np.zeros(len(y), dtype=bool)
    for session in sorted(set(table["session"])):
        pos = np.flatnonzero((table["session"] == session) & (y == 1))
        neg = np.flatnonzero((table["session"] == session) & (y == 0))
        n = min(len(pos), len(neg))
        if n:
            keep[rng.choice(pos, n, replace=False)] = True
            keep[rng.choice(neg, n, replace=False)] = True
    return keep


def setup_matrix(table: dict) -> np.ndarray:
    return np.nan_to_num(np.column_stack([table["ant_rad_cm"], table["n_session"], table["generation_code"]]).astype(float), nan=-1.0)


def folds_for(index: np.ndarray, strat: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Grouped stratified test-index arrays (indices into the full table)."""
    n_groups = len(set(groups[index]))
    n_folds = max(2, min(n_folds, n_groups))
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [index[test] for _, test in splitter.split(np.zeros(len(index)), strat[index], groups[index])]


# ---------------------------------------------------------------- tasks


def with_threads(estimator, n_jobs: int):
    """Fresh copy; inside candidate workers use one core so parallelism stays across candidates."""
    model = clone(estimator)
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=n_jobs)
    return model


class Task:
    def __init__(self, name: str, kind: str, rows: np.ndarray, target: np.ndarray, models: tuple[str, ...], strat: np.ndarray):
        self.name, self.kind, self.rows, self.target, self.models, self.strat = name, kind, rows, target, models, strat
        self.pools = POOLS

    def fit_predict(self, model_name: str, X_train, y_train, X_test, n_jobs: int = 1):
        if self.kind == "regression":
            y_train = y_train.reshape(len(y_train), -1)
            out = np.zeros((len(X_test), y_train.shape[1]))
            for col in range(y_train.shape[1]):
                out[:, col] = with_threads(regressors()[model_name], n_jobs).fit(X_train, y_train[:, col]).predict(X_test)
            return out
        model = with_threads(classifiers()[model_name], n_jobs).fit(X_train, y_train)
        if self.kind == "binary":
            return model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        return model.predict(X_test)

    def score(self, y_true, pred) -> float:
        """Higher is better."""
        if self.kind == "binary":
            return binary_metrics(y_true, pred)["auc"]
        if self.kind == "multiclass":
            return float(balanced_accuracy_score(y_true, pred))
        y_true = y_true.reshape(len(y_true), -1)
        return -float(np.mean(np.linalg.norm(pred - y_true, axis=1)))


def candidates(task: Task) -> list[tuple[str, str, str]]:
    return [(fs, model, pool) for fs in FEATURE_SETS for model in task.models for pool in task.pools]


def train_index(task: Task, fair: np.ndarray, groups: np.ndarray, excluded: set, pool: str) -> np.ndarray:
    base = task.rows & (fair if pool == "fair" else np.ones_like(fair))
    return np.flatnonzero(base & ~np.isin(groups, list(excluded)))


def predict_fold(task, table, fair, cand, test_idx, excluded, n_jobs: int = 1):
    fs, model, pool = cand
    cols = select_columns(table["feature_names"], FEATURE_SETS[fs])
    train = train_index(task, fair, table["groups"], excluded, pool)
    return task.fit_predict(model, table["X"][np.ix_(train, cols)], task.target[train], table["X"][np.ix_(test_idx, cols)], n_jobs)


def cv_predictions(task, table, fair, cand, eval_idx, n_folds, seed, extra_excluded=frozenset()):
    """Out-of-fold predictions of one candidate over ``eval_idx`` (fair rows)."""
    pred = {}
    for test_idx in folds_for(eval_idx, task.strat, table["groups"], n_folds, seed):
        excluded = set(table["groups"][test_idx]) | set(extra_excluded)
        for i, p in zip(test_idx, predict_fold(task, table, fair, cand, test_idx, excluded)):
            pred[i] = p
    return np.array([pred[i] for i in eval_idx])


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Cut-off with the highest balanced accuracy (equal weight on tumours found and healthy cleared)."""
    fpr, tpr, thresholds = roc_curve(y, score)
    return float(np.clip(thresholds[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def nested(task: Task, table: dict, fair: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Nested predictions; for detection also the alarm threshold, chosen on training phantoms only."""
    eval_idx = np.flatnonzero(task.rows & fair)
    preds, thresholds, chosen = {}, {}, []
    for test_idx in folds_for(eval_idx, task.strat, table["groups"], OUTER_FOLDS, seed):
        outer_groups = set(table["groups"][test_idx])
        inner_idx = np.setdiff1d(eval_idx, test_idx)
        cands = candidates(task)
        inner_preds = Parallel(n_jobs=N_JOBS)(
            delayed(cv_predictions)(task, table, fair, cand, inner_idx, INNER_FOLDS, seed, outer_groups) for cand in cands
        )
        scores = [task.score(task.target[inner_idx], p) for p in inner_preds]
        best_i = int(np.argmax(scores))
        best = cands[best_i]
        chosen.append("|".join(best))
        threshold = best_threshold(task.target[inner_idx], inner_preds[best_i]) if task.kind == "binary" else np.nan
        for i, p in zip(test_idx, predict_fold(task, table, fair, best, test_idx, outer_groups, n_jobs=-1)):
            preds[i] = p
            thresholds[i] = threshold
    return eval_idx, np.array([preds[i] for i in eval_idx]), chosen, np.array([thresholds[i] for i in eval_idx])


def setup_only(task: Task, table: dict, fair: np.ndarray, seed: int) -> np.ndarray:
    eval_idx = np.flatnonzero(task.rows & fair)
    S = setup_matrix(table)
    model = "random_forest"
    pred = {}
    for test_idx in folds_for(eval_idx, task.strat, table["groups"], OUTER_FOLDS, seed):
        train = train_index(task, fair, table["groups"], set(table["groups"][test_idx]), "fair")
        for i, p in zip(test_idx, task.fit_predict(model, S[train], task.target[train], S[test_idx], n_jobs=-1)):
            pred[i] = p
    return np.array([pred[i] for i in eval_idx])


# ---------------------------------------------------------------- metrics


def per_generation(table, idx, fn) -> dict:
    out = {}
    for gen in sorted(set(table["generation"][idx])):
        sel = table["generation"][idx] == gen
        if sel.sum() >= 4:
            out[str(gen)] = fn(sel)
    return out


def position_metrics(truth, pred) -> dict:
    err = np.hypot(pred[:, 0] - truth[:, 0], pred[:, 1] - truth[:, 1])
    return {"mean_cm": float(err.mean()), "median_cm": float(np.median(err)), "within_2cm": float(np.mean(err <= 2.0))}


def summarize(values: list[dict]) -> dict:
    """Mean and spread of numeric metrics over repeats."""
    keys = [k for k, v in values[0].items() if isinstance(v, (int, float))]
    return {k: {"mean": float(np.mean([v[k] for v in values])), "std": float(np.std([v[k] for v in values]))} for k in keys}


# ---------------------------------------------------------------- main


def build_tasks(table: dict) -> dict[str, Task]:
    y = table["has_tumor"].astype(int)
    gen = table["generation_code"]
    diam = table["tum_diam_cm"].astype(float)
    xy = np.column_stack([table["tum_x_cm"], table["tum_y_cm"]]).astype(float)
    tumor = (y == 1) & np.isfinite(diam)
    located = (y == 1) & np.all(np.isfinite(xy), axis=1)
    size_idx = np.array([SIZE_CLASSES.index(size_class(d)) if np.isfinite(d) else -1 for d in diam])
    return {
        "detection": Task("detection", "binary", np.ones(len(y), dtype=bool), y, ("logistic", "random_forest"), gen * 2 + y),
        "size_class": Task("size_class", "multiclass", tumor, size_idx, ("logistic", "random_forest"), size_idx),
        "diameter": Task("diameter", "regression", tumor, np.nan_to_num(diam), ("ridge", "random_forest"), gen),
        "position": Task("position", "regression", located, np.nan_to_num(xy), ("ridge", "random_forest"), gen),
    }


def evaluate(table: dict, fair: np.ndarray, repeats: int, label: str, done: dict, save) -> dict:
    """Run every task not already in ``done``; ``save(out)`` is called after each task."""
    tasks = build_tasks(table)
    if fair.all():
        for task in tasks.values():
            task.pools = ("fair",)
    names = table["feature_names"]
    out = dict(done)
    out["n_fair"] = int(fair.sum())
    out["n_fair_tumor"] = int((fair & (table["has_tumor"] == 1)).sum())
    out["n_fair_phantoms"] = int(len(set(table["groups"][fair])))
    out["fair_by_generation"] = {str(g): int(np.sum(fair & (table["generation"] == g))) for g in sorted(set(table["generation"]))}

    for task in tasks.values():
        if task.name in done:
            print(f"[{label}] {task.name}: already done, skipped", flush=True)
            continue
        started = time.time()
        runs, setup_runs, choices = [], [], []
        for rep in range(repeats):
            idx, pred, chosen, thresholds = nested(task, table, fair, SEED + rep)
            setup_pred = setup_only(task, table, fair, SEED + rep)
            truth = task.target[idx]
            choices += chosen
            if task.kind == "binary":
                m = binary_metrics(truth, pred)
                flag = pred >= thresholds
                m["tuned_sensitivity"] = float(np.mean(flag[truth == 1]))
                m["tuned_specificity"] = float(np.mean(~flag[truth == 0]))
                m["tuned_threshold"] = float(np.mean(thresholds))
                for g, v in per_generation(table, idx, lambda s: binary_metrics(truth[s], pred[s])["auc"]).items():
                    m[f"auc_{g}"] = v
                runs.append(m)
                setup_runs.append(binary_metrics(truth, setup_pred))
            elif task.kind == "multiclass":
                m = {"balanced_accuracy": float(balanced_accuracy_score(truth, pred))}
                for g, v in per_generation(table, idx, lambda s: float(np.mean(truth[s] == pred[s]))).items():
                    m[f"accuracy_{g}"] = v
                if rep == 0:
                    present = sorted(set(truth))
                    out_conf = {"classes": [SIZE_CLASSES[c] for c in present], "matrix": confusion_matrix(truth, pred, labels=present).tolist()}
                runs.append(m)
                setup_runs.append({"balanced_accuracy": float(balanced_accuracy_score(truth, setup_pred))})
            elif task.name == "diameter":
                err = np.abs(pred[:, 0] - truth)
                m = {"mae_cm": float(err.mean())}
                for g, v in per_generation(table, idx, lambda s: float(err[s].mean())).items():
                    m[f"mae_cm_{g}"] = v
                runs.append(m)
                setup_runs.append({"mae_cm": float(np.mean(np.abs(setup_pred[:, 0] - truth)))})
            else:
                m = position_metrics(truth, pred)
                for g, v in per_generation(table, idx, lambda s: position_metrics(truth[s], pred[s])["mean_cm"]).items():
                    m[f"mean_cm_{g}"] = v
                runs.append(m)
                setup_runs.append(position_metrics(truth, setup_pred))

        result = {
            "n": int(len(idx)),
            "nested": summarize(runs),
            "setup_only": summarize(setup_runs),
            "chosen_candidates": {c: choices.count(c) for c in sorted(set(choices))},
        }
        truth = task.target[idx]
        if task.kind == "binary":
            result["rule_based"] = binary_metrics(truth, table["X"][idx, names.index(RULE)], 0.45)
            result["chance_auc"] = 0.5
        elif task.kind == "multiclass":
            result["confusion_first_repeat"] = out_conf
            result["chance_balanced_accuracy"] = 1.0 / len(set(truth))
        elif task.name == "diameter":
            result["average_guess_mae_cm"] = float(np.mean(np.abs(truth - truth.mean())))
            result["diameter_values_cm"] = sorted({round(float(d), 2) for d in truth})
        else:
            spot = table["X"][np.ix_(idx, [names.index(SPOT[0]), names.index(SPOT[1])])]
            result["centre_guess"] = position_metrics(truth, np.zeros_like(truth))
            result["image_spot"] = position_metrics(truth, spot)
            result["image_spot_by_generation"] = per_generation(table, idx, lambda s: position_metrics(truth[s], spot[s]))
            result["centre_guess_by_generation"] = per_generation(table, idx, lambda s: position_metrics(truth[s], np.zeros_like(truth[s])))
            result["n_positions"] = int(len({(round(a, 2), round(b, 2)) for a, b in truth}))

        # Plain cross-validated score of every candidate (comparison only, and the
        # basis for the final model's choice; the nested score above is the honest one).
        eval_idx = np.flatnonzero(task.rows & fair)
        cands = candidates(task)
        all_preds = Parallel(n_jobs=N_JOBS)(
            delayed(cv_predictions)(task, table, fair, cand, eval_idx, OUTER_FOLDS, SEED) for cand in cands
        )
        table_scores = {"|".join(c): task.score(task.target[eval_idx], p) for c, p in zip(cands, all_preds)}
        result["all_candidates_cv"] = table_scores
        result["final_choice"] = max(table_scores, key=table_scores.get)
        if task.kind == "binary":
            final_pred = all_preds[list(table_scores).index(result["final_choice"])]
            result["final_threshold"] = best_threshold(task.target[eval_idx], final_pred)
        out[task.name] = result
        save(out)
        print(f"[{label}] {task.name}: nested {json.dumps({k: round(v['mean'], 3) for k, v in result['nested'].items()})} "
              f"setup {json.dumps({k: round(v['mean'], 3) for k, v in result['setup_only'].items()})}  {time.time() - started:.0f}s", flush=True)
    return out


def fit_final(table: dict, fair: np.ndarray, results: dict) -> dict:
    tasks = build_tasks(table)
    bundle = {"feature_names": table["feature_names"], "feature_sets": FEATURE_SETS, "size_classes": SIZE_CLASSES,
              "geometry": {g: vars(v) for g, v in BMID_GEOMETRY.items()}, "models": {}}
    for name, task in tasks.items():
        if name in NOT_SAVED:
            continue
        fs, model_name, pool = results[name]["final_choice"].split("|")
        cols = select_columns(table["feature_names"], FEATURE_SETS[fs])
        train = train_index(task, fair, table["groups"], set(), pool)
        X, y = table["X"][np.ix_(train, cols)], task.target[train]
        if task.kind == "regression":
            y = y.reshape(len(y), -1)
            fitted = [clone(regressors()[model_name]).fit(X, y[:, c]) for c in range(y.shape[1])]
        else:
            fitted = clone(classifiers()[model_name]).fit(X, y)
        bundle["models"][name] = {"feature_set": fs, "model": model_name, "pool": pool, "columns": cols, "estimator": fitted, "n_train": int(len(train))}
        if "final_threshold" in results[name]:
            bundle["models"][name]["threshold"] = results[name]["final_threshold"]
    return bundle


def save_bundle(bundle: dict, path: Path) -> dict:
    joblib.dump(bundle, path)
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("estimator", "columns")} for k, v in bundle["models"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--refit", action="store_true", help="Only refit the saved models from an existing final_summary.json.")
    args = parser.parse_args()

    table = load(OUT_DIR / "features.npz")
    fair = balanced_fair_mask(table)
    twin = load(TWIN_FEATURES) if TWIN_FEATURES.exists() else None
    twin_fair = balanced_fair_mask(twin) if twin is not None else None
    if args.refit:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        summary["final_models"] = save_bundle(fit_final(table, fair, summary["main"]), MODEL_PATH)
        if twin is not None and TWIN_KEY in summary:
            summary["final_models_twin"] = save_bundle(fit_final(twin, twin_fair, summary[TWIN_KEY]), MODEL_TWIN_PATH)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"refit {MODEL_PATH}")
        return
    started = time.time()
    protocol = {
        "version": 2,
        "test_set": "sessions with both tumor and healthy scans, balanced per session",
        "grouping": "phantom name (never in both train and test)",
        "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "repeats": args.repeats,
        "candidates": {"feature_sets": list(FEATURE_SETS), "pools": list(POOLS)},
    }
    summary = {"protocol": protocol}
    if PARTIAL_PATH.exists():
        previous = json.loads(PARTIAL_PATH.read_text(encoding="utf-8"))
        if previous.get("protocol") == protocol:
            summary = previous
            print(f"resuming from {PARTIAL_PATH}", flush=True)

    def saver(section: str):
        def save(block: dict) -> None:
            summary[section] = block
            PARTIAL_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return save

    summary["main"] = evaluate(table, fair, args.repeats, "main", summary.get("main", {}), saver("main"))

    adi_path = OUT_DIR / "features_adi_gen3.npz"
    if adi_path.exists():
        adi = load(adi_path)
        key = "reference_gen3_fat_scan"
        summary[key] = evaluate(adi, np.ones(len(adi["group"]), dtype=bool), args.repeats, "gen3 fat-only reference", summary.get(key, {}), saver(key))

    if twin is not None:
        summary[TWIN_KEY] = evaluate(twin, twin_fair, args.repeats, "same-phantom healthy baseline", summary.get(TWIN_KEY, {}), saver(TWIN_KEY))
        summary["final_models_twin"] = save_bundle(fit_final(twin, twin_fair, summary[TWIN_KEY]), MODEL_TWIN_PATH)

    summary["final_models"] = save_bundle(fit_final(table, fair, summary["main"]), MODEL_PATH)
    summary["seconds"] = round(time.time() - started, 1)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    PARTIAL_PATH.unlink(missing_ok=True)
    print(f"saved {SUMMARY_PATH} and {MODEL_PATH}")


if __name__ == "__main__":
    main()

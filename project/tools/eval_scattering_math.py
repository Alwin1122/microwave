"""First-visit physics features, with the follow-up model left as the control.

A first visit has only the empty-chamber scan. The new features are the
scattering-operator spectrum, the Rayleigh-Mie slope and the time and angle
compactness defined in classification/scattering_math.py. They are added to
the existing signal features. The comparison is fixed in advance:

    detection, first visit: random forest on signal features
                            versus the same forest on signal features plus physics
    size and criticality:   logistic regression, same two feature sets
    follow-up detection:    logistic regression on the rotationally cleaned
                            follow-up signal, versus that signal plus physics
                            computed on the same follow-up residual

Nothing else is searched. Splits are grouped by phantom. The test set is the
balanced fair set (sessions that contain both classes). Repeats use different
fold seeds. Follow-up scans keep the matched healthy-reference count already
used for the published follow-up result.

Usage:
    python tools/eval_scattering_math.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.criticality import criticality_class, criticality_index  # noqa: E402
from classification.models import SIZE_CLASSES, binary_metrics, classifiers, size_class  # noqa: E402
from classification.scattering_math import MATH_FEATURES, range_bearing_cm, scattering_descriptors  # noqa: E402
from data_loader.bmid_loader import _load_fd_cube, bmid_frequencies_hz  # noqa: E402
from reconstruction.bmid_geometry import BMID_GEOMETRY  # noqa: E402
from reconstruction.reconstruction_manager import build_circular_antenna_array  # noqa: E402
from tools.build_classification_features import GENERATIONS, fair_twins, scan_records  # noqa: E402
from tools.train_tumor_models import SEED, balanced_fair_mask, load  # noqa: E402

OUT_DIR = PROJECT / "results" / "classification"
EMP_CACHE = OUT_DIR / "features_math_emp.npz"
TWIN_CACHE = OUT_DIR / "features_math_twin.npz"
OUT_PATH = OUT_DIR / "math_analysis.json"
FOLDS = 5
REPEATS = 3
SMALL_CM = 2.0


def _matrix(rows: list[dict[str, float]], names: tuple[str, ...]) -> np.ndarray:
    return np.array([[row.get(name, 0.0) for name in names] for row in rows], dtype=float)


def build_cache(reference: str, path: Path) -> None:
    """Physics features for every tumour and healthy-fibroglandular scan."""
    freqs = bmid_frequencies_hz()
    generations, scan_ids, rows, peaks = [], [], [], []
    started = time.time()
    for generation, fd_path, md_name in GENERATIONS:
        records = scan_records(generation, fd_path.parent / md_name)
        records = [record for record in records if record["group"] in ("tumor", "healthy_fib")]
        if reference == "twin":
            records = fair_twins(records)
        print(f"{reference} {generation}: loading {len(records)} scans", flush=True)
        _, cube = _load_fd_cube(str(fd_path))
        for done, record in enumerate(records, start=1):
            raw = np.asarray(cube[record["index"]], dtype=np.complex64)
            if reference == "twin":
                ref = np.mean([np.asarray(cube[i], dtype=np.complex64) for i in record["twin_indices"]], axis=0)
            else:
                ref = np.asarray(cube[record["emp_index"]], dtype=np.complex64)
            described = scattering_descriptors(raw - ref, freqs)
            rows.append(described)
            peaks.append(described["math_peak_antenna"])
            generations.append(generation)
            scan_ids.append(record["scan_id"])
            if done % 100 == 0 or done == len(records):
                print(f"  {generation} {done}/{len(records)} {time.time() - started:.0f}s", flush=True)
        del cube
    matrix = _matrix(rows, MATH_FEATURES)
    np.savez_compressed(
        path,
        generation=np.array(generations),
        scan_id=np.array(scan_ids, dtype=int),
        feature_names=np.array(MATH_FEATURES),
        X=matrix,
        peak_antenna=np.array(peaks, dtype=float),
    )
    print(f"saved {path} {matrix.shape}", flush=True)


def align(table: dict, cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (math features, peak antenna) in the same row order as ``table``."""
    cache = np.load(cache_path, allow_pickle=False)
    names = [str(name) for name in cache["feature_names"]]
    cols = [names.index(name) for name in MATH_FEATURES]
    lookup = {(str(gen), int(scan)): i for i, (gen, scan) in enumerate(zip(cache["generation"], cache["scan_id"]))}
    math = np.zeros((len(table["has_tumor"]), len(MATH_FEATURES)))
    peaks = np.zeros(len(table["has_tumor"]))
    found = np.zeros(len(table["has_tumor"]), dtype=bool)
    for i, (gen, scan) in enumerate(zip(table["generation"], table["scan_id"])):
        key = (str(gen), int(scan))
        if key not in lookup:
            continue
        row = lookup[key]
        math[i] = cache["X"][row, cols]
        peaks[i] = cache["peak_antenna"][row]
        found[i] = True
    if not found.all():
        missing = int((~found).sum())
        print(f"{cache_path.name}: {missing} table rows have no physics features", flush=True)
    return np.nan_to_num(math, nan=0.0, posinf=0.0, neginf=0.0), peaks


def _signal_columns(names: list[str]) -> list[int]:
    return [i for i, name in enumerate(names) if name.startswith(("sig_", "rsig_"))]


def _rot_columns(names: list[str]) -> list[int]:
    return [i for i, name in enumerate(names) if name.startswith("rsig_")]


def _folds(y: np.ndarray, groups: np.ndarray, seed: int):
    smallest_class = int(min(np.bincount(y.astype(int))))
    n_splits = min(FOLDS, smallest_class, len(set(groups)))
    splitter = StratifiedGroupKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def _fit_predict(model, x_train, y_train, x_test, kind: str):
    fitted = clone(model).fit(x_train, y_train)
    if kind == "binary":
        return fitted.predict_proba(x_test)[:, list(fitted.classes_).index(1)], fitted
    return fitted.predict(x_test), fitted


def _binary_block(y: np.ndarray, score: np.ndarray, diam: np.ndarray, generation: np.ndarray) -> dict:
    row = binary_metrics(y, score)
    row["auc_gen1"] = _auc(y, score, generation == "gen1")
    row["auc_gen3"] = _auc(y, score, generation == "gen3")
    small = (y == 0) | ((y == 1) & np.isfinite(diam) & (diam <= SMALL_CM))
    row["auc_small"] = _auc(y, score, small)
    return row


def _auc(y: np.ndarray, score: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) < 8 or len(np.unique(y[mask])) < 2:
        return None
    return float(roc_auc_score(y[mask], score[mask]))


def _mean_std(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def cross_validate(x: np.ndarray, y: np.ndarray, groups: np.ndarray, model, kind: str, extra: dict | None = None) -> dict:
    """Out-of-fold scores. Each repeat is one number; the result is mean and spread."""
    repeats = []
    coefficients = []
    for rep in range(REPEATS):
        score = np.zeros(len(y) if kind == "binary" else len(y), dtype=float)
        pred = np.zeros(len(y), dtype=int)
        for train, test in _folds(y, groups, SEED + rep):
            out, fitted = _fit_predict(model, x[train], y[train], x[test], kind)
            if kind == "binary":
                score[test] = out
                if hasattr(fitted, "named_steps") and "logisticregression" in fitted.named_steps:
                    coefficients.append(fitted.named_steps["logisticregression"].coef_.ravel())
            else:
                pred[test] = out
        if kind == "binary":
            block = _binary_block(y, score, extra["diam"], extra["generation"])
        else:
            block = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)), "n": int(len(y))}
            for gen in ("gen1", "gen3"):
                mask = extra["generation"] == gen
                block[f"accuracy_{gen}"] = float(np.mean(pred[mask] == y[mask])) if int(mask.sum()) else None
        repeats.append(block)
    keys = [key for key in repeats[0] if key != "n"]
    summary = {key: _mean_std(repeats, key) for key in keys}
    summary["n"] = int(len(y))
    if coefficients:
        stacked = np.vstack(coefficients)
        summary["coefficient_mean"] = stacked.mean(axis=0).tolist()
        summary["coefficient_std"] = stacked.std(axis=0).tolist()
    return summary


def _forest():
    model = classifiers()["random_forest"]
    model.set_params(n_jobs=-1)
    return model


def _logistic():
    return classifiers()["logistic"]


def identification(table: dict, fair: np.ndarray, peaks: np.ndarray, math_x: np.ndarray) -> dict:
    """Closed-form range and bearing on fair tumour scans. No training."""
    names = table["feature_names"]
    delay_col = MATH_FEATURES.index("math_delay_peak_ns")
    tumor = fair & (table["has_tumor"] == 1) & np.isfinite(table["tum_x_cm"].astype(float))
    errors = []
    by_gen: dict[str, list[float]] = {}
    image_errors = []
    has_image = "imgl_peak_x_cm" in names
    ix = names.index("imgl_peak_x_cm") if has_image else None
    iy = names.index("imgl_peak_y_cm") if has_image else None
    for i in np.flatnonzero(tumor):
        generation = str(table["generation"][i])
        geometry = BMID_GEOMETRY[generation]
        n_ant = 72
        angles = build_circular_antenna_array(
            n_ant,
            radius=1.0,
            angle_offset_deg=geometry.offset_deg,
            clockwise=geometry.clockwise,
            span_deg=geometry.span_deg,
        )
        antenna = int(peaks[i]) % n_ant
        bearing = float(np.arctan2(angles[antenna, 1], angles[antenna, 0]))
        delay_s = float(math_x[i, delay_col]) * 1e-9
        radius_m = float(table["ant_rad_cm"][i]) / 100.0
        if not np.isfinite(radius_m) or radius_m <= 0:
            continue
        x_cm, y_cm = range_bearing_cm(delay_s, bearing, radius_m, geometry.wave_speed)
        err = float(np.hypot(x_cm - table["tum_x_cm"][i], y_cm - table["tum_y_cm"][i]))
        errors.append(err)
        by_gen.setdefault(generation, []).append(err)
        if has_image:
            image_errors.append(float(np.hypot(table["X"][i, ix] - table["tum_x_cm"][i], table["X"][i, iy] - table["tum_y_cm"][i])))

    def pack(values: list[float]) -> dict:
        arr = np.array(values, dtype=float)
        if arr.size == 0:
            return {"n": 0}
        return {"n": int(arr.size), "mean_cm": float(arr.mean()), "median_cm": float(np.median(arr)), "within_2cm": float(np.mean(arr <= 2.0))}

    return {
        "method": "d = R_ant - c t_peak / 2 at the brightest residual antenna. No training.",
        "physics": pack(errors),
        "physics_by_generation": {gen: pack(values) for gen, values in by_gen.items()},
        "image_spot_same_scans": pack(image_errors),
    }


def class_labels(diam: np.ndarray, shape: np.ndarray, kind: str) -> np.ndarray:
    if kind == "size":
        return np.array([SIZE_CLASSES.index(size_class(float(d))) for d in diam], dtype=int)
    indexes = [criticality_index(float(d), str(s)) for d, s in zip(diam, shape)]
    return np.array([("low", "moderate", "high").index(criticality_class(v)) for v in indexes], dtype=int)


def run_pair(x_base, x_extra, y, groups, model, kind, extra) -> dict:
    both = np.column_stack([x_base, x_extra])
    return {
        "baseline": cross_validate(x_base, y, groups, model, kind, extra),
        "with_physics": cross_validate(both, y, groups, clone(model), kind, extra),
    }


def main() -> None:
    started = time.time()
    if not EMP_CACHE.exists():
        build_cache("emp", EMP_CACHE)
    if not TWIN_CACHE.exists():
        build_cache("twin", TWIN_CACHE)

    table = load(OUT_DIR / "features.npz")
    fair = balanced_fair_mask(table)
    math_x, peaks = align(table, EMP_CACHE)
    signal = table["X"][:, _signal_columns(table["feature_names"])]
    base_rows = np.flatnonzero(fair)
    groups = table["groups"][base_rows]
    generation = np.array([str(g) for g in table["generation"][base_rows]])
    diam = table["tum_diam_cm"][base_rows].astype(float)
    y = table["has_tumor"][base_rows].astype(int)
    extra = {"diam": diam, "generation": generation}

    print("first-visit detection", flush=True)
    detection = run_pair(signal[base_rows], math_x[base_rows], y, groups, _forest(), "binary", extra)
    print("first-visit size and criticality", flush=True)
    tumor_rows = base_rows[y == 1]
    tumor_groups = table["groups"][tumor_rows]
    tumor_gen = np.array([str(g) for g in table["generation"][tumor_rows]])
    tumor_diam = table["tum_diam_cm"][tumor_rows].astype(float)
    tumor_extra = {"diam": tumor_diam, "generation": tumor_gen}
    size_y = class_labels(tumor_diam, table["tum_shape"][tumor_rows], "size")
    crit_y = class_labels(tumor_diam, table["tum_shape"][tumor_rows], "criticality")
    size = run_pair(signal[tumor_rows], math_x[tumor_rows], size_y, tumor_groups, _logistic(), "multiclass", tumor_extra)
    criticality = run_pair(signal[tumor_rows], math_x[tumor_rows], crit_y, tumor_groups, _logistic(), "multiclass", tumor_extra)

    print("follow-up control", flush=True)
    twin = load(OUT_DIR / "features_twin_upto3_signal.npz")
    twin_fair = balanced_fair_mask(twin)
    twin_math, _twin_peaks = align(twin, TWIN_CACHE)
    twin_rows = np.flatnonzero(twin_fair)
    twin_signal = twin["X"][:, _rot_columns(twin["feature_names"])]
    twin_y = twin["has_tumor"][twin_rows].astype(int)
    twin_extra = {
        "diam": twin["tum_diam_cm"][twin_rows].astype(float),
        "generation": np.array([str(g) for g in twin["generation"][twin_rows]]),
    }
    follow = run_pair(
        twin_signal[twin_rows],
        twin_math[twin_rows],
        twin_y,
        twin["groups"][twin_rows],
        _logistic(),
        "binary",
        twin_extra,
    )
    twin_tumor = twin_rows[twin_y == 1]
    follow_crit = run_pair(
        twin_signal[twin_tumor],
        twin_math[twin_tumor],
        class_labels(twin["tum_diam_cm"][twin_tumor].astype(float), twin["tum_shape"][twin_tumor], "criticality"),
        twin["groups"][twin_tumor],
        _logistic(),
        "multiclass",
        {"diam": twin["tum_diam_cm"][twin_tumor].astype(float), "generation": np.array([str(g) for g in twin["generation"][twin_tumor]])},
    )

    physics_coef = detection["with_physics"].pop("coefficient_mean", None)
    detection["baseline"].pop("coefficient_mean", None)
    detection["baseline"].pop("coefficient_std", None)
    coef_std = detection["with_physics"].pop("coefficient_std", None)
    # The forest has no logistic coefficients. Fit the interpretable model on the
    # same folds so the paper can show which observable carries the decision.
    print("interpretable logistic on physics features only", flush=True)
    interpretable = cross_validate(math_x[base_rows], y, groups, _logistic(), "binary", extra)
    coef_mean = interpretable.pop("coefficient_mean")
    coef_spread = interpretable.pop("coefficient_std")

    result = {
        "protocol": (
            "Phantom-grouped stratified folds, 5 folds, 3 seeds. Fair sessions only "
            "(equal tumour and healthy counts). One pre-specified model per task. "
            "Physics features are lambda1, effective rank, Rayleigh-Mie slope, "
            "Born volume proxy, temporal kurtosis and angular concentration."
        ),
        "criticality_definition": (
            "eta = (diameter_cm / 2)^3, times 1.5 when the metadata shape is spiculated. "
            "low if eta <= 1, moderate if eta <= 8, high otherwise. Not clinical BI-RADS. "
            "UM-BMID has no benign-versus-malignant label."
        ),
        "first_visit": {
            "detection_random_forest": detection,
            "physics_only_logistic": interpretable,
            "size_class_logistic": size,
            "criticality_logistic": criticality,
            "n_fair": int(fair.sum()),
            "n_fair_tumour": int((fair & (table["has_tumor"] == 1)).sum()),
            "n_spiculated_fair": int(((fair) & (table["has_tumor"] == 1) & (np.array([str(s) for s in table["tum_shape"]]) == "spiculated")).sum()),
        },
        "followup": {
            "detection_logistic_on_rotational_signal": follow,
            "criticality_logistic": follow_crit,
            "n_fair": int(twin_fair.sum()),
        },
        "identification": identification(table, fair, peaks, math_x),
        "physics_logistic_coefficients": {
            "features": list(MATH_FEATURES),
            "mean": coef_mean,
            "std": coef_spread,
            "note": "Standardised logistic weights, averaged over training folds. Positive raises the tumour score.",
        },
        "unused_forest_coefficient_placeholder": {"mean": physics_coef, "std": coef_std},
        "seconds": round(time.time() - started, 1),
    }
    result.pop("unused_forest_coefficient_placeholder")
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "first_signal_auc": detection["baseline"]["auc"]["mean"],
        "first_physics_auc": detection["with_physics"]["auc"]["mean"],
        "first_physics_only_auc": interpretable["auc"]["mean"],
        "first_gen1_signal": detection["baseline"]["auc_gen1"]["mean"],
        "first_gen1_physics": detection["with_physics"]["auc_gen1"]["mean"],
        "follow_signal_auc": follow["baseline"]["auc"]["mean"],
        "follow_physics_auc": follow["with_physics"]["auc"]["mean"],
        "size_base": size["baseline"]["balanced_accuracy"]["mean"],
        "size_physics": size["with_physics"]["balanced_accuracy"]["mean"],
        "crit_base": criticality["baseline"]["balanced_accuracy"]["mean"],
        "crit_physics": criticality["with_physics"]["balanced_accuracy"]["mean"],
        "seconds": result["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

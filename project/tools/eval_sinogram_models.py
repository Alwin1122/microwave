"""Fair-test the sinogram CNN and a PCA-then-logistic classifier.

Both are scored the same way as the official first-scan model: sessions with
both classes, equal counts, and a phantom never in both train and test.
Training may use the other scans of phantoms that are not in the test fold.
The CNN's epoch count is chosen on training phantoms only.

The CNN is the 2022 sinogram network, shrunk for this phantom count.
The PCA logistic is the simulation papers' step (a radar signature reduced
by PCA, then a linear classifier), applied to the measured sinogram.

Usage:
    python tools/eval_sinogram_models.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.models import SEED, binary_metrics  # noqa: E402
from classification.sinogram_net import ADOPT_IF_MEAN_AUC, OFFICIAL_FIRST_SCAN_AUC, fit_predict  # noqa: E402
from tools.train_tumor_models import balanced_fair_mask, folds_for, load  # noqa: E402

OUT = PROJECT / "results" / "classification"
FEATURES = OUT / "features.npz"
SINOGRAMS = OUT / "sinograms_emp.npz"
REPEATS = 3
OUTER_FOLDS = 5
PCA_COMPONENTS = 20


def align_sinograms(table: dict) -> np.ndarray:
    data = np.load(SINOGRAMS, allow_pickle=False)
    index = {(str(gen), int(scan_id)): row for row, (gen, scan_id) in enumerate(zip(data["generation"], data["scan_id"]))}
    rows = []
    for gen, scan_id in zip(table["generation"], table["scan_id"]):
        key = (str(gen), int(scan_id))
        if key not in index:
            raise KeyError(f"sinogram missing for {key}")
        rows.append(index[key])
    return np.asarray(data["X"][rows], dtype=np.float32)


def pca_logistic_scores(train_x, train_y, test_x) -> np.ndarray:
    flat_train = train_x.reshape(len(train_x), -1)
    flat_test = test_x.reshape(len(test_x), -1)
    components = min(PCA_COMPONENTS, flat_train.shape[0] - 1, flat_train.shape[1])
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=SEED),
        LogisticRegression(C=0.5, max_iter=4000, class_weight="balanced"),
    )
    model.fit(flat_train, train_y)
    return model.predict_proba(flat_test)[:, 1]


def grouped_scores(table, fair, images, predict) -> list[dict]:
    y = table["has_tumor"].astype(int)
    groups = table["groups"]
    eval_idx = np.flatnonzero(fair)
    runs = []
    for rep in range(REPEATS):
        started = time.time()
        oof = np.full(len(y), np.nan)
        folds = folds_for(eval_idx, y, groups, OUTER_FOLDS, SEED + rep)
        for fold_i, test_idx in enumerate(folds, start=1):
            banned = set(groups[test_idx])
            train_idx = np.flatnonzero(~np.isin(groups, list(banned)))
            oof[test_idx] = predict(images[train_idx], y[train_idx], groups[train_idx], images[test_idx], SEED + rep * 10 + fold_i)
            print(f"    repeat {rep + 1} fold {fold_i}/{len(folds)}  {time.time() - started:.0f}s", flush=True)
        scored = np.isfinite(oof[eval_idx])
        metrics = binary_metrics(y[eval_idx][scored], oof[eval_idx][scored])
        for gen in ("gen1", "gen3"):
            sel = (table["generation"][eval_idx] == gen) & scored
            metrics[f"auc_{gen}"] = float(roc_auc_score(y[eval_idx][sel], oof[eval_idx][sel])) if sel.sum() > 4 and len(np.unique(y[eval_idx][sel])) == 2 else float("nan")
        runs.append(metrics)
        print(f"  repeat {rep + 1} AUC {metrics['auc']:.3f}", flush=True)
    return runs


def summarize(runs: list[dict]) -> dict:
    keys = [key for key, value in runs[0].items() if isinstance(value, (int, float))]
    return {key: {"mean": float(np.nanmean([run[key] for run in runs])), "std": float(np.nanstd([run[key] for run in runs]))} for key in keys}


def main() -> None:
    if not SINOGRAMS.exists():
        raise SystemExit(f"missing {SINOGRAMS}; run tools/build_sinograms.py first")
    table = load(FEATURES)
    fair = balanced_fair_mask(table)
    images = align_sinograms(table)
    print(f"sinograms {images.shape}  fair {int(fair.sum())}", flush=True)

    print("PCA then logistic", flush=True)
    pca_runs = grouped_scores(
        table,
        fair,
        images,
        lambda tr_x, tr_y, _groups, te_x, _seed: pca_logistic_scores(tr_x, tr_y, te_x),
    )
    print("sinogram CNN", flush=True)
    cnn_runs = grouped_scores(table, fair, images, fit_predict)
    pca = summarize(pca_runs)
    cnn = summarize(cnn_runs)
    winner = "cnn" if cnn["auc"]["mean"] >= pca["auc"]["mean"] else "pca_logistic"
    winning = cnn if winner == "cnn" else pca
    report = {
        "protocol": "Fair sessions, equal class counts, StratifiedGroupKFold by phantom, 5 folds, 3 seeds. Train on every tumour or healthy scan whose phantom is outside the test fold. CNN epochs chosen on a training-phantom split.",
        "official_nested_auc": OFFICIAL_FIRST_SCAN_AUC,
        "adopt_if_mean_auc_above": ADOPT_IF_MEAN_AUC,
        "n_fair": int(fair.sum()),
        "sinogram_shape": list(images.shape[1:]),
        "pca_logistic": pca,
        "sinogram_cnn": cnn,
        "higher_of_the_two": winner,
        "adopted": bool(winning["auc"]["mean"] >= ADOPT_IF_MEAN_AUC),
    }
    path = OUT / "sinogram_models.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

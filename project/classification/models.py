"""Model factories, size classes and phantom-grouped evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIZE_CLASSES = ("small", "medium", "large")
SIZE_SMALL_MAX_CM = 2.0
SIZE_MEDIUM_MAX_CM = 4.0
SEED = 0
N_FOLDS = 5


def select_columns(feature_names: list[str], prefixes: tuple[str, ...]) -> list[int]:
    return [index for index, name in enumerate(feature_names) if name.startswith(prefixes)]


def size_class(diameter_cm: float) -> str:
    if diameter_cm <= SIZE_SMALL_MAX_CM:
        return "small"
    if diameter_cm <= SIZE_MEDIUM_MAX_CM:
        return "medium"
    return "large"


def classifiers() -> dict:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=4000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
    }


def regressors() -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.33, random_state=SEED, n_jobs=-1),
    }


def grouped_folds(y_strat: np.ndarray, groups: np.ndarray, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros(len(y_strat)), y_strat, groups))


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n": int(len(y)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

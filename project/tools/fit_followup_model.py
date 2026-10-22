"""Fit the follow-up models on the one-to-three healthy-scan features and record the nested scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import tools.train_tumor_models as trainer  # noqa: E402

trainer.FEATURE_SETS = {
    "signal": ("sig_",),
    "signal_rot": ("rsig_",),
    "breast": ("sig_w", "rsig_w"),
}

FEATURES = PROJECT / "results" / "classification" / "features_twin_upto3_signal.npz"
NESTED = PROJECT / "results" / "classification" / "followup_upto3_nested.json"
SUMMARY = PROJECT / "results" / "classification" / "final_summary.json"


def choose(task, table, fair) -> tuple[str, np.ndarray]:
    eval_idx = np.flatnonzero(task.rows & fair)
    best_name, best_score, best_pred = None, -np.inf, None
    for cand in trainer.candidates(task):
        pred = trainer.cv_predictions(task, table, fair, cand, eval_idx, trainer.OUTER_FOLDS, trainer.SEED, n_jobs=-1)
        score = task.score(task.target[eval_idx], pred)
        name = "|".join(cand)
        print(f"  {task.name} {name} {score:.4f}", flush=True)
        if score > best_score:
            best_name, best_score, best_pred = name, score, pred
    return best_name, best_pred


def main() -> None:
    table = trainer.load(FEATURES)
    fair = trainer.balanced_fair_mask(table)
    tasks = trainer.build_tasks(table)
    nested = json.loads(NESTED.read_text(encoding="utf-8"))
    results = {}
    for name in ("detection", "size_class", "diameter"):
        print(name, flush=True)
        choice, pred = choose(tasks[name], table, fair)
        block = {"final_choice": choice}
        if name == "detection":
            idx = np.flatnonzero(tasks[name].rows & fair)
            block["final_threshold"] = trainer.best_threshold(tasks[name].target[idx], pred)
            print(f"  threshold {block['final_threshold']:.4f}", flush=True)
        results[name] = block
        nested[name]["final_choice"] = choice
        if "final_threshold" in block:
            nested[name]["final_threshold"] = block["final_threshold"]

    bundle = trainer.fit_final(table, fair, results)
    saved = trainer.save_bundle(bundle, trainer.MODEL_TWIN_PATH)
    print("saved", trainer.MODEL_TWIN_PATH, saved, flush=True)

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    twin = summary["twin_baseline"]
    twin["n_fair"] = nested["n_fair"]
    twin["n_fair_tumor"] = nested["size_class"]["n"]
    twin["n_fair_phantoms"] = nested["n_fair_phantoms"]
    twin["fair_by_generation"] = nested["fair_by_generation"]
    twin["reference"] = "same sitting; both classes use the same count of other healthy scans, from 1 up to 3"
    for name in ("detection", "size_class", "diameter"):
        block = nested[name]
        block["chosen_candidates"] = block.pop("chosen")
        twin[name] = block
    twin["size_class"]["chance_balanced_accuracy"] = nested["chance_balanced_accuracy"]
    twin["diameter"]["average_guess_mae_cm"] = nested["average_guess_mae_cm"]
    summary["final_models_twin"] = saved
    summary["protocol"]["followup_reference"] = twin["reference"]
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("updated", SUMMARY, flush=True)


if __name__ == "__main__":
    main()

"""Honest nested score for the follow-up features that use one to three healthy scans.

Image features are not in this file. The published follow-up models were chosen
from the signal features, so the image candidates are left out here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import tools.train_tumor_models as trainer  # noqa: E402
from classification.models import binary_metrics  # noqa: E402

trainer.FEATURE_SETS = {
    "signal": ("sig_",),
    "signal_rot": ("rsig_",),
    "breast": ("sig_w", "rsig_w"),
}

PATH = PROJECT / "results" / "classification" / "features_twin_upto3_signal.npz"
OUT = PROJECT / "results" / "classification" / "followup_upto3_nested.json"


def run(task, table, fair, repeats: int) -> dict:
    runs, setup_runs, choices = [], [], []
    started = time.time()
    last = None
    for rep in range(repeats):
        idx, pred, chosen, thresholds = trainer.nested(task, table, fair, trainer.SEED + rep)
        setup_pred = trainer.setup_only(task, table, fair, trainer.SEED + rep)
        truth = task.target[idx]
        choices += chosen
        if task.kind == "binary":
            metrics = binary_metrics(truth, pred)
            flag = pred >= thresholds
            metrics["tuned_sensitivity"] = float(np.mean(flag[truth == 1]))
            metrics["tuned_specificity"] = float(np.mean(~flag[truth == 0]))
            for gen, value in trainer.per_generation(table, idx, lambda sel: binary_metrics(truth[sel], pred[sel])["auc"]).items():
                metrics[f"auc_{gen}"] = value
            small = np.isfinite(table["tum_diam_cm"][idx].astype(float)) & (table["tum_diam_cm"][idx].astype(float) <= 2.0)
            healthy = truth == 0
            metrics["auc_small"] = binary_metrics(truth[healthy | ((truth == 1) & small)], pred[healthy | ((truth == 1) & small)])["auc"]
            runs.append(metrics)
            setup_runs.append(binary_metrics(truth, setup_pred))
        elif task.kind == "multiclass":
            from sklearn.metrics import balanced_accuracy_score

            metrics = {"balanced_accuracy": float(balanced_accuracy_score(truth, pred))}
            for gen, value in trainer.per_generation(table, idx, lambda sel: float(np.mean(truth[sel] == pred[sel]))).items():
                metrics[f"accuracy_{gen}"] = value
            runs.append(metrics)
            setup_runs.append({"balanced_accuracy": float(balanced_accuracy_score(truth, setup_pred))})
        else:
            err = np.abs(pred[:, 0] - truth)
            metrics = {"mae_cm": float(err.mean())}
            for gen, value in trainer.per_generation(table, idx, lambda sel: float(err[sel].mean())).items():
                metrics[f"mae_cm_{gen}"] = value
            runs.append(metrics)
            setup_runs.append({"mae_cm": float(np.mean(np.abs(setup_pred[:, 0] - truth)))})
        last = (idx, pred)
        print(f"  {task.name} repeat {rep + 1}/{repeats} {time.time() - started:.0f}s", flush=True)
    result = {
        "n": int(len(last[0])),
        "nested": trainer.summarize(runs),
        "setup_only": trainer.summarize(setup_runs),
        "chosen": {item: choices.count(item) for item in sorted(set(choices))},
    }
    print(task.name, json.dumps(result["nested"]), flush=True)
    return result


def main() -> None:
    table = trainer.load(PATH)
    fair = trainer.balanced_fair_mask(table)
    tasks = trainer.build_tasks(table)
    tumor = fair & np.isfinite(table["tum_diam_cm"].astype(float)) & (table["has_tumor"] == 1)
    diam = table["tum_diam_cm"][tumor].astype(float)
    out = {
        "n_fair": int(fair.sum()),
        "n_fair_phantoms": int(len(set(table["groups"][fair]))),
        "fair_by_generation": {str(g): int(np.sum(fair & (table["generation"] == g))) for g in sorted(set(table["generation"]))},
        "chance_balanced_accuracy": 1.0 / len({trainer.build_tasks(table)["size_class"].target[i] for i in np.flatnonzero(tasks["size_class"].rows & fair)}),
        "average_guess_mae_cm": float(np.mean(np.abs(diam - diam.mean()))),
    }
    for name in ("detection", "size_class", "diameter"):
        out[name] = run(tasks[name], table, fair, repeats=3)
        OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

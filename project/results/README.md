# Results map

Prose reports are under `reports/`, grouped by topic. Numbers, models and the latest automation run stay where the programs write them.

| Folder | What it is |
|---|---|
| `reports/dataset/` | Dataset report |
| `reports/automation/` | Automation report |
| `reports/project/` | How the project works, step by step |
| `reports/techniques/` | Checklist of every implemented method |
| `reports/classification/how_classification_works.pdf` | Classification process, cut-offs and formulas |
| `reports/classification/guide_progress.pdf` | Classification numbers for a project guide |
| `classification/` | Scores (`final_summary.json`), feature caches, saved models |
| `beamformer/` | Beamformer constant sweep and its plots |
| `latest_auto_analysis.md` | Latest automation run. The app and the tests use this path. |

Official classification scores: first scan **71/100**, follow-up **97/100**. A signal-only side check near 69 did not replace the first-scan score. A sinogram network scored 50/100 and a PCA logistic scored 45/100 on the same fair test; neither was adopted.

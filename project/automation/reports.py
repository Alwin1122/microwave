"""Generate guide-facing and beginner-friendly automation reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from automation.types import DatasetVerdict, ValidationBatchResult

_DISCLAIMER = (
    "**Important:** Model / reconstruction output is **not a clinical diagnosis**. "
    "Results are research/educational estimates under stated assumptions and must not "
    "be used for medical decision-making."
)


def write_automation_reports(batch: ValidationBatchResult, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    guide_md = _build_guide_report(batch)
    beginner_md = _build_beginner_report(batch)
    payload = batch.to_dict()
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["disclaimer"] = _DISCLAIMER

    paths = {
        "guide": out / "latest_guide_progress_report.md",
        "beginner": out / "latest_beginner_explainer.md",
        "json": out / "latest_auto_validation.json",
        "guide_archive": out / f"guide_progress_report_{stamp}.md",
        "beginner_archive": out / f"beginner_explainer_{stamp}.md",
        "json_archive": out / f"auto_validation_{stamp}.json",
    }
    paths["guide"].write_text(guide_md, encoding="utf-8")
    paths["beginner"].write_text(beginner_md, encoding="utf-8")
    paths["json"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths["guide_archive"].write_text(guide_md, encoding="utf-8")
    paths["beginner_archive"].write_text(beginner_md, encoding="utf-8")
    paths["json_archive"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def _counts_table(batch: ValidationBatchResult) -> str:
    c = batch.counts or {}
    return (
        "| Status | Count |\n|---|---:|\n"
        f"| pass | {c.get('pass', 0)} |\n"
        f"| warning | {c.get('warning', 0)} |\n"
        f"| fail | {c.get('fail', 0)} |\n"
        f"| blocked | {c.get('blocked', 0)} |\n"
    )


def _sample_status_table(batch: ValidationBatchResult, limit: int = 12) -> str:
    lines = [
        "| File | Scan | Status | Recommended | Key metric / reason |",
        "|---|---:|---|---|---|",
    ]
    for item in batch.results[:limit]:
        name = Path(item.target.measurement_path).name
        scan = "" if item.target.scan_index is None else str(item.target.scan_index)
        reason = (item.reasons[0] if item.reasons else "").replace("|", "/")
        conf = None
        for m in item.metrics:
            if m.name == "confidence" and m.value is not None:
                conf = f"conf={m.value:.2f}"
                break
        evidence = conf or reason
        lines.append(
            f"| `{name}` | {scan} | {item.status.value} | "
            f"{'yes' if item.recommended_upload else 'no'} | {evidence} |"
        )
    if not batch.results:
        lines.append("| _(none)_ |  |  |  |  |")
    return "\n".join(lines)


def _truthfulness_block(batch: ValidationBatchResult) -> str:
    n = len(batch.results)
    n_pass = batch.counts.get("pass", 0)
    n_warn = batch.counts.get("warning", 0)
    reliability = "low"
    if n and (n_pass + n_warn) / n >= 0.8 and n_pass >= n_warn:
        reliability = "moderate-to-high (for research demos)"
    elif n and (n_pass + n_warn) / n >= 0.5:
        reliability = "moderate"
    return f"""### Interpretation and truthfulness analysis

- Projected/reconstructed hotspots suggest where microwave energy focuses under the
  chosen geometry and beamformer; they are **not** pathology labels.
- Confidence scores combine image quality, detection features, beamformer margin,
  localization evidence, and focus sharpness. They are heuristic, not calibrated clinical probabilities.
- Current-run reliability estimate: **{reliability}** based on {n_pass} pass / {n_warn} warning / {n} total.
- Uncertainty factors: antenna geometry assumptions, wave-speed model, clutter, ROI thresholding,
  and limited healthy-vs-tumor calibration of decision gates.
- Assumptions: BMID `_adi` uses pass-through preprocessing; Touchstone uses mild `.s2p` defaults;
  multi-scan cubes use the configured BMID selection strategy (default first tumor else first).
- Limitations: single-scan automation samples may miss hard cases; GT distance is available only
  when metadata provides tumor coordinates.

{_DISCLAIMER}
"""


def _gui_evidence_block() -> str:
    return """### GUI evidence (one-click flow)

1. **Where it starts:** Acquisition tab → **Run Auto Validation** (next to Load dataset).
2. **What the user sees in logs:** progress lines for discover → resolve → load/preprocess/reconstruct → classify → report write.
3. **Where reports are saved/opened:** `project/results/latest_guide_progress_report.md` and
   `project/results/latest_beginner_explainer.md` (also timestamped archives + `latest_auto_validation.json`).
4. **Popup:** end-of-run summary shows pass/warn/fail/blocked counts and recommended uploads.
"""


def _build_guide_report(batch: ValidationBatchResult) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    modules = """
| Module | Status | Purpose | Input | Output | Handling logic |
|---|---|---|---|---|---|
| 1 Acquisition | Completed | Load & validate measurements | `.mat` / `.sNp` / BMID cube | `MicrowaveDataset` | Extension checks, BMID scan pick, metadata extract |
| 2 Preprocessing | Completed | Clean S21 (`.s2p`) or pass-through BMID | Dataset + config | Filtered/time-domain variants | Conditional spike/filter/Hamming/IFFT/ref/clutter |
| 3 Freq→Time | Completed | IFFT path for beamforming | Frequency S21 | Time signals | `dt=1/(NΔf)` axis |
| 4 Reconstruction | Completed | Form image | Time/freq S + geometry | DAS/DMAS/DMAS-D4 images | Shared FOV/grid |
| 5 Quality select | Completed | Pick best beamformer | Candidate images | Selected image + scores | Quality / prefer / force / GT modes |
| 6 ROI localize | Completed | Find suspicious region | Selected image | Mask/bbox/centroid | Compactness + edge penalties |
| 7 ROI refine | Completed (core) | High-res ROI re-image | ROI bbox | Refined image | Local finer grid |
| 8 Detection | Completed (core) | Tumor-candidate decision | ROI features | Yes/No + confidence | Suspicion + clutter penalties |
| 9 Characterization | Completed (core) | Quantitative descriptors | ROI + FOV | cm size/shape/intensity | Physical pixel scales |
| 10 Confidence | Completed (core) | Unified reliability score | Metrics + candidate | 0–1 + label | Weighted components |
| 11 Visualization | Completed (core) | Interactive review | Pipeline state | Plots/tables | PySide6 tabs |
| 12 Reporting | Completed | Export session + automation docs | Context/batch | MD/JSON/PNG/PDF | Autosave + one-click automation |
"""
    innovations = """
### Innovations beyond baseline

- Dual-path preprocessing (BMID pass-through vs UG Touchstone Week 1–3).
- Compact/edge-aware ROI scoring and explicit tumor-candidate classifier.
- Module 9 physical characterization + Module 10 unified confidence.
- **New:** end-to-end auto validation with threshold profiles, cache, MCP tools, and dual reports.
"""
    issues = """
### Key issues faced and fixes

| Issue | Fix |
|---|---|
| Report export failed on arbitrary Windows folders | Write to `results/` first, optional copy |
| BMID multi-scan requires interactive picker | Automation resolves scan via strategy (`first_tumor_else_first`) |
| Duplicate re-checks slow batch runs | SHA cache keyed by path/mtime/size/profile |
| Metadata-only files look like datasets | Classifier + resolver pairs `md_list_*` → `fd_data_*` |
"""
    roadmap = """
### Roadmap / next steps

1. Calibrate gates on larger healthy-vs-tumor BMID subsets.
2. Optional Week-3 time-domain as default reconstruction input.
3. Expand MCP tools for per-module dry-runs.
4. Submission-ready PDF export of automation reports.
"""
    return f"""# Guide-Facing Progress Report (Report A)

- Generated: {now}
- Threshold profile: **{batch.profile}**
- Roots: {', '.join(f'`{r}`' for r in batch.roots) or '(none)'}
- Summary: {batch.message}

{_DISCLAIMER}

## 1. Module-wise progress

{modules}

{innovations}

## 2. Current validation evidence (this run)

### Pass / warn / fail counts

{_counts_table(batch)}

### Sample dataset statuses

{_sample_status_table(batch)}

{_gui_evidence_block()}

{_truthfulness_block(batch)}

{issues}

{roadmap}
"""


def _build_beginner_report(batch: ValidationBatchResult) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# Beginner-Friendly Project Explainer (Report B)

- Generated: {now}
- Auto-validation profile: **{batch.profile}**

{_DISCLAIMER}

## 1. Simple flow (input → output)

1. **Load** a measurement file (`.s2p` or BMID `.mat`).
2. **Preprocess** (clean `.s2p`, or leave BMID `_adi` as-is).
3. **Reconstruct** images with DAS / DMAS / DMAS-D4.
4. **Find ROI** (bright suspicious spot) and refine it.
5. **Decide** tumor-candidate Yes/No and measure size/confidence.
6. **Export** session report and/or run **Auto Validation** for batch checks.

## 2. Each module in plain words

| Module | Input | Process | Output |
|---|---|---|---|
| 1 | File path | Open + check | Clean dataset object |
| 2 | Dataset | Filter / optional IFFT path | Cleaner signals |
| 3 | Frequency data | Convert to time | Time traces |
| 4 | Signals + antenna geometry | Beamform | Breast/phantom image |
| 5 | Three images | Score & pick best | Selected image |
| 6 | Selected image | Find hotspot | ROI box |
| 7 | ROI | Zoom reconstruct | Sharper local image |
| 8 | ROI features | Decide candidate | Yes/No + score |
| 9 | ROI | Measure size/shape | cm descriptors |
| 10 | All evidence | Combine | Overall confidence |
| 11 | Everything | Show plots | GUI views |
| 12 | Results | Write files | MD/JSON/PNG reports |

## 3. Practical run checklist

- [ ] Place data under `project/datasets/` (or point roots elsewhere).
- [ ] For BMID, keep `fd_data_*.mat` beside `md_list_*.mat` when possible.
- [ ] Open GUI → Acquisition → **Run Auto Validation** (or CLI/MCP).
- [ ] Read `results/latest_guide_progress_report.md` and `latest_beginner_explainer.md`.
- [ ] Only upload files marked **recommended**.
- [ ] Remember: not a clinical diagnosis.

## 4. Common mistakes

| Mistake | How to avoid |
|---|---|
| Heavy filtering on BMID `_adi` | Use pass-through preset |
| Loading BMID cube with no scan index | Let automation strategy choose, or pick in GUI |
| Judging only by bright color | Check ROI distance, SCR, confidence, reasons |
| Exporting to odd folders on Windows | Prefer `results/` (automation already does) |
| Treating confidence as diagnosis | Read disclaimer; use as research signal only |

## 5. Glossary

- **S21**: Transmission measurement between antennas.
- **BMID**: University of Manitoba breast microwave imaging dataset family.
- **DAS / DMAS / DMAS-D4**: Beamforming algorithms (delay-and-sum family).
- **ROI**: Region of interest (suspicious area).
- **SCR / SNR**: Contrast / signal quality ratios.
- **Pass / Warning / Fail / Blocked**: Automation quality classes.

## 6. FAQ

**Q: Why is my tumor scan a warning?**  
A: Localization error or confidence may miss a gate. Read remediation tips in the JSON/MD.

**Q: What does recommended upload mean?**  
A: Automation judged the file usable for demo/upload under the selected profile—not medically verified.

**Q: Where do I click?**  
A: Acquisition tab → **Run Auto Validation**. Logs stream on the right; reports open from `results/`.

## 7. Evidence from this run

### Counts

{_counts_table(batch)}

### Sample statuses

{_sample_status_table(batch)}

{_gui_evidence_block()}

{_truthfulness_block(batch)}
"""

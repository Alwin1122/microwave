# BMID Integration & Reconstruction Tuning — Step-by-Step Log

**Repo:** [Alwin1122/microwave](https://github.com/Alwin1122/microwave)  
**Date:** 2026-08-23  
**Dataset:** UM-BMID `fd_data_s21_adi.mat` + `md_list_s21_adi.mat`  
**Primary scan used:** index `0` (A2F1, tumor at 2.25, 2.25 cm, diameter 3 cm)

This document records each development and experiment step so the work can be reproduced and reviewed.

---

## Step 0 — Starting point

- Desktop PySide6 app with Modules 1–3 (load → preprocess → DAS/DMAS/DMAS-D4 + ROI).
- Prior commit on `master`: Module 3 reconstruction + Touchstone S21 preprocessing track.
- Local venv: `.venv_local` (do not use stale `venv` / `.venv` paths from another machine).

Run:

```bash
.\.venv_local\Scripts\python.exe main.py
```

---

## Step 1 — UM-BMID loader

**Goal:** Load one scan from the multi-scan frequency cube instead of treating the whole cube as one measurement.

**Added / changed**
- `data_loader/bmid_loader.py` — scan extract, metadata (antenna radius, tumor GT, phantom id).
- `data_loader/matlab_loader.py` — known BMID sweep **1–8 GHz × 1001**, route `fd_data_s21` / `fd_data_s11`.
- `gui/upload_page.py` — BMID scan picker with All / Tumor / Healthy filter.
- Tests: `tests/test_bmid_loader.py`.

**Resulting dataset shape per scan:** `(1001, 72)` complex S21, antenna radius **18 cm** from metadata.

**Note:** `_adi` data is already adipose-reference subtracted → use mild / no Module 2 suppression.

---

## Step 2 — Tumor GT on reconstruction plots

**Goal:** Compare ROI centroid to known tumor location.

**Changed**
- Tumor `(tum_x, tum_y)` stored as `tumor_x_m` / `tumor_y_m` in metadata.
- Green **X** overlay on selected / comparison plots.
- Session report prints ROI ↔ GT distance.

---

## Step 3 — Session reports (Markdown + JSON + PNG)

**Goal:** Autosave a reviewable report after preprocess / reconstruct.

**Added**
- `utils/session_report.py`
- Autosave in `gui/main_window.py` → `results/latest_session_report.*` (overwritten in place).
- Manual **Download Final Report** button kept.

**Checks included:** frequency validity, SNR, artifact ratio, ROI↔GT thresholds (OK ≤1.5 cm, WARN ≤3 cm).

---

## Step 4 — Manual BMID reconstruction experiments (scan 0)

Fixed FOV **±6 cm**, radius **18 cm**, preprocess varied.

| Run | Preprocess | Wave speed | Beamformer | ROI ↔ GT |
|-----|------------|------------|------------|----------|
| Early | savgol + hybrid SVD | 3e8 | DAS | ~4.1–4.4 cm |
| Mild | none artifact, no BG | 3e8 | DAS | ~4.4 cm |
| **Best** | **all none** | **3e8** | **DMAS-D4** | **1.77 cm** |
| Lower c | all none | 1.8e8 | DAS | 4.61 cm |
| Lower c | all none | 2.1e8 | DAS | 3.10 cm |

**Conclusion from Step 4**
- Do **not** over-process `_adi` data.
- Keep **c = 3.0×10⁸** for this scan (lower c made localization worse).
- Prefer **DMAS-D4** over quality auto-pick (DAS often won on score but was farther from GT).

---

## Step 5 — Antenna geometry controls

**Goal:** Allow BMID angle / flip / arc / phase-delay experiments without code edits.

**Changed**
- `reconstruction/reconstruction_manager.py`
  - Angle offset 0/90/180/270°
  - CW / CCW
  - Flip X / Y
  - Arc 360° or 355° (BMID)
  - Optional UM-BMID phase-delay radius: `0.97*(r−0.106)+0.148`
- Wired into Module 3 GUI dropdowns.

**Experiment outcomes (scan 0)**
- Phase-delay **On** + 355° often **hurt** (e.g. 3.60 cm with Prefer DMAS-D4).
- Flip X mirrored focus → **5.74 cm** (worse).
- Angle 90° → **3.28 cm** (worse than baseline).
- **Best geometry remained:** phase Off, arc 360°, angle 0°, no flip.

---

## Step 6 — ROI and beamformer selection improvements

**Changed**
- `roi/roi_detector.py` — **Prefer off-center** mode to soft-suppress origin ring clutter.
- `quality/beamformer_selector.py` — modes:
  - Quality score
  - Prefer DMAS-D4
  - Closest to tumor GT
  - Force DAS / DMAS / DMAS-D4

**Note:** Peak-score ROI on the locked baseline still gave the **1.77 cm** result; off-center helped some DAS cases but was not always better for DMAS-D4.

---

## Step 7 — Auto Tweak Settings

**Goal:** Automatically search geometry + beamformer; keep manual controls.

**Added**
- `reconstruction/auto_calibrate.py`
  - Stage 0: score **current manual baseline** (full DAS/DMAS/DMAS-D4).
  - Stage 1: fast DAS-only geometry sweep.
  - Stage 2: full beamformers on top-K geometries.
  - Objective with GT: minimize ROI ↔ tumor distance (require ≥0.5 mm improvement to adopt).
  - Without GT: contrast + off-center compact ROI.
- GUI: **Auto Tweak Settings** button + Quick search checkbox + optional wave-speed sweep.
- **Safety rule:** if search does not beat baseline → **do not overwrite** manual settings.

**Tests:** `tests/test_auto_calibrate.py`, `tests/test_beamformer_selector.py`.

---

## Step 8 — Locked baseline (demo settings)

Use these for BMID scan 0 demos and as Auto Tweak starting point:

| Control | Value |
|---------|--------|
| Module 2 | Filter / calib / norm / artifact = **none**; BG subtract **off** |
| Source | Processed (or Raw if preprocess skipped) |
| Grid | 64×64 |
| Antenna radius | 18 cm |
| Wave speed | 3.0e8 m/s (air) — superseded: 2.1e8 + rotational mean, see Step 12 |
| FOV | 12 cm × 12 cm (±6 cm) |
| Angle / rotation / flip | 0° / CCW / None |
| Arc | 360° |
| Phase-delay | Off |
| ROI mode | Peak score |
| Beamformer pick | Force DMAS-D4 |
| **Expected ROI ↔ GT** | **≈ 1.77 cm** |

---

## Step 9 — How to reproduce

1. Place UM-BMID files under `project/datasets/` (large cube is gitignored; download separately):
   - `fd_data_s21_adi.mat`
   - `md_list_s21_adi.mat`
2. Create/activate `.venv_local` and `pip install -r requirements.txt`.
3. `python main.py`
4. Module 1 → load `fd_data_s21_adi.mat` → pick scan 0 (tumor).
5. Module 2 → all **none** → Run Preprocessing.
6. Module 3 → set locked baseline (Step 8) → Run Reconstruction  
   **or** click **Auto Tweak Settings** (Quick on) from a sane starting point.
7. Open `results/latest_session_report.md` for ROI ↔ GT and quality checks.

---

## Step 10 — What was intentionally not committed

- Large BMID frequency cubes (`fd_data_*.mat`, ~220 MB).
- Regenerated session reports under `results/` (`latest_session_report.*` only).
- Virtual environments (`.venv_local`, etc.).

Metadata lists such as `md_list_s21_adi.mat` may be committed if small; the S21 cube must be obtained from UM-BMID / IEEE DataPort.

---

## Step 11 — Suggested next work (not done in this push)

1. Healthy-scan sanity check with the locked baseline.
2. Batch evaluate several tumor scans (is 1.77 cm systematic?).
3. Tighter ROI (peak pixel vs large blob centroid) if &lt;1.5 cm is required.
4. Module 2 Week-3 handoff: Hamming default into Module 3, IFFT/time export, Tx/Rx `.mat` export, NRMSE naming.

---

## File map (this push)

| Path | Role |
|------|------|
| `data_loader/bmid_loader.py` | UM-BMID scan + metadata |
| `reconstruction/auto_calibrate.py` | Auto Tweak search |
| `utils/session_report.py` | Markdown/JSON reports |
| `gui/reconstruction_page.py` | Geometry controls + Auto Tweak UI |
| `gui/upload_page.py` | BMID scan picker |
| `gui/main_window.py` | Report autosave |
| `roi/roi_detector.py` | Off-center ROI option |
| `quality/beamformer_selector.py` | Force / GT pick modes |
| `docs/BMID_DEVELOPMENT_STEPS.md` | This log |
| `tests/test_*.py` | BMID / auto / report / ROI / selector tests |

---

## Scoreboard summary (scan 0)

Superseded by Step 12. Scan-0-only best was:

**preprocess none + c=3e8 + r=18 cm + FOV ±6 cm + DMAS-D4 + peak ROI + no geometry flips/phase-delay → ROI ↔ GT = 1.77 cm**

---

## Step 12 — Module 3 review: delay-sampling fix + cohort-validated BMID defaults

**Bug fixed** (`reconstruction/das.py::gather_delayed_samples`): the IFFT of a 1–8 GHz sweep is baseband, but
samples were used without re-modulation by exp(j2πf₀τ) and with nearest-sample delays. Coherence factor and
DMAS / DMAS-D4 pair products therefore used wrong phases. Now linear interpolation + re-modulation.
Synthetic point target: 4–6 mm peak error → on target; DAS/DMAS spot 30–40 % tighter.

**Scan-0 baseline did not generalise.** Peak ↔ GT on labelled tumour scans (FOV ±6 cm, fixed sampling):

| Settings | Tuning set (24 scans) median | Held-out (76 scans) median | Held-out ≤ 2 cm |
|----------|------------------------------|----------------------------|-----------------|
| Guess image centre | 3.18 cm | 3.18 cm | 0 % |
| none, c = 3e8 (old default) | 8.2–8.9 cm | 4.5–8.8 cm | 0 % |
| rotational mean, c = 2.1e8, DAS | 3.20 cm | **2.13 cm** | **41 %** |
| rotational mean, c = 2.1e8, DMAS | 2.55 cm | 3.82 cm | 33 % |

c = 1.9e8 and 2.3e8 were both worse than 2.1e8. Image peak contrast does **not** separate tumour from healthy
scans (AUC ≈ 0.5 for every setting): the images help localise a known tumour, not detect one.

**New BMID defaults:** c = **2.1e8**, clutter removal **Rotational mean** (new Module 3 control,
`ReconstructionConfig.clutter_removal`, also used by Auto Tweak and auto-validation). Full GUI path
(quality pick + ROI) on all 100 tumour scans:

| Defaults | ROI ↔ GT median | ≤ 1.5 cm | ≤ 3 cm |
|----------|-----------------|----------|--------|
| old (none, 3e8) | 4.45 cm | 0 % | 8 % |
| new (rot-mean, 2.1e8) | **1.96 cm** | 9 % | **80 %** |

> ⚠️ **Correction (Step 14):** this table — and the ROI numbers in Step 13 — were measured with
> GT-assisted ROI selection (`evaluate_rois` picked the candidate closest to the known tumour).
> They are **oracle** numbers, not blind predictions. Blind numbers are in Step 14.

---

## Step 13 — Solving the remaining prediction error (attempts + verdict)

Week 3 is now the Module 3 imaging input (`reconstruct_all_from_time`, GUI **Imaging input**
auto-select). Error diagnostic (100 tumour scans, DAS peak): median 2.41 cm, peak x-position
correlation with GT ≈ 0.00, bias +1.6/+0.9 cm — the focus is clutter-dominated and resolution-limited
(\(c/2B \approx 1.5\) cm at \(c=2.1\times10^8\), B = 7 GHz; median tumour diameter 2 cm).

Candidate fixes, tuned on 24 scans and verified on held-out 76 (DAS, FOV ±6 cm, ROI prediction).
**Note:** the ROI column below used GT-assisted candidate selection — see the Step 14 correction.
The *ranking* of fixes (nothing beat the kept path) was re-checked blind and still holds.

| Fix tried | Tuning ROI median | Held-out ROI median / ≤ 3 cm | Verdict |
|-----------|-------------------|------------------------------|---------|
| **Week 3 clutter-removed + DAS (γ=0.55) + Peak ROI (kept)** | 1.96 cm | **1.77 cm / 100 %** (≤1.5 cm: 22 %) | **kept as prediction path** |
| SVD low-rank clutter (rank 1–3) | 2.22–2.48 cm | not promoted | worse |
| Time gate 0.6 / 0.9 ns | 1.96 cm | identical to baseline | no effect (all pixel delays > 1.1 ns) |
| Constant bias correction | — | 1.77 → 2.22 cm | bias unstable across scans |
| Coherent DAS, DMAS, DAS γ=0 / γ=1, weighted centroid | 2.28–3.52 cm | 2.00–3.19 cm | worse |
| Geometry retune (angle 90°, c=2.0e8) | peak-best on tuning | ROI 2.36 cm / 74 % | tuning overfit |
| Freq + Week 3 ensemble, zero-padding ×3, image² | 2.38–2.96 cm | 1.87–2.66 cm | worse |
| High-res ROI refinement re-centroid | 1.86 cm | 1.78 cm / 100 % | no gain |

**Conclusion:** the kept path is already the best available on this data. Remaining error is
intrinsic: residual angle-dependent clutter, monostatic \(2d/c\) + constant-\(c\) model, and the
~1.5 cm resolution cell. Detection (tumour vs healthy) is **not** solved by imaging
(contrast AUC ≈ 0.5; classification model AUC ≈ 0.66). Further gains need new information, not
new knobs: an empty-reference (`_emp`) frequency cube, confirmed Tx–Rx (bistatic) geometry, or
frequency-dependent phase-centre/speed calibration. GUI now auto-selects this path with Week 3
data and logs the localization-only reliability warning.

---

## Step 14 — Oracle-leakage fix: blind prediction is now the default (and the honest headline)

**Problem found.** The ROI evaluation used the ground truth to choose the answer:
`evaluate_rois(..., tumor_xy_m=GT)` selected the candidate ROI *closest to GT* as the
localization ROI, `classify_tumor_candidate` boosted confidence from GT distance, and
`detect_rois(prior_xy_m=GT)` (passed by the GUI and automation, even with `prior_weight=0.0`)
kept the GT-nearest hotspot via `_ensure_prior_spot`. Every "ROI ↔ GT" number in Steps 12–13
was therefore partly GT-assisted — an oracle upper bound, not a prediction.

**Fix (code).**
- `roi/roi_detector.py::evaluate_rois` is **blind by default**: localization ROI = top-ranked
  candidate, candidates classified without GT, `gt_distance_cm` scores the blind centroid.
  `select_by_gt=True` keeps the old behaviour for lab-only analysis and is labelled as such.
- New `smoothed_peak_centroid_m(image, x_span, y_span, sigma_px=2.0)`: Gaussian-smoothed image
  peak in metres — the best blind point predictor found (blob centroids are dragged by the
  clutter ring; a smoothed peak is not).
- GUI (`gui/reconstruction_page.py`) and automation (`automation/pipeline.py`) now call
  `detect_rois(prior_xy_m=None)` and report the smoothed-peak centroid as the prediction;
  GT is only used afterwards to print/score the error.

**Blind results** (BMID gen1, DAS on Week 3 clutter-removed, 48-grid, FOV ±6 cm;
`_scratch/maximize_blind_final.py`, production code paths):

| Predictor (blind) | Tuning 24 median | Held-out 76 median | Held-out ≤ 3 cm | Held-out ≤ 1.5 cm |
|-------------------|------------------|--------------------|-----------------|-------------------|
| Centre guess | — | 3.18 cm | — | — |
| Top-ranked ROI centroid | 3.77 cm | 3.92 cm | 38 % | 3 % |
| **Smoothed peak (locked)** | 3.05 cm | **1.82 cm** | **70 %** | **14 %** |
| Oracle ROI (Steps 12–13, for reference) | 1.96 cm | 1.77 cm | 100 % | 22 % |

Also tried blind and rejected on held-out: candidate re-ranking by confidence/suspicion,
off-centre/tight ROI modes, weighted centroids, DAS γ = 0.8/1.0, sub-band (3–5 GHz) and
all-band image ensembles, peak consensus across beamformers.

**Honest headline for the report:** on unseen BMID tumour scans the system predicts the tumour
centre with **median error 1.82 cm, 70 % within 3 cm** — close to the \(c/2B \approx 1.5\) cm
resolution cell. Detection (tumour vs healthy) remains unsolved (AUC ≈ 0.5 imaging,
≈ 0.66 pooled classifier, 0.495 gen1-only). The gap between the blind 1.82 cm and the oracle
1.77 cm median is small, but the oracle's 100 % ≤ 3 cm was not achievable blind (70 %). 
166 unit tests pass, including new blind-default and smoothed-peak tests.

---

## Step 15 — Solving the solvable problems, one by one (grouped-split verdicts)

**Protocol upgrade first.** The Step 14 split (first 24 scans vs rest) shares phantoms between
sets. BMID structure: each phantom carries its tumour at exactly **one** of 4 slots,
\((\pm2.25, \pm2.25)\) cm, with phantom counts per slot {0: 14, 1: 3, 2: 2, 3: 1} × 5 scans —
so slot 0 alone is 70 % of scans ("always slot 0" scores 70 % overall; use **balanced**
accuracy). All steps below use phantom-disjoint splits (Step A/B) or leave-one-phantom-out
(LOPO, Step B classifiers). Scripts: `_scratch/stepA…stepF*.py`; images cached in
`_scratch/grouped_images.npz`.

| Step | Idea | Grouped held-out result | Verdict |
|------|------|-------------------------|---------|
| A | Phantom-grouped benchmark, Week 3 vs frequency DAS | Week 3: **2.20 cm / 69 % ≤3 cm**; freq: 2.91 cm / 52 % | Week 3 lock **confirmed** under strict split (file-order 1.82 cm was slightly optimistic) |
| B | Slot classifier (snap / slot-energy / logistic regression, LOPO) | snap balanced 45 % (best); logreg overall 68 % ≈ majority baseline, balanced 33 %, slots 2–3 at 0 % | Classifier **rejected** (minority slots unlearnable: 1–2 phantoms each) |
| B | Confidence gating (logreg probability, LOPO) | top-50 % coverage: slot-acc 68 % → **84 %**, median 1.85 cm | Concept **validated**; shipped train-free version (below) |
| C | Healthy-surrogate reference (same-phantom healthy mean into Week 3) | 2.97 cm / 51 % vs control 2.20 / 69 % | **Rejected** — session mismatch adds clutter |
| D | Phase-delay radius ON | 4.46 cm / 17 % vs OFF 2.20 / 69 % | **Rejected** — default OFF validated blind |
| E | True `_emp` empty-reference cube | Exhaustive automatic search: Drive gen-two `_emp` downloaded (1.16 GB) — signal matching proved our scans are **not** in it (0/100). Zenodo/DataPort `gen-one.zip` downloaded (350 MB) — **S11 only**. Drive gen-one/clean listing verified via embedded view — S11 only. TIB mirror DOI withdrawn; HF mirror = official repack. Conclusion: **no public gen-one S21 `_emp` cube exists** (official gen-one is S11-only; our S21 set is a curated 2021 derivative, md dates 20210730) | **Blocked (file does not exist publicly)**; wrong-gen cube quarantined to `_scratch/gen2_fd_data_s21_emp_WRONG_GEN.mat`; `_scratch/stepE_emp_compare.py` ready if the file is ever obtained (e.g. from the course/institution source of the adi set, or rebuilt from gen-one/raw) |

**Shipped from Step B:** train-free `peak_prominence_confidence`
(`roi/roi_detector.py`, smoothed peak / median; threshold 3.0 calibrated on grouped data:
above → 70 % slot / 1.82 cm, below → 52 % / 3.18 cm). GUI logs prominence, a low-confidence
"indicative only" warning, and (BMID only) the nearest phantom slot
(`reconstruction/bmid_geometry.py::nearest_bmid_slot_m`) as a reporting aid. 168 tests pass.

**Updated honest headline (strictest protocol):** phantom-grouped held-out median **2.20 cm,
69 % ≤ 3 cm, 13 % ≤ 1.5 cm**; slot-balanced accuracy 45 % (snap). Detection unchanged
(unsolved). Remaining lever with real upside: Step E's `_emp` cube — but it must come from
the same source as our curated adi set (the institution/course pack), since no public
release contains gen-one S21 at all. If obtained, place `fd_data_s21_emp.mat` in
`project/datasets/` and run `_scratch/stepE_emp_compare.py`.

---

## Step 16 — Gen-two `_emp` vs `_adi` comparison (the file that *does* exist)

Since no public gen-one S21 `_emp` exists, Step E was run on the **official gen-two Drive
release** (504 tumour pairs matched by scan_id; phantom+GT agree 504/504). Cubes:
`_scratch/gen2_fd_data_s21_adi.mat` (505×1001×72) and
`_scratch/gen2_fd_data_s21_emp_WRONG_GEN.mat` (1008×1001×72, quarantined name kept).
Antenna radius 21–23 cm per scan; FOV ±6.5 cm; Week 3 DAS + smoothed-peak predictor.
Phantom-grouped split: 76 tune / 428 held-out (4 / 62 phantoms). Scripts:
`_scratch/gen2_common.py`, `stepG2_tune.py`, `stepG3_heldout.py`.

**Tuning (76 scans)** — each arm prefers a different wave speed:

| Arm | Best c | Orientation | Tuning median |
|-----|--------|-------------|---------------|
| adi | 2.1e8 | ccw0 | 3.18 cm |
| emp | **2.7e8** | ccw0 | **2.73 cm** |
| emp | 2.1e8 | ccw0 | 5.33 cm (mismatched speed) |

**Held-out (428 phantoms, blind):**

| Predictor | Median | ≤ 3 cm | ≤ 1.5 cm |
|-----------|--------|--------|----------|
| Centre guess | 3.18 cm | 39 % | — |
| adi @ 2.1e8 (adi-tuned) | 3.75 cm | 38 % | 11 % |
| emp @ 2.7e8 (emp-tuned) | **3.55 cm** | 36 % | 7 % |
| emp @ 2.1e8 (wrong speed) | 6.14 cm | 7 % | 1 % |

**Verdict:** empty-chamber reference does **not** clearly beat adipose reference on gen-two
(3.55 vs 3.75 cm — within noise; both ≈ centre guess). Gen-two is harder than our curated
gen-one set (gen-one Week 3 held-out 2.20 cm). No production lock-in: our primary track
remains the curated gen-one adi cube, where `_emp` is unavailable. Step E's hoped-for
clutter win is closed as a negative result on the only generation that has both cubes.

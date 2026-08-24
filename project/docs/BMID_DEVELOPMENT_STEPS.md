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
- Autosave in `gui/main_window.py` → `results/latest_session_report.*` + timestamped copies.
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
| Wave speed | 3.0e8 m/s (air) |
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
- Bulk timestamped session report PNGs/JSON clutter (keep generating locally under `results/`).
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

Best so far remains:

**preprocess none + c=3e8 + r=18 cm + FOV ±6 cm + DMAS-D4 + peak ROI + no geometry flips/phase-delay → ROI ↔ GT = 1.77 cm**

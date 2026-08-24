# Microwave Imaging Framework

**Development of a Data Processing and Image Reconstruction Framework Using Standardized Measurement Data**

Desktop PySide6 application with three modules:

| Module | Role |
|--------|------|
| **1 — Data Acquisition** | Import, validate, and summarize MATLAB (`.mat`) and Touchstone (`.s1p/.s2p/.s4p/.s8p`) measurements as a canonical `MicrowaveDataset`. |
| **2 — Signal Preprocessing** | Clean frequency-domain data. **`.s2p` files** follow the UG S21 brief (`preprocessing/touchstone_s21.py`). **Other formats** use the general multi-trace pipeline (filter → calibrate → normalize → hybrid artifact suppression). |
| **3 — Reconstruction** | IFFT + DAS / DMAS / DMAS-D4 beamforming, automatic beamformer selection, ROI detection and high-resolution refine. |

Brief sources (kept alongside the local project folder): `Module_2_Signal_Preprocessing_Bullets_for_UG_Students.pdf`, `Signal Preprocessing_21082026.pdf`.

### UM-BMID breast scans

Place clean cubes in `datasets/`, e.g. `fd_data_s21_adi.mat` + matching `md_list_s21_adi.mat`.

> Large `fd_data_*.mat` cubes are **gitignored** (~220 MB). Download from UM-BMID / IEEE DataPort and place them locally. See `docs/BMID_DEVELOPMENT_STEPS.md` for the full step-by-step integration and experiment log.

- Loading a BMID `fd_data_*.mat` opens a **scan picker** (tumor / healthy filter).
- Each scan becomes a `(1001 × 72)` `MicrowaveDataset` with **1–8 GHz** frequency axis and **antenna radius 18 cm** from metadata.
- Tumor ground truth (`tum_x`, `tum_y`) is drawn as a green **X** on reconstruction plots.
- Because `_adi` data is already reference-subtracted, use mild filtering in Module 2; avoid treating the whole 200-scan cube as one measurement.
- Module 3 includes **Auto Tweak Settings** (geometry + beamformer search) while keeping all manual controls.
---

## 1. Folder Structure

```
project/
main.py                          # Application entry point
requirements.txt

gui/
    main_window.py               # Tabs: Acquisition + Preprocessing (+ worker)
    upload_page.py               # Module 1 GUI
    reconstruction_page.py       # Module 3 GUI

data_loader/
    loader.py                    # Unified load_dataset() dispatcher
    matlab_loader.py             # .mat (legacy + v7.3/HDF5)
    bmid_loader.py               # UM-BMID fd_data scan extract + metadata
    touchstone_loader.py         # .sNp + S21-only .s2p helpers
    validator.py
    dataset_info.py              # MicrowaveDataset / DatasetSummary
    physical_metadata.py         # Wave speed / radius / FOV from comments

preprocessing/
    touchstone_s21.py            # Module 2 UG S21 track (.s2p)
    filtering.py
    calibration.py
    normalization.py
    artifact_suppression.py      # Mean/BG subtract + SVD + hybrid (array path)
    preprocessing_pipeline.py    # Orchestration (routes .s2p → S21 track)

reconstruction/
    ifft.py · das.py · dmas.py · dmas_d4.py
    reconstruction_manager.py

quality/                         # Beamformer scoring metrics
roi/                             # ROI detect + refine hand-off
utils/                           # Exceptions + logger / StatusLog
datasets/                        # Samples + generate_sample_data.py
results/
tests/
```

---

## 2. Installation

```bash
python -m venv .venv_local
# Windows:
.\.venv_local\Scripts\activate
pip install -r requirements.txt
```

> Bundled `venv` / `.venv` folders may point at another machine’s Python. Prefer a fresh local venv (e.g. `.venv_local`) on this PC.

## 3. Running the Application

```bash
python main.py
```

Tabs:

1. **Data Acquisition** — Load Dataset (`.mat` / `.sNp`). Sample files live in `datasets/`.
2. **Signal Preprocessing** — Configure filters; for `.s2p` optionally select **repeated** and **reference** files, then Run Preprocessing.
3. **Reconstruction** — Choose processed or raw source, grid / radius / wave speed, run DAS·DMAS·DMAS-D4, view ROI refine.
4. **Download Final Report** — optional copy-to-folder. Reports are also **auto-saved** after preprocessing and reconstruction to `results/latest_session_report.md` (+ `.json` / PNGs), with a timestamped archive copy each run.

### Sample datasets

```bash
python datasets/generate_sample_data.py
```

Produces `sample_matlab.mat`, `sample_matlab_v73.mat`, `sample_touchstone.s2p`, `corrupted.mat`.

## 4. Running Tests

```bash
python -m unittest discover -s tests -v
```

**68** unit tests cover loaders, S21 preprocessing, the general pipeline, reconstruction, and ROI.

---

## 5. Module 1 — Data Acquisition

- Entry: `data_loader/loader.py` → always returns `MicrowaveDataset`.
- MATLAB: legacy (`scipy.io.loadmat`) and v7.3/HDF5 (`h5py`), auto-detected.
- Touchstone: full network via `scikit-rf`; S21-only helpers in `touchstone_loader.py` for Module 2.
- Validation: existence/extension/size, finite monotonic frequencies, finite non-zero S-parameters.

---

## 6. Module 2 — Two pipelines

### A. Touchstone `.s2p` S21 track (UG brief)

Triggered automatically when the loaded file is Touchstone **`.s2p`**:

`process_touchstone_s21_dataset()` in `preprocessing/touchstone_s21.py`.

Order in code (matches the bullets PDF):

```
Header + S21 extract → complex linear → validate/sort → uniform grid
→ phase wrap/unwrap → Hampel spike fix → optional complex average
→ optional complex reference subtract → mild filter (R/I separately)
→ Hamming windowed copy (+ optional |S21| normalize) → validation report
```

Processed hand-off to Module 3: filtered complex `S21(f)` on a uniform Hz grid. Hamming and phase arrays are stored under `processed_dataset.metadata["module2_s21"]`.

### B. General / MATLAB multi-trace track

```
Raw → Noise Filtering → Calibration → Normalization →
Background Subtraction → Artifact Suppression (SVD / hybrid) → Processed
```

Used for `.mat` and non-`.s2p` Touchstone loads. Runs on a `QThread` with progress callbacks.

---

## 7. Module 2 brief compliance checklist

Status vs `Module_2_Signal_Preprocessing_Bullets_for_UG_Students.pdf` and Week 1–2 of `Signal Preprocessing_21082026.pdf`.

| Status | Meaning |
|--------|---------|
| Done | Implemented and wired for `.s2p` |
| Partial | Present but incomplete vs brief |
| Missing | Not implemented on the S21 track |

| # | Requirement | Status | Code |
|---|-------------|--------|------|
| 1 | Input valid `.s2p`; extract **S21 only** | Done | `load_touchstone_s21_trace()` |
| 2 | Read header: freq unit, format (DB/MA/RI), Z₀, f start/stop, N | Done | `parse_touchstone_header()` → `TouchstoneHeaderInfo` |
| 3 | Convert all frequencies to **Hz** | Done | `_FREQUENCY_UNIT_SCALE` in `touchstone_loader.py` |
| 4–6 | Convert DB/MA/RI → complex linear (keep mag + phase) | Done | `load_touchstone_s21_trace()` |
| 7 | Average / subtract / filter on **complex** S21 (never dB) | Done | `touchstone_s21.py` averaging, subtraction, filters |
| 8 | Equal length: frequency vector ↔ S21 | Done | `_validate_frequency_and_s21_lengths()` |
| 9 | Reject NaN / Inf / empty / non-numeric / duplicate freqs | Done | `_validate_numeric_content()`, `_sort_and_validate_unique_frequencies()` |
| 10 | Sort ascending; compute Δf | Done | sort + `delta_f_hz` on result |
| 11 | If non-uniform, interpolate **real & imag** onto uniform grid | Done | `_interpolate_complex_trace()` |
| 12 | Plot raw \|S21\| dB, wrapped phase, real, imag **before** preprocess | Partial | Arrays computed (`raw_magnitude_db`, phases, complex S21); GUI shows general Mag/phase comparison, not a dedicated raw Real/Imag/phase panel set |
| 13 | Phase via `atan2(I, R)` | Done | `np.arctan2` in `process_touchstone_s21_dataset()` |
| 14–16 | Unwrap (±180° rule); keep wrapped **and** unwrapped | Done | `_unwrap_phase_degrees()`; fields on `TouchstoneS21ProcessingResult` |
| 17 | Isolated spike fix (Hampel / median / local) | Partial | `_correct_isolated_samples()` / `_hampel_mask()` — Hampel path works; `median`/`local` method names still share the Hampel detector |
| 18 | Complex average of **repeated** same-condition sweeps | Done | `_complex_average()` + GUI “Select Repeated .s2p Files” |
| 19–20 | Complex reference subtraction when freqs match; else skip + report | Done | `_complex_reference_subtraction()` + notes on `TouchstoneS21ValidationReport` |
| 21–22 | Mild filters (MA, Gaussian, median, Savitzky–Golay); same settings on R and I | Done | `filtering.apply_noise_filter()` (also offers Butterworth / none) |
| 23 | Avoid / flag excessive smoothing | Partial | `distortion_ratio` + note if > 0.75; not the brief’s named **NRMSE** metric |
| 24–25 | Hamming-windowed **copy**; keep unwindowed filtered S21 | Done | `windowed_s21 = filtered * np.hamming(N)`; both retained |
| 26 | Optional `S21 / max\|S21\|`; keep original scale | Done | `_normalized_copy()`; `generate_normalized_copy` on config |
| 27 | Final validation (finite, equal lengths, not over-distorted) | Done | re-validate + report notes |
| 28 | Validation report (filter, corrections, averaging, ref subtract) | Done | `TouchstoneS21ValidationReport.to_display_dict()` → GUI summary |
| 29 | Handoff: cleaned complex S21(f) + Hamming S21(f) + uniform f | Partial | Filtered S21 → `processed_dataset`; Hamming / phases in `metadata["module2_s21"]` — Module 3 currently reconstructs from `s_parameters` (filtered), not the Hamming copy by default |

### Work-plan Week 3 items (time-domain Module 2 → Module 3)

From `Signal Preprocessing_21082026.pdf` steps 10–16:

| Requirement | Status | Notes |
|-------------|--------|-------|
| Window target & reference **identically** before subtract | Partial | Code subtracts in frequency **before** Hamming; brief bullets subtract then filter/window. Work plan prefers matched Hamming → IFFT → subtract |
| IFFT + time vector inside Module 2 | Missing on S21 track | `reconstruction/ifft.py` used in Module 3 |
| Time-domain matched reference subtraction | Missing | Freq-domain ref subtract only |
| Group-mean channel clutter removal | Missing on S21 track | General SVD/hybrid exists for multi-trace MATLAB path |
| One **global** normalize across channels | Partial | Single-trace `max\|S21\|`; multi-channel global α not implemented |
| Export `.mat` / CSV with Tx/Rx coordinates + full parameter report | Missing | Results stay in memory / GUI; no geometry export yet |

Cited method support in the work plan: Blanco-Angulo et al. (Biosensors 2022); Hammouch et al. (Multimedia Tools Appl. 2025).

---

## 8. Module 3 — Reconstruction (current)

- Frequency → time: `reconstruction/ifft.py`
- Beamformers: DAS, DMAS, DMAS-D4 (`reconstruction/`)
- Manager: circular array with **angle offset / CW / axis flip / 355° BMID arc**, optional **phase-delay radius** (UM-BMID)
- Quality pick: `quality/beamformer_selector.py` — modes: quality score, prefer DMAS-D4, closest to tumor GT, force DAS/DMAS/DMAS-D4
- ROI: `roi/roi_detector.py` — peak score or prefer off-center (suppresses origin ring clutter)
- Auto Tweak: `reconstruction/auto_calibrate.py` — DAS coarse geometry sweep + full beamformer refine; GUI button applies best settings to manual controls
- GUI: `gui/reconstruction_page.py`

---

## 9. Error Handling

All framework errors derive from `MicrowaveFrameworkError` (`utils/exceptions.py`):

| Exception | Raised when |
|---|---|
| `UnsupportedFileFormatError` | Extension not `.mat/.s1p/.s2p/.s4p/.s8p` (or non-`.s2p` for S21 extract) |
| `CorruptedFileError` | File cannot be parsed |
| `MissingVariableError` | No frequency / S-parameter array in `.mat` |
| `EmptyDatasetError` | Zero-length usable data |
| `InvalidFrequencyError` | NaN/Inf/negative/non-uniform after process |
| `InvalidSParameterError` | Shape / NaN/Inf / length mismatch |
| `DatasetValidationError` | Aggregated validation failure |

Errors show in the Status Log and a `QMessageBox`; the app should not crash on invalid input.

---

## 10. Known gaps / next work

1. Dedicated pre-process plots for raw Real / Imag / wrapped / unwrapped phase (data already available).
2. Use Hamming-windowed S21 as the default Module 3 input when present.
3. Implement Week 3 S21 path: IFFT, time-domain ref subtract, optional channel clutter, `.mat` export with Tx/Rx coords.
4. Report **NRMSE** by name; distinguish Hampel vs median vs local spike detectors.
5. Align subtract/window order with the work-plan PDF if that brief is authoritative for grading.
6. Sweep antenna angle offset / flips on BMID tumor scans until ROI↔GT is consistently &lt;1.5 cm.

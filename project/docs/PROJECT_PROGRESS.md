# Project progress — what we built

This document summarizes the Microwave Imaging Framework work completed across Modules 1–3, focused on **data acquisition**, **preprocessing**, **BMID support**, and **UI/docs cleanup**.

Repository: https://github.com/Alwin1122/microwave  
Primary working branch for this work: `breawave` (also merged toward `master` when pushed).

---

## Goals

1. Load and validate measurement files (`.mat`, Touchstone `.sNp`, UM-BMID cubes).
2. Meet the UG Module 2 brief for **Touchstone `.s2p` / S21** (Weeks 1–3).
3. Support **UM-BMID** already-cleaned `_adi` scans with pass-through preprocessing and reconstruction demos.
4. Keep a usable desktop GUI and clear documentation.

---

## Two data paths (important)

| Path | Input | Module 2 | Why |
|------|--------|----------|-----|
| **UM-BMID** | `fd_data_s21_adi.mat` + `md_list_s21_adi.mat` (~200 scans) | **Pass-through (all none)** | `_adi` is already reference-cleaned. Metadata maps **tumor vs healthy**. |
| **UG `.s2p` brief** | Touchstone `.s2p` | Full Week 1–3 S21 pipeline | Official Module 2 PDFs describe this track only. |

Large `fd_data_*.mat` cubes are **gitignored**. Place them locally under `project/datasets/`.

---

## Module 1 — Data Acquisition (done)

**Code:** `data_loader/`, `gui/upload_page.py`

- Unified `load_dataset()` → `MicrowaveDataset`
- MATLAB legacy + HDF5 v7.3
- Touchstone via `scikit-rf` (+ S21-only helpers for Module 2)
- Validation (finite freqs/S-params, formats, errors)
- **BMID scan picker**: tumor / healthy filter from `md_list_*.mat`
- Physical metadata (antenna radius, tumor GT when present)

---

## Module 2 — Signal Preprocessing (done)

### A. Touchstone `.s2p` S21 track (UG brief)

**Code:** `preprocessing/touchstone_s21.py`, `week3_time_domain.py`, `handover_export.py`

Week 1–2:

- Extract **S21 only**, header parse, Hz + complex linear (DB/MA/RI)
- Validate / sort / uniform grid (interp R & I)
- Raw plots: |S21| dB, wrapped phase, real, imag
- Phase unwrap (±180°), spike fix (**Hampel / median / local**)
- Optional repeated-sweep average + reference subtract
- Mild filters on R and I; **NRMSE** reporting
- Hamming copy kept separately from filtered S21

Week 3:

- Matched Hamming on target + reference
- IFFT + time vector (`Δt = 1/(N Δf)`)
- Time-domain reference subtract
- Group-mean clutter (≥2 channels; skipped on single `.s2p` with note)
- Global normalize α
- **Export Module 3 Handoff** (`.mat` / `.csv` / `.json` + Tx/Rx coords)

### B. General / MATLAB multi-trace

**Code:** `preprocessing_pipeline.py` + filter / calibrate / normalize / artifacts

Used for `.mat` and non-`.s2p` loads. For BMID demos: GUI **BMID pass-through** preset.

### GUI presets

- **BMID pass-through** — all cleaning off (auto on BMID load)
- **`.s2p` defaults** — mild Savitzky–Golay + Week 3 (auto on `.s2p` load)

---

## Module 3 — Reconstruction (present)

**Code:** `reconstruction/`, `quality/`, `roi/`, `gui/reconstruction_page.py`

- IFFT + DAS / DMAS / DMAS-D4
- Beamformer selection modes (quality, prefer DMAS-D4, GT distance, force)
- ROI modes (peak / off-center / tight peak)
- Auto Tweak (geometry + beamformer search)
- BMID-friendly geometry (18 cm radius, 355° arc, phase-delay option)
- Session reports under `results/`

**Known gap:** GUI still reconstructs primarily from **filtered `S21(f)`**; Hamming / Week 3 time-domain are stored in metadata and the handoff file, not yet the default Module 3 input.

---

## UI cleanup (done)

**Code:** `gui/styles.py`, refreshed Module 1–3 panels

- Light Fusion theme (fixes dark Windows theme making Settings unreadable)
- Plot/image-first tabs: **Plots / Image** separate from **Settings / Results**
- Compact headers with primary **Run** actions
- Shared stylesheet for consistent contrast and spacing

---

## Documentation & diagrams

| Doc | Purpose |
|-----|---------|
| `README.md` | Quick start, two paths, flowcharts |
| `docs/PROJECT_PROGRESS.md` | This progress log |
| `docs/BMID_DEVELOPMENT_STEPS.md` | BMID experiment / integration log |
| `docs/canvases/module1-module2-flowchart.canvas.tsx` | Interactive Module 1–2 flowchart |

UG briefs (local, not necessarily in git):  
`Module_2_Signal_Preprocessing_Bullets_for_UG_Students.pdf`, `Signal Preprocessing_21082026.pdf`

---

## Tests

```bash
cd project
python -m unittest discover -s tests -v
```

Includes loaders, preprocessing, Week 3 / handoff (`tests/test_week3.py`), reconstruction, ROI.

---

## How to run

```bash
cd project
python -m venv .venv_local
.\.venv_local\Scripts\activate
pip install -r requirements.txt
python main.py
```

### BMID demo

1. Put `fd_data_s21_adi.mat` + `md_list_s21_adi.mat` in `datasets/`
2. Acquisition → pick tumor or healthy scan
3. Preprocessing → BMID pass-through → Run
4. Reconstruction → 18 cm, FOV 12×12, `c=3e8`, Peak ROI, Prefer DMAS-D4

### `.s2p` demo

1. Load `sample_touchstone.s2p`
2. Preprocessing → `.s2p` defaults → Run → Export handoff
3. Reconstruction from processed data

---

## Recommended next steps

1. Wire Hamming / Week 3 time-domain as default Module 3 reconstruction input when present.
2. Keep BMID localization experiments documented in `BMID_DEVELOPMENT_STEPS.md`.
3. Ensure teammates clone **`breawave`** or updated **`master`** (not an old default tip without docs).

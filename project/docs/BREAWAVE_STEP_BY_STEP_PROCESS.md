# BreaWave — Step-by-Step Process: File → Image → Report/PDF

Purpose: learn exactly what happens to a scan, stage by stage, so you can find
**where the 1.77 cm error enters** and try to minimize it yourself.

```
 input .mat / .s2p
   │  (1) Load + validate          Module 1
   ▼
 MicrowaveDataset  (freq × channels, complex)
   │  (2) Preprocess               Module 2  (steps 1–16 for .s2p; pass-through for BMID)
   ▼
 cleaned S21(f)  +  Week 3 time signal s(t)
   │  (3) Beamform every pixel     Module 3  DAS / DMAS / DMAS-D4
   ▼
 image (grid of numbers)
   │  (4) Pick beamformer + ROI    quality score, detect_rois
   ▼
 predicted tumour centre (x, y)
   │  (5) Score vs ground truth    only when labels exist
   ▼
 ROI ↔ GT distance (cm)
   │  (6) Report                   session report .md + .json + PNG figures
   ▼
 results/latest_session_report.*  →  (optional) PDF tools in tools/
```

---

## Step 1 — Load the file (Module 1)

**Code:** `data_loader/loader.py`, `bmid_loader.py`, `touchstone_loader.py`  
**GUI:** Module 1 panel, `gui/upload_page.py`

| Item | BMID path | `.s2p` path |
|---|---|---|
| Input | `fd_data_s21_adi.mat` (~230 MB) + `md_list_s21_adi.mat` | `sample_touchstone.s2p` (text) |
| What is read | a cube of many scans; you pick **one scan** | header line + frequency rows |
| Output object | `MicrowaveDataset` | `MicrowaveDataset` |
| Scan shape | **(1001 frequencies, 72 antennas)**, complex | (201, 1) S21 only |
| Labels kept | `tumor_x_m`, `tumor_y_m`, `has_tumor`, `ant_rad` (18 cm) | header: unit, format DB/MA/RI, Z₀ |

**Checks:** frequencies finite/increasing, S-parameters finite, correct shapes.  
**Error that can enter here:** wrong scan picked, wrong radius/labels in metadata.

## Step 2 — Preprocess (Module 2)

**Code:** `preprocessing/preprocessing_pipeline.py`, `touchstone_s21.py`, `week3_time_domain.py`  
**GUI:** Module 2 panel; **Module 2 Steps** tab plots steps 1–16.

- **BMID:** preset **pass-through** (all cleaning `none`). `_adi` data is already
  reference-cleaned; cleaning it again made results worse.
- **`.s2p` (UG brief 1–16):** parse header → S21 complex linear → sort/validate →
  uniform Δf → dB/phase plots → Hampel spike fix → average repeats → mild filter
  (NRMSE measured) → Hamming copy → Week 3 (below) → validate + Module 3 handoff.
- Output `processed_dataset.s_parameters` = filtered S21(f), plus metadata
  `module2_week3` with the time signals (Step 3).

**Error that can enter here:** over-filtering smears echoes; spike fix can move
samples (counts are reported, and NRMSE shows how much the filter changed the signal).

## Step 3 — Week 3: to the time domain

**Code:** `preprocessing/week3_time_domain.py::process_week3_time_domain`

1. Hamming window: \(w_H[k]=0.54-0.46\cos(2\pi k/(N-1))\), \(S_H = S_f \cdot w_H\)
2. IFFT: \(s[n]=\mathrm{IFFT}(S_H)\), with \(\Delta t = 1/(N\Delta f)\), \(T=1/\Delta f\)
3. Optional reference subtraction (if a reference scan was given)
4. Group-mean clutter removal: \(s_{clean}=s-\mu_G\) per group of comparable channels
5. Global normalization: \(\tilde s = s/\alpha\), one \(\alpha=\max|s|\) for all channels

Output: `time_clutter_removed` etc., same shape (1001, 72). This is the signal
Module 3 now images by default (**Imaging input** control in Module 3).

## Step 4 — Imaging setup (Module 3)

**Code:** `reconstruction/reconstruction_manager.py`, `gui/reconstruction_page.py`

- Antennas placed on a circle: radius from metadata (BMID 18 cm), 360° arc.
- Image area (FOV): 12×12 cm (±6 cm). Grid 64×64 in the GUI (48×48 in batch tests).
- Wave speed: BMID default \(c = 2.1\times10^8\) m/s (validated; air \(3\times10^8\) failed).
- Clutter setting: rotational mean (BMID) / none for already-cleaned Week 3 signals.

## Step 5 — Beamforming: how a pixel is computed (DAS)

**Code:** `reconstruction/das.py::gather_delayed_samples`, `das_reconstruct`

For **each pixel** \(p=(x,y)\) and **each antenna** \(a\):

1. distance \(d_a = |\,\text{antenna}_a - p\,|\)
2. two-way delay \(\tau_a = 2 d_a / c\)  ← *monostatic assumption*
3. sample the time signal at \(\tau_a\) (linear interpolation, phase restored by
   \(e^{j2\pi f_0\tau_a}\) — the IFFT signal is baseband)
4. DAS pixel value: \(\text{image}(p)=\sum_a |s_a(\tau_a)| \times \text{coherence}^{\gamma}\), \(\gamma=0.55\)

Repeat for all pixels → one DAS image. DMAS multiplies antenna pairs before summing
(sharper, more clutter-sensitive); DMAS-D4 applies a 4th-root contrast map after DMAS.

**Worked scale:** pixel at the centre, antenna at 18 cm ⇒ \(\tau = 2(0.18)/2.1\times10^8
\approx 1.7\) ns. Pixel spacing at 48 grid = 2.5 mm, but the *resolution cell* is
\(c/(2B)\approx1.5\) cm (B = 7 GHz) — the spot is a ~3 cm blob, not a point.

**Error that enters here (the big one):** wrong \(c\), wrong delay model
(bistatic would be \((d_{Tx}+d_{Rx})/c\)), antenna phase-centre error, leftover
clutter adding at the same delays as the tumour.

## Step 6 — Choose image + find the region (ROI)

**Code:** `quality/beamformer_selector.py`, `roi/roi_detector.py`

1. Run DAS, DMAS, DMAS-D4; **Quality score** pick (on BMID it picks DAS).
2. `detect_rois`: threshold the image, find blobs, score compactness/peak;
   **Peak score** mode returns candidates; the localization ROI's **centroid**
   \((x,y)\) is the prediction.
3. `reconstruct_high_resolution_roi`: re-beamform only that box at 96×96 for display
   (re-centroiding it did not improve held-out error).

**Error that enters here:** centroid of a wide, elongated blob is pulled by its
tail — scan 0 predicted (2.33, −0.45) cm vs true (2.25, 2.25) cm ⇒ 2.7 cm.

## Step 7 — Score against ground truth (evaluation only)

**Code:** `roi/roi_detector.py::evaluate_rois`

\[
\text{error} = \sqrt{(x_{pred}-x_{GT})^2+(y_{pred}-y_{GT})^2}\quad\text{(cm)}
\]

Validated results (DAS + Week 3, Peak ROI): tuning 24 scans median 1.96 cm;
**held-out 76 scans median 1.77 cm, 100 % within 3 cm, 22 % within 1.5 cm.**
Detection from the image is *not* claimed: tumour-vs-healthy contrast AUC ≈ 0.5.

## Step 8 — How the image becomes a report / PDF

**Code:** `utils/session_report.py`, GUI **Report** button, auto-save in `gui/main_window.py`

1. The reconstruction run is packed into a `ReconstructionSnapshot`
   (selected image, all beamformer images, ROI, config, GT distance, confidence).
2. Module 1/2 results + snapshot → `SessionReportContext`.
3. `write_session_report(ctx, results/, "latest_session_report")` writes:
   - `latest_session_report.md` — text report (checks, settings, ROI↔GT, notes)
   - `latest_session_report.json` — same numbers, machine-readable
   - PNG figures (auto-generated validation plots): magnitude, phase,
     filter comparison, time-domain, same-scale beamformer comparison
4. This auto-updates after every preprocessing/reconstruction run.
5. **PDF:** the app itself writes Markdown/JSON/PNG, not PDF. PDF files in
   `results/` are produced separately by the documentation tools
   (`tools/generate_three_beginner_pdfs.py` — ReportLab, A/B/C reports) or by
   exporting/printing the Markdown. The Module 3 *handoff* export is
   `.mat` + `.csv` + `.json` (Step 16), also not a PDF.

## Where to try minimizing the error (your experiments)

| Try changing | Where | Expectation (from held-out tests) |
|---|---|---|
| Clutter method | `reconstruction_manager.apply_clutter_removal` / `bmid_geometry.remove_*` | SVD/gating already worse or no-op; empty-reference needs `_emp` cube |
| Wave speed / angle / radius | `ReconstructionConfig` (GUI combos) | cohort retunes overfit tuning scans; 2.1e8 / 0° is held-out best |
| Delay model | `das._compute_travel_times` → \((d_{Tx}+d_{Rx})/c\) | only if Tx–Rx pairing is confirmed |
| Estimator | `roi_detector.detect_rois` (centroid vs peak) | peak is worse (2.41 cm); centroid is current best |
| Beamformer constants | `das.py` / `dmas.py` DEFAULT_* | γ=0.55 cohort value; γ=0/1 and DMAS lost held-out |
| New information | `_emp` reference cube, wider bandwidth | the only route below the ~1.5 cm resolution floor |

Rule for your trials: **tune on 24 scans, report on the other 76.** Every fix
that looked good on the tuning set and failed held-out is recorded in
`docs/BMID_DEVELOPMENT_STEPS.md`, Step 13.

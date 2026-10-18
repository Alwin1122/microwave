# BreaWave Report — Chapter Outline (What Each Chapter Should Contain)

Use this as a writing checklist. Target structure: **7 chapters + References** (or **6** if you merge Discussion into Results).

**Working names (recommended):**
1. Introduction  
2. Background and Related Work  
3. System Design and Methods  
4. Experimental Design  
5. Results and Analysis  
6. Discussion  
7. Conclusion and Future Work  
— References  

---

## Chapter 1 — Introduction

**Goal:** Motivate the problem and state what BreaWave contributes.

### Should contain
- Breast cancer screening need and limits of mammography / MRI (brief)
- Microwave imaging idea: dielectric contrast, non-ionizing, lower cost
- Gap: open datasets exist (UM-BMID, Touchstone), but few complete paths from raw data → image → reviewable report
- Project aim: end-to-end desktop framework (**BreaWave**)
- Objectives (bullet list)
- Scope: Module 1 acquisition, Module 2 preprocessing, Module 3 reconstruction + ROI
- Two data paths (table):
  - UM-BMID `.mat` (already-cleaned `_adi`)
  - UG Touchstone `.s2p` S21 brief
- Report organization (one paragraph pointing to later chapters)

### Module mapping
| Module | Mention in Ch 1 |
|--------|-----------------|
| Module 1 | Load / validate measurements |
| Module 2 | Clean S21 / pass-through for BMID |
| Module 3 | Image + localize tumour region |

### Do not put here
- Full equations, full code, full result tables (save for Ch 3–5)

---

## Chapter 2 — Background and Related Work

**Goal:** Show prior methods and where BreaWave sits (not “old = bad”).

### Should contain
- Microwave breast imaging literature (DAS, DMAS, confocal / radar-style)
- Clutter / skin / antenna-ring removal ideas
- UM-BMID dataset and typical use
- Touchstone / VNA S-parameter measurement basics
- Related software / toolboxes (high level)
- Comparison table: prior approach vs BreaWave (end-to-end GUI, Module 2 brief, cohort-validated BMID defaults)

### Module mapping
| Module | Background topics |
|--------|-------------------|
| Module 1 | File formats, VNA / BMID cubes, metadata |
| Module 2 | Spike detection, averaging, filtering, Hamming, IFFT, clutter |
| Module 3 | Delay-and-sum family, coherence factor, ROI |

### Do not put here
- Your full pipeline steps with all parameters (that is Ch 3)

---

## Chapter 3 — System Design and Methods

**Goal:** Describe *what BreaWave does* (architecture + methods per module).

### 3.1 System overview
- Block diagram: Module 1 → Module 2 → Module 3 → session report / handoff
- GUI overview (Acquisition, Preprocessing, Reconstruction)

### 3.2 Module 1 — Data Acquisition
**Should contain**
- Supported inputs: MATLAB `.mat`, Touchstone `.s2p`, UM-BMID cubes
- Unified container: `MicrowaveDataset`
- Validation (finite freqs/S-params, formats)
- BMID scan picker (tumour / healthy from metadata)
- Physical metadata (antenna radius, tumour GT when present)

**Key outputs to Module 2**
- Frequencies, complex S-parameters, metadata

### 3.3 Module 2 — Signal Preprocessing
**Should contain (map to UG brief steps 1–16)**

| Step group | Content |
|------------|---------|
| 1–5 | Header, S21 extract, complex linear, sort/validate, uniform Δf, mag/phase |
| 6–7 | Hampel / isolated sample fix; complex averaging of repeats |
| 8–9 | Mild filters on Re/Im; NRMSE vs input |
| 10–11 | Hamming window; matched window on target + reference |
| 12–15 | IFFT time domain; reference subtract; group clutter; global α |
| 16 | Validate + export Module 3 handoff (`.mat` / `.csv` / `.json`) |

Also:
- BMID **pass-through** preset (all cleaning off) — why `_adi` must not be over-processed
- `.s2p` brief defaults (mild Savitzky–Golay + Week 3)
- GUI **Module 2 Steps** tab (one plot per brief step)

**Key outputs to Module 3**
- Filtered S21(f) (current GUI default input)
- Week 3 time signals + handoff file (available; note gap if not yet default imaging input)

### 3.4 Module 3 — Reconstruction and Localization
**Should contain**
- IFFT frequency → time
- Beamformers: DAS, DMAS, DMAS-D4
- Delay sampling: linear interpolation + carrier remodulation \(e^{j 2\pi f_0 \tau}\) (why baseband IFFT needs this)
- Geometry: circular array, radius, arc, wave speed, optional BMID phase-delay radius
- Clutter removal option: none / rotational mean
- ROI modes: peak / off-center / tight peak
- Beamformer pick modes + Auto Tweak
- Session reports under `results/`

### 3.5 Difference from prior tools
- End-to-end path, two datasets, validated BMID defaults, exportable handoff, GUI step plots

### Equations to include (minimum)
- Complex average: \(\bar{S}_{21}[k] = \frac{1}{M}\sum_m S_{21}^{(m)}[k]\)
- NRMSE: \(\|S_f - S\| / \|S\|\)
- Hamming: \(w_H[k] = 0.54 - 0.46\cos(2\pi k/(N-1))\)
- Time step: \(\Delta t = 1/(N\Delta f)\)
- Two-way delay: \(\tau = 2d/c\) (state monostatic assumption)

---

## Chapter 4 — Experimental Design

**Goal:** Research protocol — how claims will be tested.

### Should contain
- Datasets and splits
  - BMID tumour scans (e.g. tune 24 / held-out 76) + healthy scans
  - Sample / lab `.s2p` for Module 2 checks
- Metrics
  - Localization: ROI↔GT distance (cm), % ≤ 1.5 cm, % ≤ 3 cm
  - Detection: tumour vs healthy AUC from image contrast
  - Preprocessing: NRMSE, corrected spike counts, validation checks passed
- Baselines
  - Scan-0 style defaults: no clutter, \(c = 3\times 10^8\)
  - Cohort defaults: rotational-mean clutter, \(c = 2.1\times 10^8\)
- Ablations planned
  - Remodulation on/off
  - Clutter on/off
  - Wave speed sweep
  - DAS vs DMAS vs DMAS-D4
  - (If available) filtered S21 vs Week 3 Hamming / clutter-removed input
- Hypotheses (write 3–5 clear H1…H5)
- Implementation note: Python, PySide6 GUI, unittest suite

### Module mapping
| Module | Experiments in Ch 4 |
|--------|---------------------|
| Module 1 | Load success, metadata GT availability |
| Module 2 | Spike/filter/NRMSE / handoff validation on `.s2p` |
| Module 3 | Localization + detection metrics on BMID |

---

## Chapter 5 — Results and Analysis

**Goal:** Show evidence for each hypothesis (tables + figures).

### 5.1 Module 1 results
- Example loaded scan shapes (e.g. BMID `(1001, 72)`)
- Metadata fields used (radius, tumour XY)

### 5.2 Module 2 results (`.s2p` track)
- Header parse + complex linear conversion examples
- Spike correction before/after plot
- Filter comparison + NRMSE
- Hamming / time-domain plots
- Handoff validation: checks passed / blocked on bad data
- GUI Module 2 Steps screenshots (optional appendix)

### 5.3 Module 3 results (BMID track)
**Must include (already measured)**
- Delay-sampling fix: synthetic point-target error before/after
- Cohort vs scan-tuned defaults on 100 tumour scans:

| Defaults | Median ROI↔GT | ≤ 1.5 cm | ≤ 3 cm |
|----------|---------------|----------|--------|
| Old (none, \(c=3\times10^8\)) | 4.45 cm | 0% | 8% |
| New (rot-mean, \(c=2.1\times10^8\)) | 1.96 cm | 9% | 80% |

- Held-out confirmation (tune vs test medians)
- Detection finding: image peak contrast AUC ≈ 0.5 (localize ≠ detect)
- Example reconstruction images with GT overlay

### 5.4 Cross-module result
- Does Module 2 change Module 3 error? (run when Week 3 input is wired; until then state as open experiment)

---

## Chapter 6 — Discussion

**Goal:** Interpret results honestly (research tone).

### Should contain
- Why cohort defaults beat scan-0 defaults (overfitting to one scan)
- Why rotational mean + lower effective \(c\) helps on this cube
- Localization vs detection: what the system can / cannot claim clinically
- Monostatic \(2d/c\) assumption for S21 cubes — limitation
- Week 3 outputs exist but GUI may still image filtered S21(f) — implication
- Gen1-only validation — transfer to gen2/gen3 unknown
- Comparison with related work (return to Ch 2 table)

### Module mapping
| Module | Discussion points |
|--------|-------------------|
| Module 2 | Over-cleaning `_adi` hurts; mild / none for BMID |
| Module 3 | Remodulation matters for coherence / DMAS; defaults matter as much as algorithm name |

---

## Chapter 7 — Conclusion and Future Work

### Conclusion (should contain)
- Restate aim and what was built (Modules 1–3)
- Main findings (3–5 bullets): brief compliance, delay fix, cohort defaults, localize≠detect
- Contribution statement

### Future work (should contain)
1. Use Week 3 Hamming / clutter-removed signals as default Module 3 input  
2. Tighter ROI to raise % ≤ 1.5 cm  
3. Classification module for detection after physics imaging  
4. Cross-generation BMID (gen2 / gen3) retuning  
5. Bistatic Tx–Rx delay model if S21 geometry is confirmed  

---

## References

- IEEE / journal papers on DAS, DMAS, microwave breast imaging  
- UM-BMID dataset citation  
- Any UG Module 2 brief if allowed as technical reference  
- Software libraries only if required by college format  

---

## Quick “per module” writing map

| Module | Primary chapters | Secondary |
|--------|------------------|-----------|
| **Module 1 Data Acquisition** | Ch 3.2, Ch 5.1 | Ch 1, Ch 2 |
| **Module 2 Preprocessing** | Ch 3.3, Ch 5.2 | Ch 4, Ch 6 |
| **Module 3 Reconstruction** | Ch 3.4, Ch 5.3 | Ch 4, Ch 6, Ch 7 |

---

## Suggested figure / table list

| ID | Content | Chapter |
|----|---------|---------|
| Fig 1 | System block diagram | 1 or 3 |
| Fig 2 | Module 2 pipeline (steps 1–16) | 3 |
| Fig 3 | Spike / filter / Hamming example | 5 |
| Fig 4 | DAS image + GT overlay | 5 |
| Table 1 | Related work comparison | 2 |
| Table 2 | Metrics and splits | 4 |
| Table 3 | BMID old vs new defaults | 5 |
| Table 4 | Ablation summary | 5 |

---

## Length guide (approximate)

| Chapter | Pages (UG report) |
|---------|-------------------|
| 1 Introduction | 2–3 |
| 2 Background | 3–5 |
| 3 System Design | 6–10 |
| 4 Experimental Design | 2–4 |
| 5 Results | 5–8 |
| 6 Discussion | 2–3 |
| 7 Conclusion | 1–2 |
| References | as needed |

**Total target:** ~20–35 pages excluding large appendices.

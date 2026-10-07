"""Checklist PDF of implemented techniques, by module, with formulas and file locations."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent.parent / "results"
NAVY = colors.HexColor("#1b365d")
TEAL = colors.HexColor("#1f6f8b")
CREAM = colors.HexColor("#f4f7fb")
LINE = colors.HexColor("#c5d0de")
PAGE = landscape(A4)


def styles():
    base = getSampleStyleSheet()
    return {
        "cover": ParagraphStyle("c", parent=base["Title"], fontName="Times-Bold", fontSize=20, leading=24, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontName="Times-Italic", fontSize=11, leading=14, textColor=TEAL, alignment=TA_CENTER, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Times-Bold", fontSize=13, leading=16, textColor=NAVY, spaceBefore=8, spaceAfter=6),
        "body": ParagraphStyle("b", parent=base["Normal"], fontName="Times-Roman", fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=6),
        "note": ParagraphStyle("n", parent=base["Normal"], fontName="Times-Italic", fontSize=8.5, leading=11, textColor=colors.HexColor("#444"), spaceAfter=6),
        "th": ParagraphStyle("th", parent=base["Normal"], fontName="Times-Bold", fontSize=7.4, leading=9.4, textColor=colors.white, alignment=TA_CENTER),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontName="Times-Roman", fontSize=7.2, leading=9.4, alignment=TA_LEFT),
        "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontName="Times-Bold", fontSize=7.2, leading=9.4, alignment=TA_LEFT, textColor=NAVY),
    }


def header_footer(canvas, doc, title):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE[1] - 16, PAGE[0], 16, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(1.2 * cm, PAGE[1] - 11, title)
    canvas.drawRightString(PAGE[0] - 1.2 * cm, PAGE[1] - 11, "Implemented techniques checklist")
    canvas.setFillColor(LINE)
    canvas.rect(0, 0, PAGE[0], 14, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(1.2 * cm, 4, "Research / educational — not a clinical diagnosis")
    canvas.drawRightString(PAGE[0] - 1.2 * cm, 4, f"Page {doc.page}")
    canvas.restoreState()


def rows_table(st, rows):
    header = ["Module / where", "Step and purpose", "Reason", "Formulae and definitions", "Expected output"]
    widths = [3.6 * cm, 5.4 * cm, 5.2 * cm, 8.6 * cm, 4.6 * cm]
    data = [[Paragraph(h, st["th"]) for h in header]]
    for row in rows:
        data.append(
            [
                Paragraph(row[0], st["cellb"]),
                Paragraph(row[1], st["cell"]),
                Paragraph(row[2], st["cell"]),
                Paragraph(row[3], st["cell"]),
                Paragraph(row[4], st["cell"]),
            ]
        )
    t = Table(data, colWidths=widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), CREAM))
    t.setStyle(TableStyle(cmds))
    return t


# Each row: module/where, step, reason, formulae, output
M1 = [
    [
        "M1 · loader.py",
        "1. Dispatch the file by extension",
        "One entry point for the GUI and automation. Later modules never need to know the original format.",
        "ext = lower(file suffix). .mat → MATLAB/BMID loader. .s1p/.s2p/.s4p/.s8p → Touchstone loader. Else UnsupportedFileFormatError.",
        "Call to the correct parser.",
    ],
    [
        "M1 · validator.py",
        "2. Check the path before parsing",
        "Avoid crashing on a missing, empty or unknown file.",
        "File must exist, size &gt; 0, extension ∈ {.mat,.s1p,.s2p,.s4p,.s8p}.",
        "OK to parse, or a typed error.",
    ],
    [
        "M1 · touchstone_loader.py",
        "3. Import and inspect the Touchstone header; extract S21 on .s2p",
        "Identify the frequency unit, data format and reference impedance. Imaging uses transmission S21, not the whole 2×2, on the UG track.",
        "f_Hz = f_GHz×10⁹ ; f_Hz = f_MHz×10⁶ ; f_Hz = f_kHz×10³. Option line: # UNIT S {RI|MA|DB} R Z0. Row order after f: S11, S21, S12, S22.",
        "Header (unit, RI/MA/DB, Z0) and extracted S21 plus the full (n_f, n_ports, n_ports) matrix.",
    ],
    [
        "M1 · touchstone_loader.py",
        "4. Convert S21 to complex linear form",
        "Preserve magnitude and phase. Do not average, filter or subtract dB values.",
        "DB: A=10^(D/20), φ=φ_deg·π/180, S21=A(cosφ+j sinφ). MA: S21=mag(cosφ+j sinφ). RI: S21=R+jI. j²=−1.",
        "Raw complex S21 in linear units.",
    ],
    [
        "M1 · matlab_loader.py",
        "5. Load legacy or v7.3 MATLAB files",
        "Lab benches export both old MAT and HDF5 v7.3. Complex v7.3 arrays arrive as real/imag fields.",
        "If h5py.is_hdf5: visit datasets and convert {real,imag} → real+j imag. Else scipy.io.loadmat. Drop keys starting with __.",
        "Dictionary of numeric arrays.",
    ],
    [
        "M1 · matlab_loader.py",
        "6. Find the frequency vector and S-parameter array",
        "MATLAB files have no single schema.",
        "Name hints: freq/frequency/… ; else longest real increasing vector. If max(f)&lt;1000 treat as GHz: f_Hz=f×10⁹. S: name hints s_param/s21/… plus dim matching n_f. Known cube fallback: fd_data_s21 → 1–8 GHz, 1001 points.",
        "f in Hz and S oriented with frequency on axis 0.",
    ],
    [
        "M1 · bmid_loader.py",
        "7. Slice one UM-BMID scan from the 3-D cube",
        "A cube is many experiments. Loading without an index would mix them.",
        "cube[scan, freq, ant] ∈ ℂ. S = cube[k,:,:], shape (1001, n_ant). f=linspace(1e9,8e9,1001). Pair md_list_* → labels tum_diam/tum_rad, tum_x, tum_y, birads, ant_rad.",
        "One MicrowaveDataset plus tumor/healthy metadata in SI units.",
    ],
    [
        "M1 · physical_metadata.py",
        "8. Parse radius, wave speed and FOV from comments",
        "Reconstruction needs geometry even when it is only written in a note.",
        "Regex for antenna radius (m/cm), wave speed (m/s), x span / y span. cm → m by /100.",
        "metadata: antenna_radius_m, wave_speed_m_per_s, reconstruction_*_span_m.",
    ],
    [
        "M1 · validator.py · dataset_info.py",
        "9. Validate frequencies and S; build MicrowaveDataset",
        "Finite, non-empty arrays only. Downstream code uses one object for every format.",
        "f finite, S finite, no NaN/Inf. n_samples = |S|. Summary: n_freq, range, n_ports, size, variable names.",
        "MicrowaveDataset + DatasetSummary for the GUI panel.",
    ],
]

M2 = [
    [
        "M2 · preprocessing_pipeline.py",
        "10. Route .s2p versus general / BMID",
        "The UG brief is S21-only. BMID _adi is already cleaned and must not be cleaned again.",
        "If Touchstone and .s2p → S21 track. Else five-stage general loop. BMID preset: every method = none.",
        "Chosen track and a PreprocessingConfig.",
    ],
    [
        "M2 · touchstone_s21.py",
        "11. Validate and sort samples",
        "Prevent invalid values, duplicates and reverse frequency order from breaking IFFT.",
        "N_f = N_S21. Sort so f[k+1]&gt;f[k]. Drop/repair non-finite samples. Record notes.",
        "Ascending frequency and S21 vectors of equal length.",
    ],
    [
        "M2 · touchstone_s21.py",
        "12. Check uniform spacing; interpolate only if required",
        "A conventional IFFT needs constant Δf. Matched target/reference must share one grid.",
        "Δf[k]=f[k+1]−f[k]. If not uniform: Δf_u=(f_max−f_min)/(N_u−1), f_u[k]=f_min+k Δf_u, R_u=Interp(f,R,f_u), I_u=Interp(f,I,f_u), S_u=R_u+j I_u. Linear interpolation on Re and Im.",
        "Uniform f and complex S21. Flag interpolation_performed.",
    ],
    [
        "M2 · touchstone_s21.py",
        "13. Build magnitude and phase representations",
        "Inspect the original signal. Keep both wrapped and continuous phase.",
        "S_dB=20 log10 |S|. φ_w=atan2(I,R). φ_u=unwrap(φ_w). Unwrap fixes ±π jumps; it does not denoise.",
        "Magnitude dB, Re, Im, wrapped phase, unwrapped phase (GUI plots).",
    ],
    [
        "M2 · touchstone_s21.py",
        "14. Detect isolated invalid samples (Hampel / median / local)",
        "Stop one glitch from becoming a broadband artefact after IFFT.",
        "m_k=median{x[r]: r∈W_k}. MAD_k=median{|x[r]−m_k|}. σ̂_k=1.4826 MAD_k. Flag if |x[k]−m_k|&gt;λ σ̂_k (λ commonly 3). Replace flagged Re/Im. Default method: hampel.",
        "Corrected complex S21 and a count of repaired samples.",
    ],
    [
        "M2 · touchstone_s21.py",
        "15. Average repeated sweeps when available",
        "Reduce uncorrelated noise. Do not average different antenna positions or phantoms.",
        "S̄_21[k]=(1/M) Σ_m S_21^(m)[k]. M = number of identical-condition repeats.",
        "Complex-averaged S21, or skip with Averaging=No.",
    ],
    [
        "M2 · touchstone_s21.py",
        "16. Optional reference subtraction in frequency",
        "Remove a matched empty / tumor-free sweep when the user supplies one.",
        "S_diff[k]=S_T[k]−S_R[k]. Same grid, same preprocessing.",
        "Reference-subtracted S21, or skip.",
    ],
    [
        "M2 · filtering.py · touchstone_s21.py",
        "17. Apply a mild noise filter on Re and Im",
        "Reduce point-to-point noise without wiping a weak target bump. Never filter dB as if it were linear.",
        "Default .s2p: Savitzky–Golay, window L=7, polyorder=3, on R and I. Also available: moving average R_f[k]=(1/L)Σ R[k+r], Butterworth, median, Gaussian. S_f=R_f+j I_f.",
        "Filtered complex S21. GUI comparison plots.",
    ],
    [
        "M2 · touchstone_s21.py",
        "18. Quantify how much filtering moved the signal",
        "Check that cleaning did not rewrite the measurement.",
        "NRMSE = ||S_f − S_before||_2 / ||S_before||_2. Measures alteration, not image quality.",
        "NRMSE in the validation report; chosen filter name and parameters.",
    ],
    [
        "M2 · week3_time_domain.py",
        "19. Hamming window (keep unwindowed copy)",
        "Taper band edges so IFFT sidelobes / ringing drop. Reconstruction still defaults to filtered S21(f).",
        "w_H[k]=0.54−0.46 cos(2πk/(N−1)), k=0…N−1. S_H[k]=S_f[k] w_H[k].",
        "S21_filtered and S21_Hamming both stored.",
    ],
    [
        "M2 · week3_time_domain.py",
        "20. Window target and reference identically",
        "Subtraction is only valid if both sides saw the same window.",
        "S_{T,H}=S_{T,f} w_H and S_{R,H}=S_{R,f} w_H. Same w_H, same N.",
        "Matched windowed pair, or note that no reference was given.",
    ],
    [
        "M2 · ifft.py · week3_time_domain.py",
        "21. IFFT to a complex time response",
        "Delay-domain signals for clutter work and for DAS/DMAS.",
        "s[n]=(1/N) Σ_k S[k] e^{j 2π kn/N}  (NumPy ifft). Δt=1/(N Δf), t[n]=n Δt, T_period=1/Δf, B=f_max−f_min=(N−1)Δf. Need uniform Δf (rtol 1e-3).",
        "time_s and time-domain traces with and without Hamming.",
    ],
    [
        "M2 · week3_time_domain.py",
        "22. Matched reference subtraction in time",
        "Suppress the tumor-free / system response. IFFT is linear so subtract-in-f or subtract-in-t is the same if windows match.",
        "s_diff[n]=s_T[n]−s_R[n]  and  IFFT(S_T−S_R)=IFFT(S_T)−IFFT(S_R).",
        "Reference-subtracted time (and frequency) copies when a reference exists.",
    ],
    [
        "M2 · week3_time_domain.py",
        "23. Group-mean clutter removal, only if ≥2 channels",
        "Subtract the response common to a set of comparable antennas. A single .s2p has P=1 so this is skipped.",
        "μ_G[n]=(1/P) Σ_p s_p[n],  s_clean[n]=s_p[n]−μ_G[n].",
        "Clutter-reduced channels, or a skip note.",
    ],
    [
        "M2 · week3_time_domain.py",
        "24. Optional global normalize",
        "One scale for all channels. Do not normalize each channel on its own (that would destroy relative amplitudes).",
        "α = max_{p,n} |s_clean|,  s̃ = s_clean / α.",
        "Original-scale and globally normalized copies; α stored.",
    ],
    [
        "M2 · preprocessing_pipeline.py",
        "25. General-track filter → calibrate → normalize → background → artifacts",
        "Used for multi-trace MATLAB that is not a BMID pass-through.",
        "Calibration: S_cal=S_meas/S_ref, or self-cal: subtract per-frequency mean across traces. Normalization (phase kept): max |S|, min–max |S|, or z-score |S|. Background: S−S_bg. Artifacts: hybrid / other methods in artifact_suppression.py.",
        "Processed S of the same shape; stage_outputs snapshots.",
    ],
    [
        "M2 · preprocessing_pipeline.py",
        "26. Signal-quality report before vs after",
        "Tell the GUI whether cleaning helped.",
        "SNR_dB=10 log10(P_smooth / P_residual) with 5-sample moving-average “signal”. DR_dB=20 log10(max|S|/min|S|). Artifact reduction = 1 − E_after/E_before.",
        "SignalQualityReport numbers on the Processing Summary panel.",
    ],
    [
        "M2 · handover_export.py",
        "27. Validate and export the Module 3 handoff",
        "Give reconstruction everything it needs, with traceability.",
        "Check equal lengths, uniform f, finite complex values, matching metadata. Write .mat / optional .csv / .json.",
        "frequency_Hz, time_s, S21_raw, S21_filtered, S21_Hamming, S21_reference_subtracted, S21_clutter_removed (if any), Tx/Rx coordinates, labels, processing_parameters, validation report.",
    ],
]

M3 = [
    [
        "M3 · reconstruction_manager.py",
        "28. Infer geometry (radius, c, FOV, array)",
        "Pixels and delays are meaningless without a physical layout.",
        "x=linspace(x_min,x_max,n_x), y=linspace(y_min,y_max,n_y), default 64×64. a_i=(r cos θ_i, r sin θ_i). 360°: endpoint=False. 355° BMID arc: endpoint=True. Defaults: r=0.08 m, c=3e8, FOV ±5 cm. BMID: r=0.18 m, FOV ±6 cm. Optional r_eff=0.97(r−0.106)+0.148 m.",
        "ReconstructionConfig and antenna positions (N×2 m).",
    ],
    [
        "M3 · ifft.py",
        "29. Frequency → time for every antenna trace",
        "Beamformers sample delays, not frequencies.",
        "time_signals = ifft(S, n=n_f+pad, axis=0). t[n]=n/(N Δf). Fail if Δf is not uniform.",
        "Complex (N_time, n_traces) array.",
    ],
    [
        "M3 · das.py",
        "30. Delay-and-Sum (DAS)",
        "Add each antenna’s envelope at the two-way delay of that pixel. Coherence weight fights ring clutter.",
        "d_i=||a_i−r||, τ_i=2 d_i/c, n=round(τ_i/Δt). I_plain=Σ |s_i(τ_i)|. CF=|Σ s_i|² / (N Σ |s_i|²). I_DAS=I_plain · CF^γ, γ=0.55 (γ=0 is textbook DAS; cohort average).",
        "One real image (n_y, n_x).",
    ],
    [
        "M3 · dmas.py",
        "31. Delay-Multiply-and-Sum (DMAS)",
        "Keep energy that is consistent on two antennas at once. Sharper than DAS, slower.",
        "p_ij=s_i conj(s_j). signed_power(z,p)=Re{sign(z)(|z|+ε)^p}, p=0.30, ε=1e-9. I_DMAS=Σ_{i&lt;j} signed_power(p_ij). Cohort average; textbook p=0.50.",
        "One real image.",
    ],
    [
        "M3 · dmas_d4.py",
        "32. DMAS-D4 contrast map",
        "Compress dynamic range after DMAS. Cohort sweep keeps the textbook exponent 0.25. 0.55 was not better on average.",
        "I_D4 = sign(I_DMAS) · |I_DMAS|^{0.25}.",
        "One real image.",
    ],
    [
        "M3 · quality/metrics.py",
        "33. Image quality metrics",
        "Compare the three pictures with numbers, not only by eye.",
        "SNR=mean(I)/std(I). SCR=max(I)/mean(|I−mean(I)|). CCR=(mean(top 5%)−mean(rest))/std(rest). contrast=(max−min)/(max+min). FWHM_px = mean of half-max widths on the peak row and column.",
        "A metrics dict per image.",
    ],
    [
        "M3 · beamformer_selector.py",
        "34. Select which beamformer to keep",
        "One image goes to ROI. Mode depends on labels and the user.",
        "Quality: score=0.35 SNR_n+0.35 SCR_n+0.20 contrast_n+0.10(1−time_n), metrics min–max normalized across the three. Also: prefer DMAS-D4, force DAS/DMAS/D4, or tumor_gt (closest ROI to labeled (x,y)).",
        "Selected name + score table + reason string.",
    ],
    [
        "M3 · auto_calibrate.py",
        "35. Auto Tweak: search geometry + beamformer",
        "Wrong angle/flip/arc looks like a missed tumor. Search is only applied if it clearly beats the current settings.",
        "Sweep angles {0,90,180,270}, flips, CW/CCW, 360/355°, phase-delay on/off, ROI modes. DAS first, then top-8 full beamformers. GT score=1/(dist+1e-4). Apply if GT error drops by ≥0.5 mm.",
        "Best config, or keep the baseline.",
    ],
]

ROI = [
    [
        "ROI · roi_detector.py",
        "36. Normalize the selected image",
        "Threshold 0.70 must mean the same thing on dim and bright images.",
        "I_n=(I−I_min)/(I_max−I_min+ε), ε=1e-12.",
        "I_n ≈ [0,1].",
    ],
    [
        "ROI · roi_detector.py",
        "37. Gaussian smooth",
        "Stop a 1-pixel spike becoming its own region.",
        "I_s=Gaussian(I_n, σ=1.0 px).",
        "Smoothed image.",
    ],
    [
        "ROI · roi_detector.py",
        "38. Threshold + morphological opening",
        "Turn intensity into candidate pixels, then drop speckle.",
        "B=1{I_s≥T}, T=0.70. Opening with a 3×3 ones kernel.",
        "Binary map B.",
    ],
    [
        "ROI · roi_detector.py",
        "39. Connected components",
        "Separate islands are separate candidates.",
        "Label connected 1s. Drop area &lt; 6 px. If none: raise T by 0.05 (max 0.95) or box the global peak.",
        "List of components / bounding boxes.",
    ],
    [
        "ROI · roi_detector.py",
        "40. Score each component and pick spots",
        "Prefer a compact interior hotspot, not the largest splash or an edge ring.",
        "Score=(0.65 I_peak+0.35 I_mean)×(A/A_bbox)×1/(1+10 AreaRatio)×γ_edge, γ_edge=0.65 if the box touches the border else 1. Off-centre extra factor (0.2+0.8 r_norm)(1+1.5 r_norm). Keep ≤4 spots, gap ≥6 px, score ≥0.40×best (auto 0.28). GT may add the nearest peak within 3 cm.",
        "Ranked ROI list; primary = highest score.",
    ],
    [
        "ROI · reconstruction_manager.py",
        "41. High-resolution rebuild inside the box",
        "Same physics, finer pixels, smaller FOV.",
        "Convert box to metres. n=clip(box×4, min 96, max 256). Re-run DAS or DMAS or D4 on that span only.",
        "ROIRefinement image + local spans.",
    ],
    [
        "ROI · characterization.py",
        "42. Characterize the selected region",
        "Describe size and focus in centimetres.",
        "centroid via FOV interpolation. d_eq=2√(A_cm²/π). local SCR on the box. FWHM_cm=FWHM_px×px_cm. compactness=A_px/(w h). eccentricity=(aspect−1)/(aspect+1).",
        "TumorCharacterization numbers.",
    ],
    [
        "ROI · roi_detector.py",
        "43. Tumor-candidate suspicion and YES/NO",
        "Winning the ROI contest is not the same as looking tumor-like. Not a diagnosis.",
        "local_contrast=peak/mean_outside. ring_excess=peak/median(same radius). Suspicion=clip(0.38 tanh((lc−1)/2)+0.22/(1+(AR/0.08)²)+0.15(0.45+0.55 r_norm)+0.25 tanh((ring−1.15)/0.35),0,1). Conf=clip(Susp×0.85_edge×0.65_big×0.55_lowring,0,1). YES if Conf≥0.45. Forced NO if ring&lt;1.40 (no close GT). GT ≤3 cm can force YES.",
        "is_tumor_candidate, confidence, reason sentence.",
    ],
    [
        "ROI · confidence.py",
        "44. Overall reconstruction confidence",
        "One 0–1 number that blends picture quality, detection, beamformer margin, localization and focus.",
        "overall=0.22 image_q+0.28 detection+0.15 margin+0.22 localization+0.13 focus. High≥0.75, medium≥0.45, else low. localization=e^{−d_cm/2.5} if GT exists.",
        "ReconstructionConfidence + reasons.",
    ],
    [
        "Quality · tumor_taxonomy.py",
        "45. Metadata taxonomy (labels only)",
        "File experiments as healthy / small / medium / large, quadrant, shape. Not inferred from the image.",
        "small if d≤2 cm, medium if d≤4 cm, else large. Quadrant from (x,y); central if √(x²+y²)&lt;1 cm. Severity placeholder from size and BIRADS.",
        "short_label such as small/moderate/upper-left or healthy.",
    ],
]

AUTO = [
    [
        "Auto · discovery.py",
        "46. Discover measurement vs metadata files",
        "Do not treat md_list_*.mat as S-parameters.",
        "fd_data_*.mat = cube. md_list_*/metadata_*.mat = labels. .sNp / other .mat = measurement.",
        "List of DiscoveredFile with a kind and reason.",
    ],
    [
        "Auto · resolver.py",
        "47. Pair files and pick BMID scans",
        "A demo should cover healthy plus size/location variety, not 2000 identical clicks.",
        "Strategies: diverse (default ~8), all, tumor_and_healthy, first. Tumor rank prefers larger d near ~3 cm offset.",
        "ResolvedTarget list (path + scan index).",
    ],
    [
        "Auto · pipeline.py",
        "48. Run the same M1→M2→M3→ROI chain unattended",
        "Repeatable experiment. BMID: pass-through, FOV ±6 cm, 64×64. Else: savgol 7/3 + time-domain stages on.",
        "load → preprocess → reconstruct_all → select_best_beamformer → detect_rois → refine → characterize → confidence.",
        "Per-file evidence dict + figures.",
    ],
    [
        "Auto · policy.py · classifier.py",
        "49. Threshold profiles and pass/warn/fail",
        "Mentors can tighten or loosen gates without editing physics.",
        "Hard fail: load error, non-finite image, energy&lt;min, ROI area&lt;min. Soft warn: loc error, SCR, SNR, confidence, GT-but-not-candidate. balanced: loc≤3 cm, SCR≥2, SNR≥0.5, Conf≥0.35, area≥4.",
        "DatasetVerdict + MetricEvidence + recommended_upload.",
    ],
    [
        "Auto · reports.py · figures.py · cache.py · mcp_server.py",
        "50. Write analysis pack; cache; MCP/CLI/GUI facade",
        "One engine, three doors. Humans get markdown; machines get JSON.",
        "Showcase = labeled tumor first, else strongest candidate. Files: latest_auto_analysis.md, latest_auto_validation.json, PNG copies.",
        "Report paths and optional GUI handoff.",
    ],
]


def main():
    st = styles()
    path = OUT / "D_Techniques_Checklist.pdf"
    OUT.mkdir(parents=True, exist_ok=True)
    story = []
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Techniques and algorithms checklist", st["cover"]))
    story.append(
        Paragraph(
            "Every implemented method, where it lives, why it is there, the formula, and what it must produce. "
            "No week labels — this is the software as built.",
            st["sub"],
        )
    )
    story.append(
        Paragraph(
            "Read each row as a contract: <b>where</b> (module and file), <b>step and purpose</b>, "
            "<b>reason</b>, <b>formulae</b>, <b>expected output</b>. Module 2 rows 11–27 are the S21 teaching track "
            "plus the general MATLAB track and the Module 3 handoff. Module 3 and ROI start after that. "
            "Output is not a clinical diagnosis.",
            st["body"],
        )
    )
    story.append(Paragraph("Module 1 — Data acquisition", st["h1"]))
    story.append(rows_table(st, M1))
    story.append(PageBreak())
    story.append(Paragraph("Module 2 — Signal preprocessing and handoff", st["h1"]))
    story.append(rows_table(st, M2))
    story.append(PageBreak())
    story.append(Paragraph("Module 3 — Reconstruction", st["h1"]))
    story.append(rows_table(st, M3))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("ROI, quality labels, and automation", st["h1"]))
    story.append(rows_table(st, ROI + AUTO))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Required Module 3 handover (what export must contain)", st["h1"]))
    story.append(
        Paragraph(
            "Implemented in <b>preprocessing/handover_export.py</b>. The pack is: frequency_Hz and time_s; "
            "S21_raw, S21_filtered and S21_Hamming; S21_reference_subtracted and S21_clutter_removed where those stages ran; "
            "Tx_coordinates, Rx_coordinates and measurement_labels; processing_parameters and the validation report. "
            "If the file did not provide coordinates, Tx/Rx are placed on a circle of antenna_radius_m. "
            "Formats: .mat, optional .csv, .json.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "50 rows. If a method is not in this table, it is not part of the implemented pipeline.",
            st["note"],
        )
    )

    doc = SimpleDocTemplate(
        str(path),
        pagesize=PAGE,
        leftMargin=1.15 * cm,
        rightMargin=1.15 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.1 * cm,
        title="Techniques and algorithms checklist",
        author="Microwave Imaging Framework",
    )
    doc.build(
        story,
        onFirstPage=lambda c, d: header_footer(c, d, "Techniques and algorithms checklist"),
        onLaterPages=lambda c, d: header_footer(c, d, "Techniques and algorithms checklist"),
    )
    print("Wrote", path)


if __name__ == "__main__":
    main()

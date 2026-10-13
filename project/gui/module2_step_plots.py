"""
gui/module2_step_plots.py

One figure per Module 2 step (Signal Preprocessing brief, steps 1–16), drawn
from a finished PreprocessingResult. Used by the "Module 2 Steps" tab.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from matplotlib.figure import Figure

from gui.plotting import finish_figure, show_message
from preprocessing.preprocessing_pipeline import PreprocessingResult
from preprocessing.week3_time_domain import hamming_window

STEP_TITLES: list[str] = [
    "1. Identify header, extract S21",
    "2. Convert S21 to complex linear",
    "3. Validate and sort",
    "4. Uniform spacing / interpolation",
    "5. Magnitude and phase plots",
    "6. Isolated invalid samples",
    "7. Average repeated sweeps",
    "8. Apply and compare mild filters",
    "9. Quantify filtering change (NRMSE)",
    "10. Hamming window",
    "11. Window target and reference identically",
    "12. Complex time-domain response",
    "13. Matched reference subtraction",
    "14. Common channel clutter removal",
    "15. Globally normalized copy",
    "16. Validate and export for Module 3",
]

_S2P_ONLY = (
    "Steps 1–11 operate on a single Touchstone (.s2p) S21 trace.\n"
    "This dataset is multi-channel (.mat); its frequency-domain stages are in the 'Stages' tab.\n"
    "Select steps 12–16 in the list for this dataset."
)


def _ghz(hz: np.ndarray) -> np.ndarray:
    return np.asarray(hz, dtype=float) / 1e9


def _first_channel(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)
    return data[:, 0] if data.ndim == 2 else data


def _info(ax, lines: list[str], fontsize: float = 9) -> None:
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=fontsize,
            transform=ax.transAxes)


def _finish(fig: Figure, title: str, formula: str = "") -> None:
    finish_figure(fig)
    fig.suptitle(title + (f"\n{formula}" if formula else ""), fontsize=10)


def _step1(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    h = ts.header
    ax_info = fig.add_subplot(1, 2, 1)
    _info(ax_info, [
        f"File            : {r.original_dataset.file_name}",
        f"Frequency unit  : {h.frequency_unit}",
        f"Data format     : {h.data_format}",
        f"Ref. impedance  : {h.reference_impedance_ohm:g} ohm",
        f"Samples         : {h.n_frequency_samples}",
        f"Range           : {h.start_frequency_hz / 1e9:.4g} – {h.end_frequency_hz / 1e9:.4g} GHz",
        "",
        "Extracted       : S21 only (column 3/4)",
    ])
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(_ghz(ts.raw_frequencies_hz), np.abs(ts.raw_s21))
    ax.set_title("Extracted |S21| (linear)")
    ax.set_xlabel("Frequency (GHz)")
    ax.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[0], "Option line '# <unit> S <DB|MA|RI> R <Z0>' parsed; only S21 kept")


def _step2(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    f = _ghz(ts.raw_frequencies_hz)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(f, ts.raw_s21.real, label="Re")
    ax1.plot(f, ts.raw_s21.imag, label="Im")
    ax1.set_title("Complex linear S21")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ts.raw_s21.real, ts.raw_s21.imag, lw=1)
    ax2.set_title("Complex plane (Re vs Im)")
    ax2.set_xlabel("Re")
    ax2.set_ylabel("Im")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[1], "DB: 10^(dB/20)·e^{jθ}   MA: |S|·e^{jθ}   RI: Re + j·Im   (no dB averaging)")


def _step3(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    f = ts.raw_frequencies_hz
    df = np.diff(f)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(np.arange(f.size), _ghz(f), ".", ms=3)
    ax1.set_title("Frequency vs sample index (sorted)")
    ax1.set_xlabel("index k")
    ax1.set_ylabel("GHz")
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(1, 2, 2)
    _info(ax2, [
        f"Samples            : {f.size}",
        f"Lengths f / S21    : {f.size} / {ts.raw_s21.size}",
        f"Finite frequencies : {bool(np.all(np.isfinite(f)))}",
        f"Finite S21         : {bool(np.all(np.isfinite(ts.raw_s21)))}",
        f"Strictly ascending : {bool(np.all(df > 0))}",
        f"Duplicates         : {int(f.size - np.unique(f).size)}",
    ])
    _finish(fig, STEP_TITLES[2], "Equal lengths, no NaN/Inf, ascending order, unique frequencies")


def _step4(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    raw_df = np.diff(ts.raw_frequencies_hz) / 1e6
    uni_df = np.diff(ts.uniform_frequencies_hz) / 1e6
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(raw_df, ".", label="raw Δf")
    ax1.plot(uni_df, "-", label="uniform Δf")
    ax1.set_title("Δf per interval (MHz)")
    ax1.set_xlabel("interval k")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(1, 2, 2)
    _info(ax2, [
        f"Interpolation performed : {'Yes' if ts.report.interpolation_performed else 'No'}",
        f"Raw Δf min / max (MHz)  : {raw_df.min():.4g} / {raw_df.max():.4g}",
        f"Uniform Δf (MHz)        : {uni_df.mean():.6g}",
        f"N raw / N uniform       : {ts.raw_frequencies_hz.size} / {ts.uniform_frequencies_hz.size}",
        "",
        "Interpolation on Re and Im separately",
    ])
    _finish(fig, STEP_TITLES[3], "f_u[k] = f_min + kΔf,  S21,u = interp(Re) + j·interp(Im)")


def _step5(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    f = _ghz(ts.uniform_frequencies_hz)
    panels = [
        (ts.raw_magnitude_db, "|S21| (dB)", None),
        (ts.uniform_s21.real, "Real(S21)", None),
        (ts.uniform_s21.imag, "Imag(S21)", None),
    ]
    for i, (y, title, _) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i)
        ax.plot(f, y)
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(f, ts.wrapped_phase_deg, label="wrapped")
    ax.plot(f, ts.unwrapped_phase_deg, label="unwrapped")
    ax.set_title("Phase (deg)")
    ax.set_xlabel("Frequency (GHz)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[4], "S21,dB = 20 log10|S21,u|,  φw = atan2(I, R),  φu = unwrap(φw)")


def _hampel_band(x: np.ndarray, k: int, lam: float) -> tuple[np.ndarray, np.ndarray]:
    n = x.size
    med = np.array([np.median(x[max(0, i - k):i + k + 1]) for i in range(n)])
    mad = np.array([np.median(np.abs(x[max(0, i - k):i + k + 1] - med[i])) for i in range(n)])
    d = np.diff(x)
    floor = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0) if n > 2 else 0.0
    return med, lam * np.maximum(1.4826 * mad, floor)


def _step6(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    f = _ghz(ts.uniform_frequencies_hz)
    kwargs = getattr(r.config, "spike_detection_kwargs", {}) or {}
    k, lam = int(kwargs.get("window_size", 3)), float(kwargs.get("n_sigmas", 3.0))
    idx = list(ts.report.corrected_sample_indices)
    for i, (part, name) in enumerate(((np.real, "Real"), (np.imag, "Imag")), start=1):
        ax = fig.add_subplot(2, 1, i)
        x = part(ts.uniform_s21)
        med, band = _hampel_band(x, k, lam)
        ax.fill_between(f, med - band, med + band, color="orange", alpha=0.25, label="m_k ± λσ̂_k")
        ax.plot(f, x, color="lightgray", lw=1.5, label="before")
        ax.plot(f, part(ts.corrected_s21), lw=1, label="after")
        if idx:
            ax.scatter(f[idx], part(ts.corrected_s21)[idx], color="red", s=15, zorder=5, label="corrected")
        ax.set_title(f"{name}(S21) — {len(idx)} sample(s) corrected")
        ax.set_xlabel("Frequency (GHz)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[5], f"m_k = median(W_k), σ̂_k = 1.4826·MAD_k, flag |x−m_k| > {lam:g}σ̂_k (window {2 * k + 1})")


def _step7(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    notes = [n for n in ts.report.notes if "repeated" in n.lower() or "averaged" in n.lower()]
    if ts.averaged_s21 is None:
        show_message(fig, "No repeated sweeps were averaged.\n\nSelect repeated .s2p files in Settings and run again.\n\n"
                     + "\n".join(notes))
        _finish(fig, STEP_TITLES[6], "S̄21[k] = (1/M) Σ_m S21^(m)[k]")
        return
    f = _ghz(ts.uniform_frequencies_hz)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(f, ts.corrected_s21.real, color="lightgray", lw=1.5, label="single sweep (corrected)")
    ax.plot(f, ts.averaged_s21.real, lw=1, label="complex average")
    ax.set_title("Real(S21)")
    ax.set_xlabel("Frequency (GHz)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(f, np.abs(ts.corrected_s21 - ts.averaged_s21), color="tab:green")
    ax2.set_title("|single − average| (noise removed) — " + ("; ".join(notes) or "averaging performed"), fontsize=9)
    ax2.set_xlabel("Frequency (GHz)")
    ax2.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[6], "S̄21[k] = (1/M) Σ_m S21^(m)[k]   (complex, never in dB)")


def _step8(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    if not ts.filter_comparison or ts.filter_input_s21 is None:
        show_message(fig, "Filter comparison not available for this run.")
        _finish(fig, STEP_TITLES[7])
        return
    f = _ghz(ts.uniform_frequencies_hz)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(f, ts.filter_input_s21.real, color="lightgray", lw=1.5, label="unfiltered")
    for method, e in ts.filter_comparison.items():
        ax1.plot(f, e["filtered"].real, lw=0.9,
                 label=f"{method}: noise ÷{e['noise_reduction']:.1f}, NRMSE {e['nrmse']:.3f}")
    ax1.set_title("Real(S21), identical mild settings")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.legend(fontsize=6, loc="best")
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(2, 1, 2)
    for method, e in ts.filter_comparison.items():
        ax2.plot(f, np.abs(e["filtered"] - ts.filter_input_s21), lw=0.8, label=method)
    ax2.set_title("|filtered − unfiltered|")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.legend(fontsize=6, loc="best")
    ax2.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[7], "R_f[k] = (1/L) Σ_{r=-q..q} R[k+r] (same for I),  L = 2q + 1, reflect edges")


def _step9(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    f = _ghz(ts.uniform_frequencies_hz)
    base = ts.filter_input_s21 if ts.filter_input_s21 is not None else ts.corrected_s21
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(f, np.abs(base), color="lightgray", lw=1.5, label="S̄21 (filter input)")
    ax.plot(f, np.abs(ts.filtered_s21), lw=1, label="S21,f (selected filter)")
    ax.set_title("|S21| before vs after selected filter")
    ax.set_xlabel("Frequency (GHz)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    rep = ts.report
    _info(fig.add_subplot(2, 1, 2), [
        f"Selected filter : {rep.filter_used}",
        f"Parameters      : {rep.filter_parameters or 'defaults'}",
        f"NRMSE           : {rep.nrmse:.4f}",
        "",
        "NRMSE measures alteration, not reconstruction quality.",
    ])
    _finish(fig, STEP_TITLES[8], "NRMSE = ||S21,f − S̄21|| / ||S̄21||")


def _step10(fig: Figure, r: PreprocessingResult) -> None:
    ts = r.touchstone_s21_result
    f = _ghz(ts.uniform_frequencies_hz)
    w = hamming_window(f.size)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(f, w, color="tab:orange")
    ax1.set_title(f"w_H[k], N = {f.size}")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(f, ts.filtered_s21.real, color="gray", label="S21,f (kept)")
    ax2.plot(f, ts.windowed_s21.real, label="S21,H = S21,f · w_H")
    ax2.set_title("Real(S21): filtered vs windowed")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[9], "w_H[k] = 0.54 − 0.46 cos(2πk/(N−1)),  S21,H[k] = S21,f[k]·w_H[k]")


def _step11(fig: Figure, r: PreprocessingResult) -> None:
    w3 = r.week3_result
    if w3 is None or w3.s21_reference_hamming is None:
        show_message(fig, "No matched reference was windowed.\n\n"
                     "Select a tumour-free reference .s2p in Settings (Week 3 enabled) and run again.")
        _finish(fig, STEP_TITLES[10], "S_p,T,H[k] = S_p,T,f[k]·w_H[k],  S_p,R,H[k] = S_p,R,f[k]·w_H[k]")
        return
    f = _ghz(w3.frequencies_hz)
    w = hamming_window(f.size)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(f, np.abs(_first_channel(w3.s21_hamming)), label="|S_T,H| target")
    ax.plot(f, np.abs(_first_channel(w3.s21_reference_hamming)), "--", label="|S_R,H| reference")
    ax.plot(f, w * np.abs(_first_channel(w3.s21_hamming)).max(), ":", color="gray", label="w_H (scaled)")
    ax.set_xlabel("Frequency (GHz)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[10], "S_p,T,H[k] = S_p,T,f[k]·w_H[k],  S_p,R,H[k] = S_p,R,f[k]·w_H[k]")


def _week3_or_message(fig: Figure, r: PreprocessingResult, title: str, formula: str):
    if r.week3_result is None:
        show_message(fig, "Week 3 time-domain stage did not run.\n\nEnable 'Week 3 time-domain' in Settings and run again.")
        _finish(fig, title, formula)
        return None
    return r.week3_result


def _step12(fig: Figure, r: PreprocessingResult) -> None:
    formula = "s[n] = (1/N) Σ S[k] e^{j2πkn/N},  Δt = 1/(NΔf),  T = 1/Δf,  δt ≈ 1/B,  B = (N−1)Δf"
    w3 = _week3_or_message(fig, r, STEP_TITLES[11], formula)
    if w3 is None:
        return
    t = w3.time_s * 1e9
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, np.abs(_first_channel(w3.time_filtered)), color="gray", label="without Hamming")
    ax.plot(t, np.abs(_first_channel(w3.time_hamming)), label="with Hamming")
    ax.set_title("|s21(t)| (channel 0)")
    ax.set_xlabel("Time (ns)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _info(fig.add_subplot(2, 1, 2), [
        f"N          : {w3.time_s.size}",
        f"Δf         : {w3.delta_f_hz / 1e6:.4g} MHz",
        f"Δt         : {w3.delta_t_s * 1e12:.4g} ps",
        f"T_period   : {w3.period_s * 1e9:.4g} ns",
        f"B          : {w3.bandwidth_hz / 1e9:.4g} GHz",
        f"δt ≈ 1/B   : {w3.time_resolution_s * 1e12:.4g} ps",
    ])
    _finish(fig, STEP_TITLES[11], formula)


def _step13(fig: Figure, r: PreprocessingResult) -> None:
    formula = "S_diff,H = S_T,H − S_R,H,   s_diff[n] = s_T[n] − s_R[n]"
    w3 = _week3_or_message(fig, r, STEP_TITLES[12], formula)
    if w3 is None:
        return
    if w3.time_reference_subtracted is None:
        notes = [n for n in (r.validation_report.notes if r.validation_report else []) if "eference" in n]
        show_message(fig, "No matched reference subtraction in this run.\n\n" + "\n".join(notes))
        _finish(fig, STEP_TITLES[12], formula)
        return
    t = w3.time_s * 1e9
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, np.abs(_first_channel(w3.time_hamming)), label="|s_T(t)| target")
    ax1.plot(t, np.abs(_first_channel(w3.time_reference_hamming)), "--", label="|s_R(t)| reference")
    ax1.set_title("Before subtraction")
    ax1.set_xlabel("Time (ns)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, np.abs(_first_channel(w3.time_reference_subtracted)), color="tab:red", label="|s_diff(t)|")
    ax2.set_title("After matched subtraction")
    ax2.set_xlabel("Time (ns)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[12], formula)


def _step14(fig: Figure, r: PreprocessingResult) -> None:
    formula = "μ_G[n] = (1/P) Σ_{p∈G} s_p,diff[n],   s_p,clean[n] = s_p,diff[n] − μ_G[n]"
    w3 = _week3_or_message(fig, r, STEP_TITLES[13], formula)
    if w3 is None:
        return
    notes = [n for n in w3.notes if "clutter" in n.lower()]
    if not w3.clutter_removal_performed:
        show_message(fig, "Clutter removal not applied.\n\n" + "\n".join(notes))
        _finish(fig, STEP_TITLES[13], formula)
        return
    before = w3.time_reference_subtracted if w3.time_reference_subtracted is not None else w3.time_hamming
    t = w3.time_s * 1e9
    ax1 = fig.add_subplot(2, 1, 1)
    im = ax1.imshow(np.abs(before).T, aspect="auto", origin="lower",
                    extent=(t[0], t[-1], -0.5, before.shape[1] - 0.5))
    fig.colorbar(im, ax=ax1)
    ax1.set_title("|s_p,diff(t)| before")
    ax1.set_ylabel("channel p")
    ax2 = fig.add_subplot(2, 1, 2)
    im = ax2.imshow(np.abs(w3.time_clutter_removed).T, aspect="auto", origin="lower",
                    extent=(t[0], t[-1], -0.5, before.shape[1] - 0.5))
    fig.colorbar(im, ax=ax2)
    ax2.set_title("|s_p,clean(t)| after — " + "; ".join(notes))
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("channel p")
    _finish(fig, STEP_TITLES[13], formula)


def _step15(fig: Figure, r: PreprocessingResult) -> None:
    formula = "α = max_{p,n} |s_p,clean[n]|,   s̃_p,clean[n] = s_p,clean[n] / α   (one α for all channels)"
    w3 = _week3_or_message(fig, r, STEP_TITLES[14], formula)
    if w3 is None:
        return
    if w3.time_global_normalized is None:
        show_message(fig, "Global normalization is disabled for this run.")
        _finish(fig, STEP_TITLES[14], formula)
        return
    original = np.asarray(w3.time_clutter_removed)
    normalized = np.asarray(w3.time_global_normalized)
    if original.ndim == 1:
        original, normalized = original.reshape(-1, 1), normalized.reshape(-1, 1)
    ax1 = fig.add_subplot(2, 1, 1)
    peaks = np.max(np.abs(normalized), axis=0)
    ax1.bar(np.arange(peaks.size), peaks)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(np.arange(peaks.size) if peaks.size <= 32 else np.linspace(0, peaks.size - 1, 9, dtype=int))
    ax1.set_title(f"Peak per channel after normalization (α = {w3.global_scale_alpha:.4g})")
    ax1.set_xlabel("channel p")
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(2, 1, 2)
    t = w3.time_s * 1e9
    ax2.plot(t, np.abs(original[:, 0]), color="gray", label="original scale (ch 0)")
    ax2b = ax2.twinx()
    ax2b.plot(t, np.abs(normalized[:, 0]), color="tab:blue", lw=1, label="normalized (ch 0)")
    ax2.set_title("Original-scale and normalized copies")
    ax2.set_xlabel("Time (ns)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2b.legend(fontsize=8, loc="center right")
    ax2.grid(alpha=0.3)
    _finish(fig, STEP_TITLES[14], formula)


def _step16(fig: Figure, r: PreprocessingResult) -> None:
    from preprocessing.handover_export import build_handover_payload, validate_handover_payload

    payload = build_handover_payload(r)
    checks = validate_handover_payload(payload)
    passed = sum(c["passed"] for c in checks)
    lines = [("✔ " if c["passed"] else "✘ ") + c["check"] for c in checks]
    half = (len(lines) + 1) // 2
    _info(fig.add_subplot(1, 2, 1), lines[:half], fontsize=7.5)
    _info(fig.add_subplot(1, 2, 2), lines[half:] + ["", f"code version: {payload['code_version']}"], fontsize=7.5)
    status = "ready to export" if passed == len(checks) else "EXPORT WILL BE BLOCKED"
    _finish(fig, STEP_TITLES[15],
            f"{passed}/{len(checks)} checks passed — {status}  (Export writes .mat, .csv, .json report, plots)")


_STEP_FUNCS: list[Callable[[Figure, PreprocessingResult], None]] = [
    _step1, _step2, _step3, _step4, _step5, _step6, _step7, _step8,
    _step9, _step10, _step11, _step12, _step13, _step14, _step15, _step16,
]


def plot_module2_step(figure: Figure, step_index: int, result: PreprocessingResult | None) -> None:
    """Draw step ``step_index`` (0-based) of Module 2 for ``result`` into ``figure``."""
    figure.clear()
    if result is None:
        show_message(figure, "Run Module 2 preprocessing to see each step.")
        return
    if step_index < 11 and result.touchstone_s21_result is None:
        show_message(figure, _S2P_ONLY)
        figure.suptitle(STEP_TITLES[step_index], fontsize=10)
        return
    try:
        _STEP_FUNCS[step_index](figure, result)
    except Exception as exc:  # a plotting problem must not break the GUI
        figure.clear()
        show_message(figure, f"Could not draw {STEP_TITLES[step_index]}:\n{exc}")

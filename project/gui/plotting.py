"""Shared matplotlib helpers for tracing S-parameter transformations."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


def make_gui_figure(width: float = 8.0, height: float = 6.0) -> Figure:
    """Create a figure that will not run tight_layout (it overlaps in Qt)."""
    figure = Figure(figsize=(width, height))
    figure.set_tight_layout(False)
    figure.set_constrained_layout(False)
    return figure


def finish_figure(figure: Figure) -> None:
    """Leave room for titles, axis labels, and colorbars so panels do not stack."""
    figure.set_constrained_layout(False)
    figure.set_tight_layout(False)
    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.88,
        bottom=0.12,
        hspace=0.72,
        wspace=0.42,
    )


def style_axes(ax, title: str, xlabel: str, ylabel: str, *, legend: bool = False) -> None:
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.9, borderaxespad=0.25)


def show_message(figure: Figure, message: str) -> None:
    figure.clear()
    ax = figure.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=10)
    finish_figure(figure)


def _flat_traces(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data.reshape(data.shape[0], -1)


def _freq_axis_ghz(frequencies_hz: np.ndarray, n_samples: int) -> tuple[np.ndarray, str]:
    freqs_ghz = np.asarray(frequencies_hz, dtype=float) / 1e9
    if freqs_ghz.size == n_samples:
        return freqs_ghz, "Frequency (GHz)"
    return np.arange(n_samples, dtype=float), "Sample"


def plot_complex_matrix(
    figure: Figure,
    frequencies_hz: np.ndarray,
    s_parameters,
    title: str,
    n_traces: int = 4,
    previous=None,
) -> None:
    """Show how a complex S matrix looks: real, imag, |S| dB, heatmap.

    If ``previous`` is provided, the heatmap becomes the change vs that step
    so a no-op stage is obvious.
    """
    figure.clear()
    flat = _flat_traces(s_parameters)
    x, xlabel = _freq_axis_ghz(frequencies_hz, flat.shape[0])
    count = min(n_traces, flat.shape[1])
    rows = 2 if previous is None else 3
    figure.suptitle(title, fontsize=11, y=0.98)
    gs = figure.add_gridspec(rows, 2, hspace=0.78, wspace=0.38)

    ax_re = figure.add_subplot(gs[0, 0])
    ax_im = figure.add_subplot(gs[0, 1])
    ax_db = figure.add_subplot(gs[1, 0])
    ax_hm = figure.add_subplot(gs[1, 1])
    for idx in range(count):
        ax_re.plot(x, flat[:, idx].real, alpha=0.85, label=f"tr {idx}")
        ax_im.plot(x, flat[:, idx].imag, alpha=0.85, label=f"tr {idx}")
        ax_db.plot(
            x,
            20.0 * np.log10(np.abs(flat[:, idx]) + 1e-12),
            alpha=0.85,
            label=f"tr {idx}",
        )
    style_axes(ax_re, "Real", xlabel, "Real", legend=count > 1)
    style_axes(ax_im, "Imag", xlabel, "Imag")
    style_axes(ax_db, "|S| (dB)", xlabel, "Magnitude (dB)")

    mag_db = 20.0 * np.log10(np.abs(flat) + 1e-12)
    im = ax_hm.imshow(
        mag_db.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    style_axes(ax_hm, "Matrix |S| (dB)", "Frequency sample", "Antenna / trace")
    figure.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    if previous is not None:
        prev = _flat_traces(previous)
        n0 = min(flat.shape[0], prev.shape[0])
        n1 = min(flat.shape[1], prev.shape[1])
        delta = flat[:n0, :n1] - prev[:n0, :n1]
        rms = float(np.sqrt(np.mean(np.abs(delta) ** 2)))
        ax_d = figure.add_subplot(gs[2, 0])
        ax_d.plot(x[:n0], delta[:, 0].real, label="Δ real")
        ax_d.plot(x[:n0], delta[:, 0].imag, label="Δ imag", alpha=0.8)
        style_axes(
            ax_d,
            f"Change vs previous (RMS={rms:.2e})",
            xlabel,
            "Δ S",
            legend=True,
        )
        ax_n = figure.add_subplot(gs[2, 1])
        if rms < 1e-15:
            ax_n.axis("off")
            ax_n.text(
                0.5,
                0.5,
                "This step did not change the matrix\n(same Real/Imag as the previous step).",
                ha="center",
                va="center",
                fontsize=10,
            )
        else:
            im2 = ax_n.imshow(
                np.abs(delta).T,
                aspect="auto",
                origin="lower",
                cmap="magma",
                interpolation="nearest",
            )
            style_axes(ax_n, "|ΔS| matrix", "Frequency sample", "Antenna / trace")
            figure.colorbar(im2, ax=ax_n, fraction=0.046, pad=0.04)
    finish_figure(figure)


def plot_freq_to_time(
    figure: Figure,
    frequencies_hz: np.ndarray,
    s_parameters,
    time_s: np.ndarray,
    time_signals,
) -> None:
    """One-screen look at frequency-domain S becoming a time-domain waveform."""
    figure.clear()
    flat = _flat_traces(s_parameters)
    x, xlabel = _freq_axis_ghz(frequencies_hz, flat.shape[0])
    t_ns = np.asarray(time_s, dtype=float) * 1e9
    time_flat = np.abs(_flat_traces(time_signals))
    n = min(3, flat.shape[1], time_flat.shape[1])
    figure.suptitle("Frequency domain → time domain", fontsize=11, y=0.98)
    gs = figure.add_gridspec(1, 2, wspace=0.32)

    ax1 = figure.add_subplot(gs[0, 0])
    for idx in range(n):
        ax1.plot(x, flat[:, idx].real, alpha=0.8, label=f"real {idx}")
        ax1.plot(x, flat[:, idx].imag, alpha=0.55, linestyle="--", label=f"imag {idx}")
    style_axes(ax1, "Input S (freq)", xlabel, "Real / Imag", legend=True)

    ax2 = figure.add_subplot(gs[0, 1])
    for idx in range(n):
        ax2.plot(t_ns, time_flat[:, idx], alpha=0.85, label=f"|s(t)| {idx}")
    ax2.plot(t_ns, np.mean(time_flat, axis=1), color="k", lw=1.1, alpha=0.7, label="mean")
    style_axes(ax2, "After freq → time", "Time (ns)", "|s(t)|", legend=True)
    finish_figure(figure)


def plot_reconstruction_trace(
    figure: Figure,
    frequencies_hz: np.ndarray,
    s_parameters,
    time_s,
    time_signals,
    images: dict,
    selected_name: str,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    configure_image,
    overlay_rois=None,
    scores: dict | None = None,
) -> None:
    """Trace reconstruction: S real/imag → time look → DAS/DMAS/D4 images."""
    figure.clear()
    flat = _flat_traces(s_parameters)
    x, xlabel = _freq_axis_ghz(frequencies_hz, flat.shape[0])
    figure.suptitle("Reconstruction trace (S → time → images)", fontsize=11, y=0.98)
    gs = figure.add_gridspec(2, 3, hspace=0.62, wspace=0.36)
    ax_re = figure.add_subplot(gs[0, 0])
    ax_im = figure.add_subplot(gs[0, 1])
    ax_t = figure.add_subplot(gs[0, 2])
    n = min(3, flat.shape[1])
    for idx in range(n):
        ax_re.plot(x, flat[:, idx].real, alpha=0.85, label=f"tr {idx}")
        ax_im.plot(x, flat[:, idx].imag, alpha=0.85, label=f"tr {idx}")
    style_axes(ax_re, "1. Input Real(S)", xlabel, "Real", legend=True)
    style_axes(ax_im, "2. Input Imag(S)", xlabel, "Imag")

    if time_s is not None and time_signals is not None:
        t_ns = np.asarray(time_s, dtype=float) * 1e9
        env = np.abs(_flat_traces(time_signals))
        for idx in range(min(n, env.shape[1])):
            ax_t.plot(t_ns, env[:, idx], alpha=0.85, label=f"tr {idx}")
        style_axes(ax_t, "3. Freq → time |s(t)|", "Time (ns)", "|s(t)|")
    else:
        ax_t.axis("off")
        ax_t.set_title("3. Freq → time unavailable", fontsize=10, pad=8)

    for col, name in enumerate(("DAS", "DMAS", "DMAS-D4")):
        ax = figure.add_subplot(gs[1, col])
        image = images.get(name)
        if image is None:
            ax.axis("off")
            ax.set_title(f"{col + 4}. {name} (missing)", fontsize=10, pad=8)
            continue
        marker = " ★" if name == selected_name else ""
        score = None if not scores else scores.get(name)
        score_txt = f"  {score:.3f}" if isinstance(score, (int, float)) else ""
        configure_image(
            ax,
            np.asarray(image),
            f"{col + 4}. {name}{marker}{score_txt}",
            x_span=x_span,
            y_span=y_span,
        )
        if overlay_rois is not None:
            overlay_rois(ax, np.asarray(image))
    finish_figure(figure)

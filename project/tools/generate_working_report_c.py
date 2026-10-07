"""Rebuild Report C: module-wise, tiny-step working report with sample figures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.ndimage import binary_opening, find_objects, gaussian_filter, label

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results"
FIG = OUT / "report_c_figures"
S2P = PROJECT / "datasets" / "sample_touchstone.s2p"

NAVY = colors.HexColor("#1b365d")
TEAL = colors.HexColor("#1f6f8b")
CREAM = colors.HexColor("#f4f7fb")
LINE = colors.HexColor("#c5d0de")
EPS = 1e-12


def styles():
    base = getSampleStyleSheet()
    return {
        "cover": ParagraphStyle("c", parent=base["Title"], fontName="Times-Bold", fontSize=20, leading=24, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontName="Times-Italic", fontSize=11, leading=15, textColor=TEAL, alignment=TA_CENTER, spaceAfter=12),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Times-Bold", fontSize=14, leading=18, textColor=NAVY, spaceBefore=12, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Times-Bold", fontSize=12, leading=15, textColor=TEAL, spaceBefore=9, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="Times-Bold", fontSize=11, leading=14, textColor=NAVY, spaceBefore=7, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["Normal"], fontName="Times-Roman", fontSize=10, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=6),
        "note": ParagraphStyle("n", parent=base["Normal"], fontName="Times-Italic", fontSize=9, leading=12, textColor=colors.HexColor("#444"), spaceAfter=7),
        "io": ParagraphStyle("io", parent=base["Normal"], fontName="Times-Roman", fontSize=9.5, leading=13, backColor=CREAM, borderPadding=5, spaceAfter=7),
        "formula": ParagraphStyle("f", parent=base["Normal"], fontName="Courier", fontSize=8.4, leading=11.5, backColor=CREAM, borderPadding=5, spaceAfter=7),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontName="Courier", fontSize=7, leading=9, alignment=TA_LEFT),
        "th": ParagraphStyle("th", parent=base["Normal"], fontName="Times-Bold", fontSize=8, leading=10, textColor=colors.white, alignment=TA_CENTER),
        "cap": ParagraphStyle("cap", parent=base["Normal"], fontName="Times-Italic", fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#333"), spaceAfter=10),
        "bullet": ParagraphStyle("bu", parent=base["Normal"], fontName="Times-Roman", fontSize=10, leading=13.2, leftIndent=10, spaceAfter=2),
    }


def header_footer(canvas, doc, title):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 18, A4[0], 18, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(1.6 * cm, A4[1] - 13, title)
    canvas.drawRightString(A4[0] - 1.6 * cm, A4[1] - 13, "Microwave Imaging Framework")
    canvas.setFillColor(LINE)
    canvas.rect(0, 0, A4[0], 16, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(1.6 * cm, 5, "Research / educational — not a clinical diagnosis")
    canvas.drawRightString(A4[0] - 1.6 * cm, 5, f"Page {doc.page}")
    canvas.restoreState()


def P(text, st, key="body"):
    return Paragraph(text, st[key])


def bullets(items, st):
    return [Paragraph(f"• {x}", st["bullet"]) for x in items]


def table(data, widths, st):
    rows = []
    for i, row in enumerate(data):
        cells = []
        for cell in row:
            cells.append(cell if isinstance(cell, Paragraph) else Paragraph(str(cell), st["th"] if i == 0 else st["cell"]))
        rows.append(cells)
    t = Table(rows, colWidths=widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), CREAM))
    t.setStyle(TableStyle(cmds))
    return t


def fig_img(path: Path, width=15.8 * cm):
    name = path.name
    if "refine" in name:
        ratio = 0.90
    elif any(key in name for key in ("beam", "s21", "roi_steps", "filtered")):
        ratio = 0.38
    else:
        ratio = 0.55
    return Image(str(path), width=width, height=width * ratio)


def parse_s2p(path: Path):
    freqs, s21 = [], []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header_seen = False
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                header_seen = True
                continue
            if not header_seen:
                continue
            nums = [float(t) for t in line.split("!", 1)[0].split()]
            freqs.append(nums[0] * 1e9)
            s21.append(nums[3] + 1j * nums[4])
    return np.asarray(freqs), np.asarray(s21, dtype=complex)


def synthetic_array(n_freq=201, n_ant=8, seed=1):
    rng = np.random.default_rng(seed)
    f = np.linspace(1e9, 9e9, n_freq)
    center, bw = 5e9, 3e9
    base = 0.6 * np.exp(-((f - center) ** 2) / (2 * bw**2))
    traces = np.zeros((n_freq, n_ant), dtype=complex)
    for i in range(n_ant):
        amp = 1.0 + 0.05 * np.sin(2 * np.pi * i / n_ant)
        bump = 0.03 * np.exp(-((f - (5.5e9 + 0.05e9 * i)) ** 2) / (2 * (0.15e9) ** 2))
        phase = -f / 1e9 * 0.8 + 0.3 * i
        noise = rng.normal(0, 0.01, n_freq) + 1j * rng.normal(0, 0.01, n_freq)
        traces[:, i] = (amp * base + bump) * np.exp(1j * phase) + noise
    return f, traces


def circular_array(n, radius=0.08):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=-1)


def frequency_to_time(s, f):
    return np.fft.ifft(s, axis=0)


def gather_delayed(time_signals, frequencies, ants, points, c=3e8):
    n_time, n_tr = time_signals.shape
    dist = np.linalg.norm(ants[:, None, :] - points[None, :, :], axis=-1)
    tau = 2.0 * dist / c
    df = frequencies[1] - frequencies[0]
    dt = 1.0 / (df * n_time)
    delayed = np.zeros((n_tr, points.shape[0]), dtype=np.complex128)
    for i in range(n_tr):
        idx = np.round(tau[i] / dt).astype(int)
        ok = (idx >= 0) & (idx < n_time)
        delayed[i, ok] = time_signals[idx[ok], i]
    return delayed


def coherence_factor(delayed):
    n = max(delayed.shape[0], 1)
    return np.abs(np.sum(delayed, axis=0)) ** 2 / (n * np.sum(np.abs(delayed) ** 2, axis=0) + EPS)


def das_image(s, f, ants, gx, gy):
    t = frequency_to_time(s, f)
    pts = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    d = gather_delayed(t, f, ants, pts)
    img = np.sum(np.abs(d), axis=0) * coherence_factor(d) ** 0.55
    return img.reshape(len(gy), len(gx))


def dmas_image(s, f, ants, gx, gy):
    t = frequency_to_time(s, f)
    pts = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    d = gather_delayed(t, f, ants, pts)
    img = np.zeros(pts.shape[0])
    for i in range(d.shape[0]):
        for j in range(i + 1, d.shape[0]):
            prod = d[i] * np.conj(d[j])
            mag = np.abs(prod) + 1e-9
            img += np.real(np.sign(prod) * np.power(mag, 0.30))
    return img.reshape(len(gy), len(gx))


def detect_primary_roi(image, threshold=0.7):
    arr = np.asarray(image, dtype=float)
    norm = (arr - arr.min()) / (arr.max() - arr.min() + EPS)
    sm = gaussian_filter(norm, 1.0)
    binary = binary_opening(sm >= threshold, structure=np.ones((3, 3)))
    labeled, n = label(binary)
    if n == 0:
        y, x = np.unravel_index(int(np.argmax(sm)), sm.shape)
        box = (max(0, x - 3), max(0, y - 3), min(sm.shape[1], x + 4), min(sm.shape[0], y + 4))
        return box, sm, binary, [(box, 1.0)]
    scored = []
    objects = find_objects(labeled)
    for cid, slc in enumerate(objects, 1):
        if slc is None:
            continue
        mask = labeled[slc] == cid
        area = int(np.sum(mask))
        if area < 6:
            continue
        vals = sm[slc][mask]
        peak, mean = float(np.max(vals)), float(np.mean(vals))
        yh, xw = slc
        bbox_a = max(1, (yh.stop - yh.start) * (xw.stop - xw.start))
        compact = area / (bbox_a + EPS)
        ar = area / (sm.size + EPS)
        edge = 0.65 if (xw.start <= 0 or yh.start <= 0 or xw.stop >= sm.shape[1] or yh.stop >= sm.shape[0]) else 1.0
        score = (0.65 * peak + 0.35 * mean) * compact * (1.0 / (1.0 + 10 * ar)) * edge
        box = (xw.start, yh.start, xw.stop, yh.stop)
        scored.append((score, box, area, compact, peak, mean, ar, edge))
    scored.sort(reverse=True)
    best = scored[0][1] if scored else (0, 0, 8, 8)
    return best, sm, binary, scored


def tumor_features(image, box):
    arr = np.asarray(image, float)
    ny, nx = arr.shape
    x0, y0, x1, y1 = box
    mask = np.zeros(arr.shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    inside = arr[mask]
    outside = arr[~mask]
    peak = float(np.max(inside))
    bg = float(np.mean(outside))
    local = peak / (abs(bg) + EPS)
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    rcx, rcy = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
    r_norm = float(np.hypot(rcx - cx, rcy - cy) / (np.hypot(cx, cy) + EPS))
    area = float(np.sum(mask))
    area_ratio = area / (ny * nx + EPS)
    yy, xx = np.ogrid[:ny, :nx]
    radius = np.hypot(xx - cx, yy - cy)
    r0 = float(np.hypot(rcx - cx, rcy - cy))
    other = (np.abs(radius - r0) <= 1.5) & ~mask
    ring = peak / (abs(float(np.median(arr[other]))) + EPS) if np.sum(other) >= 8 else 1.0
    size_score = 1.0 / (1.0 + (area_ratio / 0.08) ** 2)
    contrast_score = float(np.tanh(max(local - 1.0, 0.0) / 2.0))
    ring_score = float(np.tanh(max(ring - 1.15, 0.0) / 0.35))
    location_score = 0.45 + 0.55 * r_norm
    suspicion = float(np.clip(0.38 * contrast_score + 0.22 * size_score + 0.15 * location_score + 0.25 * ring_score, 0, 1))
    penalty = 1.0
    if x0 <= 0 or y0 <= 0 or x1 >= nx or y1 >= ny:
        penalty *= 0.85
    if area_ratio > 0.35:
        penalty *= 0.65
    if ring < 1.20:
        penalty *= 0.55
    conf = float(np.clip(suspicion * penalty, 0, 1))
    is_yes = conf >= 0.45
    if ring < 1.40:
        is_yes = False
    return {
        "peak": peak,
        "local": local,
        "r_norm": r_norm,
        "area": area,
        "area_ratio": area_ratio,
        "ring": ring,
        "suspicion": suspicion,
        "conf": conf,
        "yes": is_yes,
        "centroid_px": (rcx, rcy),
    }


def quality(image):
    img = np.asarray(image, float)
    mean, std = float(np.mean(img)), float(np.std(img))
    peak = float(np.max(img))
    snr = mean / (std + EPS)
    scr = peak / (float(np.mean(np.abs(img - mean))) + EPS)
    contrast = (peak - float(np.min(img))) / (peak + float(np.min(img)) + EPS)
    return {"snr": snr, "scr": scr, "contrast": contrast, "peak": peak, "mean": mean}


def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    f_s2p, s21 = parse_s2p(S2P)
    ghz = f_s2p / 1e9
    mag_db = 20 * np.log10(np.abs(s21) + 1e-30)
    phase = np.unwrap(np.angle(s21))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    axes[0].plot(ghz, mag_db, color="#1f6f8b")
    axes[0].set_title("|S21| (dB)")
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("dB")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ghz, np.degrees(phase), color="#1b365d")
    axes[1].set_title("Unwrapped phase of S21")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("degrees")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(ghz, s21.real, label="real", color="#1f6f8b")
    axes[2].plot(ghz, s21.imag, label="imag", color="#c45c26")
    axes[2].set_title("S21 real and imaginary")
    axes[2].set_xlabel("Frequency (GHz)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    p_s21 = FIG / "sample_s21_traces.png"
    fig.savefig(p_s21, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # mild Savitzky-Golay style moving average on Re/Im for "after filter" plot
    win = np.ones(7) / 7
    s21_f = np.convolve(s21.real, win, mode="same") + 1j * np.convolve(s21.imag, win, mode="same")
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    ax.plot(ghz, mag_db, alpha=0.45, label="raw |S21| dB")
    ax.plot(ghz, 20 * np.log10(np.abs(s21_f) + 1e-30), label="after mild Re/Im smoothing")
    ax.set_title("Module 2 on sample_touchstone.s2p — raw vs filtered |S21|")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("dB")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_filt = FIG / "sample_s21_filtered.png"
    fig.savefig(p_filt, dpi=140, bbox_inches="tight")
    plt.close(fig)

    f, traces = synthetic_array()
    ants = circular_array(traces.shape[1], 0.08)
    gx = np.linspace(-0.05, 0.05, 64)
    gy = np.linspace(-0.05, 0.05, 64)
    images = {
        "DAS": das_image(traces, f, ants, gx, gy),
        "DMAS": dmas_image(traces, f, ants, gx, gy),
    }
    images["DMAS-D4"] = np.sign(images["DMAS"]) * np.abs(images["DMAS"]) ** 0.25
    qmap = {k: quality(v) for k, v in images.items()}
    # quality selection like the project
    snrs = [qmap[k]["snr"] for k in images]
    scrs = [qmap[k]["scr"] for k in images]
    cons = [qmap[k]["contrast"] for k in images]

    def nrm(v, vs):
        return (v - min(vs)) / (max(vs) - min(vs) + EPS)

    scores = {}
    for name, q in qmap.items():
        scores[name] = 0.35 * nrm(q["snr"], snrs) + 0.35 * nrm(q["scr"], scrs) + 0.20 * nrm(q["contrast"], cons) + 0.10
    selected = max(scores, key=scores.get)
    img = images[selected]
    box, sm, binary, scored = detect_primary_roi(img)
    feats = tumor_features(img, box)

    extent = [-5, 5, -5, 5]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    for ax, name in zip(axes, ("DAS", "DMAS", "DMAS-D4")):
        im = ax.imshow(images[name], origin="lower", extent=extent, cmap="inferno")
        ax.set_title(f"{name}\nscore={scores[name]:.3f}")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Module 3 sample output — 8-antenna synthetic UWB array (same family as sample_matlab.mat)")
    fig.tight_layout()
    p_bf = FIG / "sample_beamformers.png"
    fig.savefig(p_bf, dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    im0 = axes[0].imshow(sm, origin="lower", extent=extent, cmap="inferno")
    axes[0].set_title("Normalized + Gaussian smoothed")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[1].imshow(binary, origin="lower", extent=extent, cmap="gray")
    axes[1].set_title("Binary map after T = 0.70")
    im2 = axes[2].imshow(img, origin="lower", extent=extent, cmap="inferno")
    x0, y0, x1, y1 = box
    left = -5 + x0 / 63 * 10
    bottom = -5 + y0 / 63 * 10
    w = (x1 - x0) / 63 * 10
    h = (y1 - y0) / 63 * 10
    axes[2].add_patch(Rectangle((left, bottom), w, h, fill=False, edgecolor="cyan", linewidth=1.6))
    axes[2].set_title(f"Selected image ({selected}) + ROI box")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
    fig.suptitle("ROI detection on the selected reconstruction")
    fig.tight_layout()
    p_roi = FIG / "sample_roi_steps.png"
    fig.savefig(p_roi, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # refine zoom = crop
    pad = 2
    x0, y0, x1, y1 = box
    xs0, ys0 = max(0, x0 - pad), max(0, y0 - pad)
    xs1, ys1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
    crop = img[ys0:ys1, xs0:xs1]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(crop, origin="lower", cmap="inferno")
    ax.set_title("High-resolution look inside the ROI box")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p_ref = FIG / "sample_roi_refine.png"
    fig.savefig(p_ref, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "s21": p_s21,
        "filt": p_filt,
        "bf": p_bf,
        "roi": p_roi,
        "refine": p_ref,
        "s21_start": (float(s21[0].real), float(s21[0].imag), float(mag_db[0])),
        "s21_mid": (float(s21[len(s21)//2].real), float(s21[len(s21)//2].imag)),
        "n_freq": int(len(s21)),
        "df_mhz": float((f_s2p[1] - f_s2p[0]) / 1e6),
        "selected": selected,
        "scores": scores,
        "qmap": qmap,
        "box": box,
        "scored": scored,
        "feats": feats,
    }


def story(st, demo):
    s = []
    s.append(Spacer(1, 0.8 * cm))
    s.append(P("Report C — Project Working Report", st, "cover"))
    s.append(P("Every tiny step from loading a file to the YES / NO tumor-candidate decision, with sample outputs.", st, "sub"))
    s.append(P(
        "This report walks the software the way a beginner should learn it: one module at a time, and inside each module "
        "one small action at a time. For every action we say the <b>input</b>, the <b>technique</b>, and the <b>output</b>. "
        "Formulas are the ones in the code, not textbook-only versions. Pictures come from the in-repo sample files.",
        st,
    ))
    s.append(P("Important: the reconstructed image and the YES/NO label are research estimates. They are not a medical diagnosis.", st, "note"))

    s.append(P("How to read this report", st, "h1"))
    s.extend(bullets([
        "The project has three teaching modules: (1) load data, (2) clean the signal, (3) make an image and find a region.",
        "Automation later repeats the same chain on many files. Auto Tweak only searches geometry on one already-loaded scan.",
        "sample_touchstone.s2p is used for Module 1–2 pictures. Reconstruction pictures use the 8-antenna synthetic family of sample_matlab.mat, because a single S21 trace cannot form a useful multi-antenna image.",
        "When a number is compared to a limit, that limit is written next to the step.",
    ], st))

    s.append(P("0. The whole workflow in one place", st, "h1"))
    s.append(P(
        "Load file\n"
        "    ↓  Module 1\n"
        "MicrowaveDataset  (frequencies + complex S)\n"
        "    ↓  Module 2\n"
        "Cleaned traces  (or unchanged BMID _adi data)\n"
        "    ↓  IFFT\n"
        "Time-domain traces\n"
        "    ↓  DAS / DMAS / DMAS-D4\n"
        "Three images\n"
        "    ↓  beamformer selection\n"
        "One selected image\n"
        "    ↓  ROI detection\n"
        "One or more boxes\n"
        "    ↓  high-resolution rebuild of the box\n"
        "Refined local image\n"
        "    ↓  characterization + suspicion\n"
        "YES / NO tumor-candidate + confidence",
        st,
        "formula",
    ))
    s.append(P(
        "The purpose changes at each arrow. First we only want a valid matrix. Then we want a cleaner matrix. "
        "Then we want a picture. Then we stop asking “which algorithm looks nicest?” and start asking "
        "“where is the most relevant region in the chosen picture?” Then we describe that region with numbers. "
        "Only after that do we decide whether it looks tumor-like under the project’s rules.",
        st,
    ))

    # MODULE 1
    s.append(P("Module 1 — Data acquisition (loading)", st, "h1"))
    s.append(P(
        "Module 1 does not make an image. It only turns a file on disk into one common object called MicrowaveDataset.",
        st,
    ))

    s.append(P("Step 1. The user (or automation) chooses a path", st, "h3"))
    s.append(P(
        "<b>Input:</b> a file path such as datasets/sample_touchstone.s2p or a UM-BMID fd_data_*.mat plus a scan index.<br/>"
        "<b>Technique:</b> look at the extension. .mat goes to the MATLAB loader. .s1p/.s2p/.s4p/.s8p go to the Touchstone loader. "
        "Anything else raises UnsupportedFileFormatError.<br/>"
        "<b>Output:</b> a decision about which parser to run. Nothing is reconstructed yet.",
        st,
        "io",
    ))

    s.append(P("Step 2. If the file is Touchstone .s2p", st, "h3"))
    s.append(P(
        "The parser reads comment lines (starting with !) and the option line (starting with #). "
        "Our sample option line is:",
        st,
    ))
    s.append(P("# GHz S RI R 50.0", st, "formula"))
    s.append(P(
        "That sentence means: frequencies are in GHz, the numbers are S-parameters, the pair format is Real/Imaginary, "
        "and the reference impedance is 50 ohm. Then each data row is one frequency followed by eight numbers: "
        "Re(S11) Im(S11) Re(S21) Im(S21) Re(S12) Im(S12) Re(S22) Im(S22).",
        st,
    ))
    s.append(P(
        "S = Re + j Im<br/>"
        "|S| = sqrt(Re² + Im²)<br/>"
        "phase = atan2(Im, Re)<br/>"
        "|S|_dB = 20 log10(|S|)<br/>"
        "If the file had been DB format: |S| = 10^(dB/20), then S = |S|(cos θ + j sin θ).",
        st,
        "formula",
    ))
    s.append(P(
        "<b>Input:</b> the text file.<br/>"
        "<b>Technique:</b> scikit-rf Network plus a project S21 helper that rebuilds S21 from the eight-number rows.<br/>"
        f"<b>Output:</b> frequencies in Hz (length {demo['n_freq']}), S-matrix shape (n_freq, 2, 2), and metadata. "
        f"Δf = {demo['df_mhz']:.2f} MHz. First S21 = {demo['s21_start'][0]:+.6f} {demo['s21_start'][1]:+.6f}j.",
        st,
        "io",
    ))

    s.append(P("Step 3. If the file is a general MATLAB .mat", st, "h3"))
    s.append(P(
        "The loader first asks: is this HDF5 (MATLAB v7.3) or a legacy MAT? Then it lists numeric variables. "
        "It looks for a frequency vector by name (freq, frequency, …) or by “real, increasing, looks like RF”. "
        "If the largest frequency is less than 1000, it treats the vector as GHz and multiplies by 1e9. "
        "It then picks the S array (name hints such as s_param / s21, plus “has the frequency length in its shape”). "
        "Axis 0 is forced to be frequency. A 2-D array stays (n_freq, n_traces). A square 3-D array stays (n_freq, n_ports, n_ports).",
        st,
    ))
    s.append(P(
        "<b>Input:</b> .mat variables.<br/>"
        "<b>Technique:</b> name heuristics + documented fallbacks (fd_data_s21 → 1–8 GHz, 1001 points).<br/>"
        "<b>Output:</b> MicrowaveDataset. If no numbers exist, it tells you the file is probably metadata-only.",
        st,
        "io",
    ))

    s.append(P("Step 4. If the file is a UM-BMID cube", st, "h3"))
    s.append(P(
        "A BMID measurement file is a 3-D complex cube. The code finds an array that contains the length 1001 "
        "and reorders it to:",
        st,
    ))
    s.append(P("cube[scan, frequency, antenna]  ∈ complex<br/>one scan S = cube[k, :, :]   shape (1001, n_antennas)<br/>f = linspace(1 GHz, 8 GHz, 1001)", st, "formula"))
    s.append(P(
        "If you do not give scan_index, loading stops with ScanSelectionRequiredError. "
        "That is on purpose: the cube is many experiments in one file. "
        "Companion md_list_*.mat supplies labels: tum_diam or tum_rad, tum_x, tum_y, birads, tum_shape, ant_rad, phant_id. "
        "A scan is called tumor if diameter &gt; 0 or a non-zero (x, y) exists. Radius 18 cm and FOV ±6 cm are stored in metadata.",
        st,
    ))
    s.append(P(
        "<b>Input:</b> fd_data_*.mat + optional md_list_*.mat + scan index.<br/>"
        "<b>Technique:</b> slice one scan; attach labels in metres.<br/>"
        "<b>Output:</b> one MicrowaveDataset for that scan only.",
        st,
        "io",
    ))

    s.append(P("Step 5. Validation before anyone plots", st, "h3"))
    s.append(P(
        "Frequencies and S-values must be finite. Empty files fail. Wrong Touchstone headers fail. "
        "The GUI shows the summary panel: file name, type, number of samples, number of frequencies, range, ports, size, variable names. "
        "Number of samples means how many complex entries are stored, not how many tumors exist.",
        st,
    ))

    s.append(P("Sample Module 1 output — sample_touchstone.s2p", st, "h2"))
    s.append(fig_img(demo["s21"], 16.2 * cm))
    s.append(P(
        "Figure. Left: |S21| in dB across 1–9 GHz. Middle: unwrapped phase (the delay story). "
        "Right: the actual real and imaginary parts that later stages filter. "
        f"At 1 GHz, |S21| ≈ {demo['s21_start'][2]:.2f} dB. This is a synthetic teaching file, so the curve is a smooth resonance plus noise, not a hospital measurement.",
        st,
        "cap",
    ))
    s.append(P(
        "<b>How to interpret this figure.</b> A falling then rising |S21| means the two-port sample has a frequency-dependent transmission. "
        "A steadily changing unwrapped phase means delay. If real and imaginary were both exactly zero, the file would be empty of energy and later reconstruction would fail the energy gate. "
        "Module 1’s job ends when these arrays sit in memory. It does not decide if a tumor exists.",
        st,
    ))

    # MODULE 2
    s.append(P("Module 2 — Signal preprocessing", st, "h1"))
    s.append(P(
        "The purpose now changes. We already have a valid matrix. We ask: can we make the traces cleaner without inventing a fake target? "
        "There are two tracks.",
        st,
    ))
    s.append(table(
        [
            ["Track", "When", "What happens"],
            [".s2p UG track", "file ends with .s2p", "Extract S21 only, then Week 1–3 cleaning"],
            ["General MATLAB", "other .mat / multi-trace", "filter → calibrate → normalize → background → artifacts"],
            ["BMID pass-through", "dataset_family = UM-BMID", "every method = none, Week 3 off"],
        ],
        [4.2 * cm, 5.4 * cm, 6.6 * cm],
        st,
    ))

    s.append(P("Step 6. Router", st, "h3"))
    s.append(P(
        "<b>Input:</b> MicrowaveDataset + PreprocessingConfig (GUI preset or automation defaults).<br/>"
        "<b>Technique:</b> if file_type starts with Touchstone and path ends with .s2p, call process_touchstone_s21_dataset. Else run the five-stage general loop.<br/>"
        "<b>Output:</b> PreprocessingResult: original dataset, processed dataset, stage snapshots, quality report, optional Week 3 object.",
        st,
        "io",
    ))

    s.append(P("Step 7. Extract S21 only (.s2p track)", st, "h3"))
    s.append(P(
        "A 2×2 matrix has four stories. The UG brief uses transmission S21. After this step the working array is 1-D (or n_freq × 1), not 2×2. "
        "S11, S12, S22 stay in the original dataset if you need them, but reconstruction later sees only S21.",
        st,
    ))
    s.append(P(
        "<b>Input:</b> (n_freq, 2, 2) complex S.<br/>"
        "<b>Technique:</b> take index [:, 1, 0] in memory, which is S21 from the Touchstone row order.<br/>"
        "<b>Output:</b> raw_s21, length n_freq.",
        st,
        "io",
    ))

    s.append(P("Step 8. Sort and put frequencies on a uniform grid", st, "h3"))
    s.append(P(
        "IFFT later requires a constant Δf. The code sorts frequency, then interpolates real and imaginary separately onto a uniform axis. "
        "If the file was already uniform (our sample Δf = 40 MHz), interpolation barely changes the numbers, but the flag still records that a uniform grid is guaranteed.",
        st,
    ))
    s.append(P(
        "<b>Input:</b> possibly uneven (f, S21).<br/>"
        "<b>Technique:</b> interpolate Re(S21) and Im(S21).<br/>"
        "<b>Output:</b> uniform_frequencies_hz and S21 on that grid.",
        st,
        "io",
    ))

    s.append(P("Step 9. Magnitude, wrapped phase, unwrap", st, "h3"))
    s.append(P(
        "Wrapped phase jumps by about ±180° when the true delay crosses a branch cut. Unwrap adds or subtracts 360° so the delay curve is continuous. "
        "The report stores how many unwrap adjustments happened. This does not change |S21|; it only repairs the angle story.",
        st,
    ))

    s.append(P("Step 10. Spike repair", st, "h3"))
    s.append(P(
        "Default method is Hampel. A sample that is far from its local median is replaced. Alternatives are median or local replacement. "
        "This is for isolated glitches, not for removing a real resonance.",
        st,
    ))
    s.append(P(
        "<b>Input:</b> corrected complex S21.<br/>"
        "<b>Technique:</b> Hampel / median / local on the sweep.<br/>"
        "<b>Output:</b> corrected_s21 and a count of repaired samples.",
        st,
        "io",
    ))

    s.append(P("Step 11. Optional average and optional reference subtract", st, "h3"))
    s.append(P(
        "If the user supplies repeated sweeps, they are averaged. If a reference measurement is supplied, it is subtracted. "
        "On a single teaching .s2p both are usually skipped, and the validation report says Averaging = No, Reference = No.",
        st,
    ))

    s.append(P("Step 12. Mild filter on real and imaginary", st, "h3"))
    s.append(P(
        "Default for the .s2p / automation non-BMID path is Savitzky–Golay, window_length = 7, polyorder = 3. "
        "The filter is applied to Re and Im separately so phase is not destroyed by filtering |S| alone. "
        "NRMSE = ||filtered − before|| / ||before|| tells how much the trace moved.",
        st,
    ))
    s.append(P(
        "SNR used here (signal vs sweep noise, not image SNR):<br/>"
        "smooth |S| with a 5-sample moving average<br/>"
        "noise = |S| − smooth<br/>"
        "SNR_dB = 10 log10(P_smooth / P_noise)<br/>"
        "dynamic_range_dB = 20 log10(max|S| / min|S|)",
        st,
        "formula",
    ))

    s.append(P("Step 13. Hamming copy (kept aside)", st, "h3"))
    s.append(P(
        "w[k] = 0.54 − 0.46 cos(2π k / (N−1))<br/>"
        "S_hamming(f) = S21_filtered(f) · w[k]",
        st,
        "formula",
    ))
    s.append(P(
        "The Hamming copy reduces IFFT sidelobes later. The GUI still reconstructs from filtered S21(f) by default. "
        "The Hamming / Week-3 time arrays are stored in metadata and in the Module 3 handoff file. That gap is documented in the README.",
        st,
    ))

    s.append(P("Step 14. Week 3 time-domain (when enabled)", st, "h3"))
    s.append(P(
        "Δf = mean consecutive step<br/>"
        "Δt = 1 / (N Δf)<br/>"
        "t[n] = n Δt<br/>"
        "s(t) = IFFT(S_hamming(f))<br/>"
        "If a reference exists: subtract the matched Hamming IFFT in time.<br/>"
        "If ≥ 2 channels: μ_G[n] = mean over channels; s_clean = s − μ_G<br/>"
        "α = max |s|;   s ← s / α   (global normalize)",
        st,
        "formula",
    ))
    s.append(P(
        "On sample_touchstone.s2p there is only one channel, so group clutter is skipped and a note is stored. "
        "Automation for .s2p turns Week 3 on. Automation for BMID turns it off.",
        st,
    ))

    s.append(P("Step 15. General MATLAB stages (non-.s2p)", st, "h3"))
    s.extend(bullets([
        "Noise filtering: savgol / moving average / Butterworth / median / gaussian, always on Re and Im along frequency.",
        "Calibration: self or an external reference array.",
        "Normalization: e.g. divide by the max magnitude.",
        "Background subtraction: optional, only if do_background_subtraction is true.",
        "Artifact suppression: hybrid or other methods in artifact_suppression.py.",
        "Each stage writes a snapshot into stage_outputs so the GUI Stages tab can show before/after.",
    ], st))
    s.append(P(
        "BMID pass-through sets filter=calibration=normalization=artifact=none, background off, Week 3 off. "
        "The output dataset is a copy of the input. That is correct for already-cleaned _adi cubes. "
        "Cleaning them again can erase the tumor focus.",
        st,
    ))

    s.append(P("Step 16. What Module 2 hands to Module 3", st, "h3"))
    s.append(P(
        "processed_dataset.s_parameters is 2-D (n_freq, n_traces). For .s2p that is (201, 1) containing only S21. "
        "For BMID it is (1001, n_antennas). Optional handoff export writes .mat/.csv/.json plus circular Tx/Rx coordinates.",
        st,
    ))

    s.append(P("Sample Module 2 output", st, "h2"))
    s.append(fig_img(demo["filt"], 16.2 * cm))
    s.append(P(
        "Figure. Raw |S21| (light) versus the same sweep after a mild 7-sample smoother on real and imaginary parts. "
        "The overall shape is the same. Sharp jitter is reduced. That is what we want: less noise, same resonance.",
        st,
        "cap",
    ))
    s.append(P(
        "<b>How to interpret this figure.</b> If the dark curve had flattened the dip completely, the filter would be too strong. "
        "If the two curves were identical down to every wiggle, the filter did almost nothing (that is also fine for an already-clean file). "
        "A large NRMSE with a destroyed shape means you should not reconstruct yet — go back to filter settings.",
        st,
    ))

    # MODULE 3
    s.append(P("Module 3 — Reconstruction", st, "h1"))
    s.append(P(
        "The purpose now changes again. We stop looking at a line versus frequency. We ask: if each antenna heard a delayed echo, "
        "where in the x–y plane would those echoes line up?",
        st,
    ))

    s.append(P("Step 17. Read geometry from metadata or defaults", st, "h3"))
    s.append(P(
        "x = linspace(x_min, x_max, n_x)<br/>"
        "y = linspace(y_min, y_max, n_y)<br/>"
        "default n_x = n_y = 64<br/>"
        "antenna i:  (r cos θ_i, r sin θ_i)<br/>"
        "360° ring: θ = linspace(0, 2π, N, endpoint=False)<br/>"
        "355° BMID arc: θ = linspace(0, 355°, N, endpoint=True)<br/>"
        "optional: angle offset, clockwise, flip x, flip y<br/>"
        "optional phase-delay radius: r_eff = 0.97(r − 0.106) + 0.148 m",
        st,
        "formula",
    ))
    s.append(P(
        "<b>Input:</b> processed traces + ReconstructionConfig.<br/>"
        "<b>Technique:</b> infer_reconstruction_config reads antenna_radius_m, wave_speed_m_per_s, FOV spans. "
        "Missing values become 0.08 m, 3e8 m/s, ±5 cm. BMID uses 0.18 m and ±6 cm. Phase-delay starts OFF.<br/>"
        "<b>Output:</b> antenna coordinates (N × 2 metres) and a pixel grid.",
        st,
        "io",
    ))
    s.append(P(
        "Wave speed c is the conversion from distance to time. Wrong c is like focusing a camera at the wrong distance. "
        "Auto Tweak can later try 2.1e8 and 1.8e8 m/s. Batch automation does not, so many files stay comparable.",
        st,
    ))

    s.append(P("Step 18. IFFT of every trace", st, "h3"))
    s.append(P(
        "s(t) = IFFT(S(f)) along frequency<br/>"
        "N_out = n_freq + zero_padding<br/>"
        "t[n] = n / (N Δf)<br/>"
        "Requirement: Δf nearly constant (relative tolerance 1e-3), at least 2 samples.",
        st,
        "formula",
    ))
    s.append(P(
        "<b>Input:</b> complex S, shape (n_freq, n_traces).<br/>"
        "<b>Technique:</b> NumPy ifft, optional zero padding for finer time bins.<br/>"
        "<b>Output:</b> complex time_signals, shape (N_out, n_traces). This is the shared starting point for DAS, DMAS and DMAS-D4.",
        st,
        "io",
    ))
    s.append(P(
        "For sample_touchstone.s2p: N = 201, Δf = 40 MHz, Δt = 1/(201×40e6) ≈ 0.124 ns. "
        "Two-way delay from an 8 cm radius antenna to the origin is 2×0.08/3e8 ≈ 0.533 ns ≈ 4 samples. "
        "With only one trace, later pairing in DMAS has no partner. That is why the reconstruction pictures below use the 8-antenna synthetic array.",
        st,
    ))

    s.append(P("Step 19. Delay-and-Sum (DAS)", st, "h3"))
    s.append(P(
        "For every pixel r and every antenna a_i:<br/>"
        "d_i = ||a_i − r||<br/>"
        "τ_i = 2 d_i / c     (two-way time)<br/>"
        "n = round(τ_i / Δt)<br/>"
        "if n is outside the trace, that antenna adds 0",
        st,
        "formula",
    ))
    s.append(P(
        "I_plain(r) = Σ_i |s_i(τ_i)|<br/>"
        "CF(r) = |Σ_i s_i|² / (N Σ_i |s_i|²)<br/>"
        "I_DAS(r) = I_plain(r) · CF(r)^γ    with γ = 0.55",
        st,
        "formula",
    ))
    s.append(P(
        "γ = 0 is textbook DAS (just add envelopes). γ = 0.55 is the cohort average from the beamformer sweep: "
        "it pulls the top spot closer to labeled tumors than γ = 0, while γ = 0.75 is almost the same and γ ≥ 0.80 starts healthy false positives. "
        "When antennas disagree, CF is small and that pixel is pushed down.",
        st,
    ))
    s.append(P(
        "<b>Input:</b> time traces, antenna positions, grid, c, γ.<br/>"
        "<b>Technique:</b> nearest-sample delay, then optional coherence weight.<br/>"
        "<b>Output:</b> one real image of shape (n_y, n_x).",
        st,
        "io",
    ))

    s.append(P("Step 20. Delay-Multiply-and-Sum (DMAS)", st, "h3"))
    s.append(P(
        "Same delayed samples s_i. For every pair i &lt; j:<br/>"
        "p_ij = s_i · conjugate(s_j)<br/>"
        "signed_power(z,p) = Re{ sign(z) · (|z|+ε)^p }<br/>"
        "p = 0.30 (textbook 0.50), ε = 1e-9<br/>"
        "I_DMAS = Σ_{i&lt;j} signed_power(p_ij) · CF^{0}",
        st,
        "formula",
    ))
    s.append(P(
        "Multiplying two antennas keeps energy that is consistent in both. Random clutter is less likely to survive every pair. "
        "That is why DMAS is often sharper than DAS, and also why it is slower.",
        st,
    ))

    s.append(P("Step 21. DMAS-D4", st, "h3"))
    s.append(P(
        "I_D4 = sign(I_DMAS) · |I_DMAS|^{0.25}",
        st,
        "formula",
    ))
    s.append(P(
        "Exponent 1/4 is the textbook map and, on the cohort sweep, the average-best. "
        "0.10 through 0.80 were within 0.01 of that peak. 0.00 scattered the spot. 0.85 and above raised healthy false positives. "
        "D4 is not a new delay model. It is a contrast map after DMAS.",
        st,
    ))

    s.append(P("Step 22. Quality numbers on each image", st, "h3"))
    s.append(P(
        "SNR = mean(I) / std(I)<br/>"
        "SCR = max(I) / mean(|I − mean(I)|)<br/>"
        "CCR = (mean(top 5% pixels) − mean(rest)) / std(rest)<br/>"
        "contrast = (max − min) / (max + min)<br/>"
        "FWHM_px = average of half-max widths on the peak row and the peak column",
        st,
        "formula",
    ))

    s.append(P("Step 23. Choose one beamformer", st, "h3"))
    s.append(P(
        "We are still asking “which picture should we trust?” not yet “where is the region?”",
        st,
    ))
    s.append(P(
        "Quality mode (default when there is no tumor label):<br/>"
        "normalize each metric across DAS/DMAS/DMAS-D4 to 0–1<br/>"
        "score = 0.35 SNR_n + 0.35 SCR_n + 0.20 contrast_n + 0.10 (1 − time_n)",
        st,
        "formula",
    ))
    s.extend(bullets([
        "prefer_dmas_d4: always pick DMAS-D4; still print the other scores.",
        "force_das / force_dmas / force_dmas_d4: user override.",
        "tumor_gt: pick the image whose ROI centroid is closest to the labeled (x, y). Used by automation when metadata has a tumor.",
    ], st))
    qrows = [["Image", "SNR", "SCR", "contrast", "quality score"]]
    for name in ("DAS", "DMAS", "DMAS-D4"):
        q = demo["qmap"][name]
        qrows.append([name, f"{q['snr']:.3f}", f"{q['scr']:.3f}", f"{q['contrast']:.3f}", f"{demo['scores'][name]:.3f}"])
    s.append(table(qrows, [3.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.4 * cm], st))
    s.append(P(
        f"On this sample run the winner is <b>{demo['selected']}</b> because it had the highest weighted score. "
        "These scores are relative to each other on the same scan. A high score does not mean “tumor present”.",
        st,
    ))

    s.append(P("Sample Module 3 output", st, "h2"))
    s.append(fig_img(demo["bf"], 16.2 * cm))
    s.append(P(
        "Figure. Three reconstructions of the same 8-antenna synthetic dataset, 64×64 pixels, FOV 10 cm × 10 cm, r = 8 cm, c = 3e8. "
        "Color is image intensity (inferno: dark = low, yellow = high).",
        st,
        "cap",
    ))
    s.append(P(
        "<b>How to interpret this figure.</b> A useful reconstruction shows a compact brighter patch, not a full bright ring and not a uniformly noisy field. "
        "DAS is usually the smoothest. DMAS and DMAS-D4 often look more peaked. If all three images are a circular ring at the antenna radius, "
        "the focus is sitting on clutter, not on a compact interior target — later the ring-excess test is designed to catch that. "
        "This sample is synthetic teaching data, so the “hotspot” is a made-up resonance, not a labeled tumor.",
        st,
    ))

    # ROI - user's style
    s.append(P("ROI detection starts", st, "h1"))
    s.append(P(
        "ROI = Region of Interest. The purpose now changes. We are no longer asking “which beamformer is best?” "
        "We are asking “where in the selected image is the most relevant / suspicious region?”",
        st,
    ))

    s.append(P("Step 24. Image normalization", st, "h3"))
    s.append(P(
        "I_n = (I − I_min) / (I_max − I_min + ε)",
        st,
        "formula",
    ))
    s.append(P(
        "This stretches intensity so the darkest pixel is about 0 and the brightest is about 1. "
        "Threshold 0.70 then means “70% of the way from darkest to brightest on this image”, not an absolute physical unit.",
        st,
    ))
    s.append(P(
        "<b>Input:</b> selected real image I.<br/>"
        "<b>Technique:</b> min–max scale, ε = 1e-12.<br/>"
        "<b>Output:</b> I_n in [0, 1].",
        st,
        "io",
    ))

    s.append(P("Step 25. Gaussian smoothing", st, "h3"))
    s.append(P(
        "I_s = Gaussian(I_n, σ)    default σ = 1.0 pixel",
        st,
        "formula",
    ))
    s.append(P(
        "Reconstructed image\n        ↓\nGaussian smoothing\n        ↓\nReduced tiny fluctuations",
        st,
        "formula",
    ))
    s.append(P(
        "Without this step, a single noisy pixel can become its own “region”. Tight-peak mode may reduce σ further, but the default is 1.0.",
        st,
    ))

    s.append(P("Step 26. Thresholding", st, "h3"))
    s.append(P(
        "B = 1{ I_s ≥ T }    default T = 0.70",
        st,
        "formula",
    ))
    s.append(P(
        "Meaning: if smoothed intensity ≥ threshold → 1, otherwise → 0. Example (not from the sample image):",
        st,
    ))
    s.append(P(
        "Intensity:\n0.2  0.3  0.8\n0.1  0.9  0.7\n0.2  0.6  0.3\n\nT = 0.6 gives\n0  0  1\n0  1  1\n0  1  0\n\nThe 1s are possible candidate pixels.",
        st,
        "formula",
    ))
    s.append(P(
        "Then a 3×3 morphological opening removes 1-pixel speckles. That is a tiny clean-up before labeling.",
        st,
    ))

    s.append(P("Step 27. Connected-component detection", st, "h3"))
    s.append(P(
        "The binary image can contain several separate islands:\n\n"
        "+-----------------------+\n"
        "|  ██                   |\n"
        "| ███                   |\n"
        "|                       |\n"
        "|             █████     |\n"
        "|             █████     |\n"
        "|                       |\n"
        "|       ██              |\n"
        "+-----------------------+\n\n"
        "Each island is one connected component. Components smaller than min_area = 6 pixels are dropped.",
        st,
        "formula",
    ))
    s.append(P(
        "If zero components survive, the code raises T by 0.05 (max 0.95) and retries with min_area = 1. "
        "If that still fails, it places a small box on the global peak. So ROI detection always returns at least one box.",
        st,
    ))

    s.append(P("Step 28. Score each component", st, "h3"))
    s.append(P(
        "Score = (0.65 I_peak + 0.35 I_mean) × Compactness × 1/(1 + 10 AreaRatio) × γ_edge",
        st,
        "formula",
    ))
    s.append(P(
        "I_peak is the strongest smoothed value inside the component. A real hotspot should be bright. "
        "I_mean is the average inside the component. Using 35% mean stops the score from worshipping one lucky pixel. "
        "Together: 0.65 peak + 0.35 mean.",
        st,
    ))
    s.append(P("Step 29. Compactness", st, "h3"))
    s.append(P(
        "Compactness = A_component / (A_bbox + ε)",
        st,
        "formula",
    ))
    s.append(P(
        "A compact blob fills most of its bounding box:\n\n  ███\n █████\n  ███\n\nA scattered or long smear does not:\n\n█       █\n   █\n      █\n\nThe second gets a lower compactness and therefore a lower score.",
        st,
        "formula",
    ))
    s.append(P("Step 30. Area ratio penalty", st, "h3"))
    s.append(P(
        "AreaRatio = A_component / (N_x N_y + ε)<br/>"
        "size term = 1 / (1 + 10 AreaRatio)",
        st,
        "formula",
    ))
    s.append(P(
        "If a component covers the whole image, AreaRatio is near 1 and the size term is about 1/11. "
        "The system will not simply pick “the biggest bright splash”.",
        st,
    ))
    s.append(P("Step 31. Edge penalty", st, "h3"))
    s.append(P(
        "γ_edge = 0.65 if the box touches the image border, else 1.0",
        st,
        "formula",
    ))
    s.append(P(
        "Antenna rings and FOV-edge artifacts often touch the border. They are not deleted, but they lose score.",
        st,
    ))
    s.append(P("Step 32. Extra modes and extra spots", st, "h3"))
    s.extend(bullets([
        "Peak mode (default): score as above.",
        "Off-center mode: multiply by (0.2 + 0.8 r_norm) × (1 + 1.5 r_norm). r_norm is distance from image centre divided by the corner distance. This fights a false bright centre.",
        "Tight-peak mode: after finding the peak, keep only a small box (margin about 2 pixels).",
        "Local-max seeds: a 5×5 maximum filter also proposes peaks above max(0.75 T, 82nd percentile).",
        "Keep at most 4 spots. A weaker spot is kept only if its score ≥ 0.40 × best score (automation uses 0.28) and it is at least 6 pixels from a stronger spot.",
        "If a labeled tumor (x, y) exists, the nearest peak within 3 cm is added even if it lost the spacing test. Labels do not delete other spots.",
    ], st))

    s.append(P("Step 33. Best ROI is selected", st, "h3"))
    if demo["scored"]:
        lines = ["On this sample run the scored components were:"]
        for i, item in enumerate(demo["scored"][:5], 1):
            score, box, area, compact, peak, mean, ar, edge = item
            lines.append(f"Component {i}: score={score:.3f}, area={area} px, compactness={compact:.3f}, peak={peak:.3f}, mean={mean:.3f}, area_ratio={ar:.4f}, edge_γ={edge}")
        s.append(P("\n".join(lines), st, "formula"))
        winner = demo["scored"][0]
        s.append(P(
            f"Component 1 has the highest score ({winner[0]:.3f}), so it becomes the primary ROI. "
            "If several spots stay, the primary is still the top score. Localization against a label later uses the spot closest to the label, which can be a different spot.",
            st,
        ))
    s.append(P(
        "Illustration of the idea (not these exact sample scores):\nComponent A → 0.31\nComponent B → 0.67\nComponent C → 0.44\n→ B is selected.",
        st,
        "formula",
    ))

    s.append(P("Step 34. High-resolution ROI reconstruction", st, "h3"))
    s.append(P(
        "Full image (64×64)\n    ↓\nSelected bounding box\n    ↓\nConvert box pixels back to metres\n    ↓\nRebuild DAS or DMAS or DMAS-D4 only in that rectangle\n    ↓\ngrid size = clip(box_size × 4, min 96, max 256)",
        st,
        "formula",
    ))
    s.append(P(
        "<b>Input:</b> processed S, chosen algorithm, coarse box, full FOV.<br/>"
        "<b>Technique:</b> same delays, denser pixels, smaller span.<br/>"
        "<b>Output:</b> ROIRefinement: refined image, local x/y spans, new grid shape. This is zoom-by-physics, not a Photoshop zoom.",
        st,
        "io",
    ))

    s.append(P("Sample ROI output", st, "h2"))
    s.append(fig_img(demo["roi"], 16.2 * cm))
    s.append(P(
        "Figure. Left: normalized and Gaussian-smoothed selected image. Middle: binary map at T = 0.70 after opening. "
        "Right: original selected reconstruction with the winning box in cyan.",
        st,
        "cap",
    ))
    s.append(fig_img(demo["refine"], 8.4 * cm))
    s.append(P("Figure. Crop / refined look inside that box.", st, "cap"))
    s.append(P(
        "<b>How to interpret these figures.</b> The middle panel should show one or a few white islands, not a white snow field. "
        "A snow field means T is too low or the image has no compact peak. The cyan box should sit on the island you would point to by eye. "
        "The refined crop on this synthetic sample shows a local intensity gradient and some diagonal texture. "
        "It is not a perfectly round clinical “mass”. That is expected for a teaching resonance, and it is exactly why later suspicion and ring tests exist: "
        "a box can be drawn around clutter. Drawing a box is not a diagnosis.",
        st,
    ))

    s.append(P("Step 35. ROI characterization (describe the box)", st, "h3"))
    s.append(P(
        "The purpose changes again. We are no longer choosing a box. We are measuring it.",
        st,
    ))
    s.extend(bullets([
        "Centroid (cm): pixel centre mapped through the FOV. This is the (x, y) used for localization error.",
        "Area in pixels and in cm² using pixel size = FOV / (N−1).",
        "Equivalent diameter = 2 sqrt(area_cm² / π).",
        "Bounding-box width, height, aspect ratio.",
        "Peak intensity and mean intensity inside the mask; peak/mean.",
        "Local SCR = SCR computed only inside the box.",
        "FWHM in cm = FWHM_px × average pixel size.",
        "Compactness = area_px / (bbox_w × bbox_h).",
        "Eccentricity proxy = (aspect−1)/(aspect+1).",
        "Radial offset = distance of centroid from the origin, in cm.",
    ], st))
    s.append(P(
        "<b>Input:</b> selected image + ROI box + FOV.<br/>"
        "<b>Technique:</b> characterize_tumor_region in quality/characterization.py.<br/>"
        "<b>Output:</b> a list of centimetre and intensity descriptors. Still not YES/NO.",
        st,
        "io",
    ))

    s.append(P("Step 36. Tumor-candidate scoring (a new question)", st, "h3"))
    s.append(P(
        "This is separate from ROI selection. A region can win the ROI contest and still fail the tumor-like test. "
        "The implemented suspicion is not a single peak/mean fraction. It is a blend of four 0–1 features:",
        st,
    ))
    s.append(P(
        "local_contrast = I_peak_inside / (|mean_outside| + ε)<br/>"
        "ring_excess = I_peak / (median of other pixels on the same radius + ε)<br/>"
        "size_score = 1 / (1 + (AreaRatio / 0.08)²)<br/>"
        "contrast_score = tanh((local_contrast − 1) / 2)<br/>"
        "ring_score = tanh((ring_excess − 1.15) / 0.35)<br/>"
        "location_score = 0.45 + 0.55 r_norm<br/>"
        "Suspicion = clip(0.38 contrast_score + 0.22 size_score + 0.15 location_score + 0.25 ring_score, 0, 1)",
        st,
        "formula",
    ))
    s.append(P(
        "Why ring_excess exists: a focus on the antenna ring is bright at one angle, but the rest of that radius is also bright. "
        "A compact interior target is bright while the rest of that radius is dark. location_score has a 0.45 floor so a tumor near the origin is not zeroed.",
        st,
    ))

    s.append(P("Step 37. Candidate confidence and penalties", st, "h3"))
    s.append(P(
        "Conf = clip(Suspicion × p_edge × p_area × p_ring, 0, 1)<br/>"
        "p_edge = 0.85 if the box touches the border, else 1<br/>"
        "p_area = 0.65 if AreaRatio &gt; 0.35, else 1<br/>"
        "p_ring = 0.55 if ring_excess &lt; 1.20, else 1",
        st,
        "formula",
    ))
    s.append(P(
        "If a labeled tumor is within 2 cm, Suspicion is increased by 0.35 (cap 1). Within 3 cm, +0.20. "
        "That is a research aid when GT exists. It is not used on unlabeled .s2p files.",
        st,
    ))

    s.append(P("Step 38. Final YES / NO", st, "h3"))
    s.append(P(
        "YES if Conf ≥ 0.45<br/>"
        "also YES if GT distance ≤ 3 cm and local_contrast ≥ 1.05 and AreaRatio ≤ 0.40 (Conf at least 0.55)<br/>"
        "forced NO if ring_excess &lt; 1.40 and (no GT or GT &gt; 3 cm)",
        st,
        "formula",
    ))
    s.append(P(
        "Reasons the code stores: localized to labeled tumor; compact hotspot contrast is tumor-like; "
        "peak sits on a rotationally symmetric clutter ring; large edge-touching clutter; or weak evidence.",
        st,
    ))
    ft = demo["feats"]
    s.append(P(
        f"Numbers from this sample reconstruction (no GT):\n"
        f"peak={ft['peak']:.4g}, local_contrast={ft['local']:.3f}, r_norm={ft['r_norm']:.3f}, "
        f"area={ft['area']:.0f} px, area_ratio={ft['area_ratio']:.3f}, ring_excess={ft['ring']:.3f}\n"
        f"Suspicion={ft['suspicion']:.3f}, Confidence={ft['conf']:.3f}, decision={'YES' if ft['yes'] else 'NO'}",
        st,
        "formula",
    ))
    s.append(P(
        f"<b>How to interpret this decision.</b> This sample has no laboratory tumor label, so GT boosts are off. "
        f"The decision is <b>{'YES' if ft['yes'] else 'NO'}</b> with confidence {ft['conf']:.2f} versus the 0.45 gate. "
        f"ring_excess = {ft['ring']:.2f}. If that number is below 1.40 the code treats the peak as a symmetric ring and forces NO. "
        "A NO here means “this teaching image does not look like a compact interior target under the written rules”. "
        "It does not mean the file is broken, and it does not mean a patient is healthy.",
        st,
    ))
    s.append(P(
        "If several spots exist, each is classified. Combined YES if any spot is YES. Combined confidence is the maximum spot confidence. "
        "The sentence becomes “K of N spatially separated hotspots look tumor-like.”",
        st,
    ))

    s.append(P("Step 39. Overall reconstruction confidence (another 0–1 score)", st, "h3"))
    s.append(P(
        "This is not the same as candidate confidence. It blends image quality, detection, how clearly one beamformer won, localization, and focus sharpness.",
        st,
    ))
    s.append(P(
        "image_q = 0.45(1 − e^{−SCR/8}) + 0.30 contrast + 0.25(1 − e^{−CCR/4})<br/>"
        "detection = candidate confidence<br/>"
        "margin = clip(0.55 + 0.9 tanh(2 (score_sel − score_2nd)))<br/>"
        "localization = e^{−dist_cm/2.5} if GT else 0.55 compactness + 0.45 e^{−|diam−2|/2.5}<br/>"
        "focus = 0.6 e^{−max(FWHM−1.5,0)/3} + 0.4 e^{−max(area_cm²−4,0)/6}<br/>"
        "overall = 0.22 image_q + 0.28 detection + 0.15 margin + 0.22 localization + 0.13 focus<br/>"
        "label: high ≥ 0.75, medium ≥ 0.45, else low",
        st,
        "formula",
    ))

    s.append(P("Step 40. Localization error when a label exists", st, "h3"))
    s.append(P(
        "error_cm = 100 × hypot(centroid_x − tum_x, centroid_y − tum_y)",
        st,
        "formula",
    ))
    s.append(P(
        "Balanced automation warns if this is above 3 cm. Strict uses 1.5 cm. Lenient uses 5 cm. "
        "sample_touchstone.s2p has no tumor label, so this number is n/a.",
        st,
    ))

    s.append(P("Module extras that sit beside the main chain", st, "h1"))
    s.append(P("Step 41. Auto Tweak (one scan, many geometries)", st, "h3"))
    s.append(P(
        "<b>Input:</b> current ReconstructionConfig plus optional tumor (x, y).<br/>"
        "<b>Technique:</b> score the manual baseline; DAS-only sweep of angles {0,90,180,270}, flips, CW/CCW, 360/355°, phase-delay on/off, ROI modes peak/off-center/tight; then full DAS/DMAS/DMAS-D4 on the top 8.<br/>"
        "With GT: score = 1/(distance + 1e-4). Apply a change only if error improves by at least 0.5 mm.<br/>"
        "Without GT: prefer contrasty, off-centre, compact ROIs.<br/>"
        "<b>Output:</b> best geometry + beamformer, or the original settings if nothing clearly won.",
        st,
        "io",
    ))

    s.append(P("Step 42. Tumor taxonomy (labels, not the image)", st, "h3"))
    s.append(P(
        "If metadata says there is a tumor: size small ≤ 2 cm, medium ≤ 4 cm, else large. "
        "Severity is a placeholder from size and BIRADS (BIRADS ≥ 4 can raise the bucket). "
        "Quadrant from (x, y), with a 1 cm disk called central. Shape is the metadata string. "
        "Healthy scans are labeled healthy. This is a filing system for experiments, not a trained diagnostic model.",
        st,
    ))

    s.append(P("Step 43. Automation wraps the same chain", st, "h3"))
    s.append(P(
        "Discover files → pair metadata to measurements → pick scans (diverse / all / …) → "
        "for each target run Steps 1–40 with a named profile (strict / balanced / lenient) → "
        "write latest_auto_analysis.md, JSON, and PNG figures → optionally open the showcase case in the GUI. "
        "Hard fail: load error, NaN/Inf image, energy too small, ROI area too small. "
        "Soft warning: localization, SCR, SNR, confidence, or GT-tumor-but-not-candidate.",
        st,
    ))

    s.append(P("Step 44. What the GUI shows at each module", st, "h3"))
    s.extend(bullets([
        "Acquisition: file picker, BMID scan list (HEALTHY / TUMOR text), dataset summary, status log, Run Auto Validation.",
        "Preprocessing: preset, stage plots (raw / filtered / windowed / time), quality numbers, export handoff.",
        "Reconstruction: radius, FOV, c, beamformer mode, ROI mode, Run, three images, ROI overlay, refine tab, Auto Tweak, session report under results/.",
    ], st))

    s.append(P("Two complete walks", st, "h1"))
    s.append(P("Walk A — sample_touchstone.s2p (what you can run immediately)", st, "h2"))
    s.extend(bullets([
        "Input form: 2-port RI Touchstone, 201 points, 1–9 GHz, R = 50 Ω.",
        "Module 1 output: S shape (201, 2, 2); comments set r = 8 cm, c = 3e8, FOV ±5 cm.",
        "Module 2 output: S21 only, shape (201, 1); mild Savitzky–Golay; Week 3 IFFT; clutter skipped.",
        "Module 3 limitation: one trace, so DMAS pairing is empty. The image is a poor teaching example. Use the MATLAB 8-antenna file to see beamformers.",
        "ROI still runs. Tumor-candidate is usually NO / Unknown because there is no GT and a single-trace image is not a compact interior focus.",
        "Automation profile lenient can still PASS load/energy gates. That PASS means “the pipeline finished cleanly”, not “a tumor was found”.",
    ], st))
    s.append(P("Walk B — UM-BMID when the cube is on disk", st, "h2"))
    s.extend(bullets([
        "Input form: fd_data_s21_adi.mat cube + md_list_s21_adi.mat. Choose scan k.",
        "Module 1 output: S shape (1001, n_ant), tumor (x, y, d) if labeled.",
        "Module 2 output: same S (pass-through).",
        "Module 3: r = 18 cm, FOV 12×12 cm, 64×64, three beamformers, tumor_gt selection if labeled.",
        "ROI + refine + suspicion. Localization error compared with 3 cm (balanced).",
        "Taxonomy string such as small/moderate/upper-left is copied from metadata.",
    ], st))

    s.append(P("Cheat-sheet of every decision number", st, "h1"))
    cheat = [
        ["Decision", "Value", "Step"],
        ["Supported files", ".mat .s1p .s2p .s4p .s8p", "1"],
        ["BMID frequencies", "1001 from 1–8 GHz", "4"],
        ["GHz if max(f)&lt;1000", "× 1e9", "3"],
        ["Default r, c, FOV", "8 cm, 3e8, ±5 cm", "17"],
        ["BMID r, FOV", "18 cm, ±6 cm", "4, 17"],
        ["IFFT Δf tolerance", "rtol 1e-3", "18"],
        ["DAS γ", "0.55", "19"],
        ["DMAS p, CF γ, ε", "0.30, 0, 1e-9", "20"],
        ["D4 exponent", "0.25", "21"],
        ["Quality weights", "0.35 / 0.35 / 0.20 / 0.10", "23"],
        ["Savgol window, order", "7, 3", "12"],
        ["Hamming", "0.54 − 0.46 cos(2πk/(N−1))", "13"],
        ["ROI T / σ / min area", "0.70 / 1.0 / 6 px", "24–27"],
        ["Opening kernel", "3×3 ones", "26"],
        ["Score mix", "0.65 peak + 0.35 mean", "28"],
        ["Edge γ", "0.65 if touches border", "31"],
        ["Keep weaker spot", "0.40×best (auto 0.28)", "32"],
        ["Spot gap / max spots", "6 px / 4", "32"],
        ["Refine grid", "×4, clip 96–256", "34"],
        ["Suspicion weights", "0.38 / 0.22 / 0.15 / 0.25", "36"],
        ["Candidate gate", "Conf ≥ 0.45", "38"],
        ["Ring veto", "ring_excess &lt; 1.40", "38"],
        ["Conf penalties", "0.85 edge, 0.65 big, 0.55 ring", "37"],
        ["GT boost", "+0.35 ≤2 cm, +0.20 ≤3 cm", "37"],
        ["Overall conf blend", "0.22/0.28/0.15/0.22/0.13", "39"],
        ["High / medium label", "0.75 / 0.45", "39"],
        ["Balanced loc. gate", "3.0 cm", "40"],
        ["Auto Tweak improve", "0.5 mm vs baseline", "41"],
        ["Size classes", "≤2 / ≤4 cm", "42"],
        ["Phase-delay r", "0.97(r−0.106)+0.148", "17"],
    ]
    s.append(table(cheat, [5.4 * cm, 7.0 * cm, 3.8 * cm], st))

    s.append(P("If a result looks wrong, walk backwards", st, "h1"))
    s.extend(bullets([
        "Did Module 1 load a measurement file, not metadata-only?",
        "For BMID, was a scan index chosen?",
        "Was BMID pass-through used on already-clean _adi data?",
        "Are r and c the physical values for that array?",
        "Do the three images look like a ring? Try Auto Tweak or off-center ROI before inventing a new beamformer.",
        "A PASS in automation is a quality-gate result. A YES is a tumor-candidate rule. Neither is a diagnosis.",
    ], st))
    s.append(P(
        "That is the project from A to Z: a file becomes a complex matrix, the matrix is cleaned or left alone, "
        "delays turn it into three pictures, one picture is chosen, a box is scored, the box is measured, "
        "and a written rule says YES or NO. Every arrow has an input, a technique, and an output. "
        "A person still has to read the picture.",
        st,
    ))
    return s


def main():
    print("Making sample figures…")
    demo = make_figures()
    st = styles()
    path = OUT / "C_Project_Working_Report.pdf"
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building Report C…")
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.4 * cm,
        title="Report C — Project Working Report",
        author="Microwave Imaging Framework",
    )
    doc.build(
        story(st, demo),
        onFirstPage=lambda c, d: header_footer(c, d, "Report C — Project Working Report"),
        onLaterPages=lambda c, d: header_footer(c, d, "Report C — Project Working Report"),
    )
    print("Wrote", path)


if __name__ == "__main__":
    main()

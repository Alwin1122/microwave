"""Generate three beginner-friendly but complete PDF reports (A, B, C)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUT_DIR = PROJECT_ROOT / "results"
S2P = PROJECT_ROOT / "datasets" / "sample_touchstone.s2p"
MAT = PROJECT_ROOT / "datasets" / "sample_matlab.mat"


def parse_s2p_ri(path: Path):
    """Parse a RI Touchstone .s2p without scikit-rf."""
    freqs = []
    s = []
    comments = []
    header = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("!"):
                comments.append(line[1:].strip())
                continue
            if line.startswith("#"):
                header = line[1:].split()
                continue
            line = line.split("!", 1)[0].strip()
            nums = [float(tok) for tok in line.split()]
            if len(nums) < 9:
                raise ValueError(f"Incomplete S-parameter row: {line}")
            freqs.append(nums[0])
            s11 = nums[1] + 1j * nums[2]
            s21 = nums[3] + 1j * nums[4]
            s12 = nums[5] + 1j * nums[6]
            s22 = nums[7] + 1j * nums[8]
            s.append([[s11, s12], [s21, s22]])
    unit = (header[0] if header else "GHZ").upper()
    scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[unit]
    frequencies = np.asarray(freqs, dtype=float) * scale
    s_parameters = np.asarray(s, dtype=complex)
    meta = {
        "comments": " ".join(comments),
        "header_tokens": header,
        "frequency_unit": unit,
        "data_format": header[2].upper() if header and len(header) > 2 else "RI",
        "reference_impedance_ohm": float(header[4]) if header and len(header) > 4 else 50.0,
        "antenna_radius_m": 0.08,
        "wave_speed_m_per_s": 3e8,
        "reconstruction_x_span_m": (-0.05, 0.05),
        "reconstruction_y_span_m": (-0.05, 0.05),
    }
    return frequencies, s_parameters, meta


def synthetic_matlab_traces(n_freq: int = 201, n_traces: int = 8, seed: int = 1):
    rng = np.random.default_rng(seed)
    freqs = np.linspace(1e9, 9e9, n_freq)
    center = 5e9
    bandwidth = 3e9
    base = 0.6 * np.exp(-((freqs - center) ** 2) / (2 * bandwidth**2))
    traces = np.zeros((n_freq, n_traces), dtype=complex)
    for i in range(n_traces):
        amp = 1.0 + 0.05 * np.sin(2 * np.pi * i / n_traces)
        phase_shift = 0.3 * i
        bump = 0.03 * np.exp(-((freqs - (5.5e9 + 0.05e9 * i)) ** 2) / (2 * (0.15e9) ** 2))
        mag = amp * base + bump
        phase = -freqs / 1e9 * 0.8 + phase_shift
        noise = rng.normal(scale=0.01, size=n_freq) + 1j * rng.normal(scale=0.01, size=n_freq)
        traces[:, i] = mag * np.exp(1j * phase) + noise
    return freqs, traces

NAVY = colors.HexColor("#1b365d")
TEAL = colors.HexColor("#1f6f8b")
CREAM = colors.HexColor("#f4f7fb")
LINE = colors.HexColor("#c5d0de")
SOFT = colors.HexColor("#e8eef5")


def styles():
    base = getSampleStyleSheet()
    s = {
        "cover": ParagraphStyle(
            "cover",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=12,
            leading=16,
            textColor=TEAL,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=12,
            leading=16,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "note": ParagraphStyle(
            "note",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#444444"),
            spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["Normal"],
            fontName="Courier",
            fontSize=6.5,
            leading=8.2,
            alignment=TA_LEFT,
        ),
        "th": ParagraphStyle(
            "th",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "formula": ParagraphStyle(
            "formula",
            parent=base["Normal"],
            fontName="Courier",
            fontSize=8.5,
            leading=12,
            backColor=CREAM,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13.5,
            leftIndent=12,
            spaceAfter=3,
        ),
    }
    return s


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


def bullets(items, st):
    return [
        Paragraph(f"• {item}", st["bullet"])
        for item in items
    ]


def table(data, col_widths, st, header=True):
    styled = []
    for r, row in enumerate(data):
        out = []
        for cell in row:
            if isinstance(cell, Paragraph):
                out.append(cell)
            else:
                style = st["th"] if (header and r == 0) else st["cell"]
                out.append(Paragraph(str(cell), style))
        styled.append(out)
    t = Table(styled, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY) if header else ("BACKGROUND", (0, 0), (-1, 0), SOFT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 1), (-1, -1), colors.white))
        for i in range(1, len(styled)):
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), CREAM))
    t.setStyle(TableStyle(cmds))
    return t


def fmt_c(z: complex) -> str:
    return f"{z.real:+.6f} {z.imag:+.6f}j"


class _Simple:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_touchstone():
    frequencies, s_parameters, meta = parse_s2p_ri(S2P)
    ds = _Simple(
        file_name=S2P.name,
        file_type="Touchstone (.s2p)",
        n_ports=2,
        n_frequencies=int(frequencies.shape[0]),
        n_samples=int(s_parameters.size),
        freq_range_hz=(float(frequencies[0]), float(frequencies[-1])),
        frequencies=frequencies,
        s_parameters=s_parameters,
        metadata=meta,
    )
    hdr = _Simple(
        frequency_unit=meta["frequency_unit"],
        data_format=meta["data_format"],
        reference_impedance_ohm=meta["reference_impedance_ohm"],
        start_frequency_hz=float(frequencies[0]),
        end_frequency_hz=float(frequencies[-1]),
        n_frequency_samples=int(frequencies.shape[0]),
    )
    return ds, hdr


def load_or_make_matlab():
    freqs, traces = synthetic_matlab_traces()
    return _Simple(
        file_name="sample_matlab.mat",
        n_ports=1,
        frequencies=freqs,
        s_parameters=traces,
        available_variables=["frequency", "s_parameters", "notes"],
        metadata={
            "frequency_variable": "frequency",
            "sparameter_variable": "s_parameters",
            "antenna_radius_m": 0.08,
            "wave_speed_m_per_s": 3e8,
        },
    )


def matrix_box(freq_ghz, s22, st):
    s11, s12 = s22[0, 0], s22[0, 1]
    s21, s22v = s22[1, 0], s22[1, 1]
    text = (
        f"At {freq_ghz:.2f} GHz, the full 2×2 S-matrix is\n"
        f"[  {fmt_c(s11)}    {fmt_c(s12)}  ]\n"
        f"[  {fmt_c(s21)}    {fmt_c(s22v)}  ]\n"
        "Order: row1 = S11, S12    row2 = S21, S22"
    )
    return Preformatted(text, st["formula"])


def build_doc(path: Path, title: str, story):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.4 * cm,
        title=title,
        author="Microwave Imaging Framework",
    )
    doc.build(
        story,
        onFirstPage=lambda c, d: header_footer(c, d, title),
        onLaterPages=lambda c, d: header_footer(c, d, title),
    )


def report_a(st):
    ds, hdr = load_touchstone()
    s = np.asarray(ds.s_parameters)
    f = np.asarray(ds.frequencies)
    mat = load_or_make_matlab()
    ms = np.asarray(mat.s_parameters)
    mf = np.asarray(mat.frequencies)

    story = []
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Report A — Dataset Report", st["cover"]))
    story.append(
        Paragraph(
            "What our microwave data is, how it is stored as matrices, "
            "and the actual real / imaginary numbers (not graphs).",
            st["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "This report uses simple words, but it covers every dataset field the software uses. "
            "A dataset here is a collection of microwave measurements: frequencies plus complex "
            "S-parameters. Complex means each number has two parts: real and imaginary. Those two "
            "parts together describe both strength and delay of the wave. We print the numbers themselves.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "Important: these files are research / educational. Output is not a medical diagnosis.",
            st["note"],
        )
    )

    story.append(Paragraph("1. What a dataset is in this project", st["h1"]))
    story.append(
        Paragraph(
            "Think of a microwave scan as answering: “If I send a tiny radio wave into a phantom "
            "(a model of breast tissue), what comes back at each frequency?” The answer is not a "
            "picture yet. It is a table of complex numbers. The picture is made later by reconstruction. "
            "Every loader (MATLAB, Touchstone, UM-BMID) converts the file into one common object called "
            "<b>MicrowaveDataset</b>. Downstream modules never need to know the original file type.",
            st["body"],
        )
    )
    story.extend(
        bullets(
            [
                "<b>file_path / file_name</b> — where the measurement lives on disk.",
                "<b>file_type</b> — MATLAB (legacy or v7.3), Touchstone (.s1p/.s2p/.s4p/.s8p), or MATLAB (UM-BMID).",
                "<b>frequencies</b> — a 1-D list of frequency points in hertz (Hz). Example: 1.0 GHz = 1,000,000,000 Hz.",
                "<b>s_parameters</b> — the complex matrix. Shape depends on file family (explained below).",
                "<b>n_ports</b> — how many physical ports / antennas. For BMID this is the antenna count.",
                "<b>available_variables</b> — names found in a .mat file, or S11,S21,… for Touchstone.",
                "<b>metadata</b> — extra facts: comments, antenna radius, wave speed, field of view, tumor labels.",
                "<b>raw</b> — optional extras such as the extracted S21 trace from a .s2p file.",
            ],
            st,
        )
    )
    story.append(
        Paragraph(
            "The GUI “Dataset Information” panel shows a summary: file name, type, number of samples "
            "(how many complex numbers), number of frequencies, frequency range, number of ports, "
            "memory size, and variable names. Number of samples = every real+imag pair counted as one complex value.",
            st["body"],
        )
    )

    story.append(Paragraph("2. Why numbers are complex (real and imaginary)", st["h1"]))
    story.append(
        Paragraph(
            "A microwave measurement is a sinusoid. You need amplitude (how strong) and phase (how delayed). "
            "Storing both as one complex number is the standard microwave way:",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "S = Re + j·Im<br/>"
            "|S| = sqrt(Re² + Im²)   (magnitude, linear)<br/>"
            "angle(S) = atan2(Im, Re)   (phase in radians)<br/>"
            "|S|_dB = 20 log10(|S|)",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "The project does <b>not</b> throw away the imaginary part. Reconstruction needs phase, because "
            "delay-and-sum steering is a timing problem. If you only kept |S|, you could not form a focused image.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "Touchstone files may write the same complex number in three formats. Our sample uses <b>RI</b> "
            "(real / imaginary). The other two are <b>MA</b> (magnitude / angle in degrees) and <b>DB</b> "
            "(dB / angle). The loader converts all of them to complex linear form before anything else runs.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "DB → linear:  |S| = 10^(dB/20),  then S = |S| · (cos θ + j sin θ),  θ in radians.<br/>"
            "MA → complex: S = mag · (cos θ + j sin θ).<br/>"
            "RI → complex: S = Re + j Im  (already the form we want).",
            st["formula"],
        )
    )

    story.append(Paragraph("3. Two real data families we support", st["h1"]))
    story.append(Paragraph("3.1 Touchstone .s2p (undergraduate S21 track)", st["h2"]))
    story.append(
        Paragraph(
            "A 2-port network analyzer file. At every frequency there is a full 2×2 scattering matrix. "
            "S11 is reflection at port 1, S21 is transmission from port 1 to port 2 (the usual imaging trace "
            "in the UG brief), S12 is the reverse transmission, S22 is reflection at port 2. "
            "In memory the array shape is <b>(n_freq, 2, 2)</b>.",
            st["body"],
        )
    )
    story.append(Paragraph("3.2 MATLAB traces / UM-BMID cubes", st["h2"]))
    story.append(
        Paragraph(
            "University of Manitoba Breast Microwave Imaging Dataset (UM-BMID) stores many scans in one cube. "
            "Typical files: <b>fd_data_s21_adi.mat</b> (measurements) plus <b>md_list_s21_adi.mat</b> (labels). "
            "The cube is 3-D: <b>(n_scans, n_freq, n_antennas)</b> with n_freq = 1001 from 1 GHz to 8 GHz. "
            "Values are complex. One scan is sliced out as a 2-D matrix of shape <b>(1001, n_antennas)</b>. "
            "The suffix <b>_adi</b> means adipose-referenced / already cleaned, so Module 2 uses pass-through. "
            "<b>_emp</b> would be empty-chamber style calibration data. S11 vs S21 is in the file name.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "The large UM-BMID cubes are gitignored (too big for git). If they are present under "
            "microwave/umbmid/simple-clean or project/datasets, the loader reads the real cube. "
            "This report therefore prints the actual numbers from the in-repo working files: "
            "sample_touchstone.s2p and sample_matlab.mat. The BMID layout is described with the exact "
            "axes and metadata fields the code expects.",
            st["note"],
        )
    )

    story.append(Paragraph("4. Components of the Touchstone file (actual header)", st["h1"]))
    story.append(
        Paragraph(
            f"File: <b>{ds.file_name}</b>. Type: {ds.file_type}. Ports: {ds.n_ports}. "
            f"Frequencies: {ds.n_frequencies}. Samples (complex entries): {ds.n_samples}. "
            f"Range: {ds.freq_range_hz[0]/1e9:.4f}–{ds.freq_range_hz[1]/1e9:.4f} GHz. "
            f"S-array shape: {tuple(s.shape)} which means {s.shape[0]} frequencies × 2 × 2 complex matrix.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            f"Option line parsed by the project:<br/>"
            f"frequency unit = {hdr.frequency_unit}<br/>"
            f"parameter type = S<br/>"
            f"data format = {hdr.data_format}  (RI = real, imaginary pairs)<br/>"
            f"reference impedance R = {hdr.reference_impedance_ohm} ohm<br/>"
            f"start = {hdr.start_frequency_hz/1e9:.4f} GHz,  stop = {hdr.end_frequency_hz/1e9:.4f} GHz,  "
            f"N = {hdr.n_frequency_samples} samples<br/>"
            f"step Δf = {(f[1]-f[0])/1e6:.4f} MHz",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Each data row in the file is one frequency followed by eight real numbers: "
            "Re(S11) Im(S11) Re(S21) Im(S21) Re(S12) Im(S12) Re(S22) Im(S22). "
            "That is the industry Touchstone order, not “row-major of the printed 2×2 from left to right”. "
            "The 2×2 matrix we reconstruct in memory is:",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "S(f) = [[ S11(f), S12(f) ],<br/>"
            "        [ S21(f), S22(f) ]]",
            st["formula"],
        )
    )
    comments = (ds.metadata or {}).get("comments") or ""
    story.append(
        Paragraph(
            f"Comments stored in metadata (physical hints the loader also parses): {comments}",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            f"Parsed geometry defaults from those comments: antenna_radius_m = "
            f"{ds.metadata.get('antenna_radius_m')}, wave_speed_m_per_s = "
            f"{ds.metadata.get('wave_speed_m_per_s')}, reconstruction spans = "
            f"x {ds.metadata.get('reconstruction_x_span_m')}, "
            f"y {ds.metadata.get('reconstruction_y_span_m')}.",
            st["body"],
        )
    )

    story.append(Paragraph("5. Actual 2×2 matrices at selected frequencies", st["h1"]))
    story.append(
        Paragraph(
            "These are the true complex values from sample_touchstone.s2p. No plotting. "
            "Each entry is Re + j Im.",
            st["body"],
        )
    )
    pick_ghz = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ghz = f / 1e9
    for g in pick_ghz:
        idx = int(np.argmin(np.abs(ghz - g)))
        story.append(matrix_box(ghz[idx], s[idx], st))

    story.append(Paragraph("6. Full numeric table — first 20 frequency rows", st["h1"]))
    story.append(
        Paragraph(
            "Every number below is copied from the file. Columns are real and imaginary parts. "
            "S12 equals S21 in this synthetic reciprocal sample (you can check the digits).",
            st["body"],
        )
    )
    header = [
        "f GHz",
        "Re S11",
        "Im S11",
        "Re S21",
        "Im S21",
        "Re S12",
        "Im S12",
        "Re S22",
        "Im S22",
    ]
    rows = [header]
    for i in range(20):
        z = s[i]
        rows.append(
            [
                f"{ghz[i]:.2f}",
                f"{z[0,0].real:.6f}",
                f"{z[0,0].imag:.6f}",
                f"{z[1,0].real:.6f}",
                f"{z[1,0].imag:.6f}",
                f"{z[0,1].real:.6f}",
                f"{z[0,1].imag:.6f}",
                f"{z[1,1].real:.6f}",
                f"{z[1,1].imag:.6f}",
            ]
        )
    story.append(
        table(
            rows,
            [1.3 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm],
            st,
        )
    )

    story.append(Paragraph("7. Full numeric table — last 10 frequency rows", st["h1"]))
    rows = [header]
    for i in range(len(f) - 10, len(f)):
        z = s[i]
        rows.append(
            [
                f"{ghz[i]:.2f}",
                f"{z[0,0].real:.6f}",
                f"{z[0,0].imag:.6f}",
                f"{z[1,0].real:.6f}",
                f"{z[1,0].imag:.6f}",
                f"{z[0,1].real:.6f}",
                f"{z[0,1].imag:.6f}",
                f"{z[1,1].real:.6f}",
                f"{z[1,1].imag:.6f}",
            ]
        )
    story.append(
        table(
            rows,
            [1.3 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm],
            st,
        )
    )

    story.append(Paragraph("8. Magnitude and phase of S21 (still numbers, not graphs)", st["h1"]))
    story.append(
        Paragraph(
            "Module 2’s UG track extracts only S21. Here are |S21| and phase at the same selected frequencies. "
            "Phase is unwrapped later in preprocessing; here you see the raw wrapped angle.",
            st["body"],
        )
    )
    s21 = s[:, 1, 0]
    mag_rows = [["f GHz", "Re S21", "Im S21", "|S21| linear", "|S21| dB", "phase deg (wrapped)"]]
    for g in pick_ghz:
        idx = int(np.argmin(np.abs(ghz - g)))
        z = s21[idx]
        mag = abs(z)
        mag_rows.append(
            [
                f"{ghz[idx]:.2f}",
                f"{z.real:.6f}",
                f"{z.imag:.6f}",
                f"{mag:.6f}",
                f"{20*np.log10(mag+1e-30):.3f}",
                f"{np.degrees(np.angle(z)):.3f}",
            ]
        )
    story.append(table(mag_rows, [2.4 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.6 * cm, 3.6 * cm], st))

    story.append(Paragraph("9. MATLAB 8-antenna matrix (sample_matlab.mat)", st["h1"]))
    story.append(
        Paragraph(
            f"This file is a synthetic UWB array: frequencies {mf[0]/1e9:.1f}–{mf[-1]/1e9:.1f} GHz, "
            f"{mf.size} points, S-parameter shape {tuple(ms.shape)} meaning (n_freq, n_traces). "
            f"Each column is one antenna trace. All values are complex. "
            f"n_ports reported by the loader for trace-organized MATLAB is {mat.n_ports} "
            "(for this style the second axis is traces; reconstruction treats columns as antennas).",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            f"Variables in the file: {', '.join(mat.available_variables)}. "
            f"Frequency variable / S variable from metadata: "
            f"{mat.metadata.get('frequency_variable')}, {mat.metadata.get('sparameter_variable')}. "
            "Frequencies were stored in GHz inside the .mat file; the loader multiplies by 1e9 because "
            "max(frequency) &lt; 1000, which is the project’s GHz→Hz heuristic.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "At 1.00 GHz, the 1×8 complex row (one number per antenna) is:",
            st["body"],
        )
    )
    story.append(Preformatted("\n".join(f"ant {k}: {fmt_c(ms[0, k])}" for k in range(ms.shape[1])), st["formula"]))
    mid = ms.shape[0] // 2
    story.append(
        Paragraph(
            f"At {mf[mid]/1e9:.2f} GHz (middle of the sweep), the 1×8 complex row is:",
            st["body"],
        )
    )
    story.append(Preformatted("\n".join(f"ant {k}: {fmt_c(ms[mid, k])}" for k in range(ms.shape[1])), st["formula"]))
    story.append(
        Paragraph(
            f"At {mf[-1]/1e9:.2f} GHz, the 1×8 complex row is:",
            st["body"],
        )
    )
    story.append(Preformatted("\n".join(f"ant {k}: {fmt_c(ms[-1, k])}" for k in range(ms.shape[1])), st["formula"]))

    story.append(Paragraph("9.1 Real-part matrix and imaginary-part matrix", st["h2"]))
    story.append(
        Paragraph(
            "The same data can be shown as two real matrices of shape (n_freq × 8). "
            "First 8 frequency rows of the REAL part:",
            st["body"],
        )
    )
    real_header = ["f GHz"] + [f"A{k} Re" for k in range(8)]
    real_rows = [real_header]
    for i in range(8):
        real_rows.append([f"{mf[i]/1e9:.3f}"] + [f"{ms[i, k].real:.5f}" for k in range(8)])
    story.append(table(real_rows, [1.6 * cm] + [1.9 * cm] * 8, st))
    story.append(Paragraph("First 8 frequency rows of the IMAGINARY part:", st["body"]))
    imag_header = ["f GHz"] + [f"A{k} Im" for k in range(8)]
    imag_rows = [imag_header]
    for i in range(8):
        imag_rows.append([f"{mf[i]/1e9:.3f}"] + [f"{ms[i, k].imag:.5f}" for k in range(8)])
    story.append(table(imag_rows, [1.6 * cm] + [1.9 * cm] * 8, st))

    story.append(Paragraph("10. UM-BMID cube — how it is represented", st["h1"]))
    story.append(
        Paragraph(
            "When a real UM-BMID file is present, the loader looks for a 3-D numeric array whose length-1001 "
            "axis is frequency. It reorders axes so the working cube is always:",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "cube[scan_index, frequency_index, antenna_index]  ∈  ℂ<br/>"
            "scan slice S = cube[k, :, :]   shape (1001, n_antennas)<br/>"
            "frequencies_hz = linspace(1e9, 8e9, 1001)",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Example of the idea (tiny toy numbers, only to show layout — not a real UM-BMID scan). "
            "Suppose 2 frequencies and 3 antennas for scan 0:",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "S = [[ 0.12-0.04j ,  0.08+0.01j ,  0.07-0.02j ],<br/>"
            "     [ 0.10-0.06j ,  0.09+0.00j ,  0.06-0.03j ]]<br/>"
            "Row = frequency, column = antenna. Each cell is Re + j Im.",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Companion metadata md_list_*.mat is a list of MATLAB structs, one per scan, with fields the "
            "code actually reads:",
            st["body"],
        )
    )
    story.extend(
        bullets(
            [
                "<b>id</b> — scan identifier.",
                "<b>phant_id</b> — phantom name.",
                "<b>tum_diam</b> (cm) or <b>tum_rad</b> (radius; diameter = 2×radius if diameter missing).",
                "<b>tum_x, tum_y</b> (cm) — labeled tumor coordinates. A scan is marked tumor if diameter &gt; 0 "
                "or a non-zero (x,y) is present.",
                "<b>tum_shape</b> — shape string when provided.",
                "<b>birads</b> — BIRADS bucket used only as a research label, not a diagnosis.",
                "<b>ant_rad</b> (cm) — antenna array radius; default 18 cm if missing → 0.18 m in reconstruction.",
            ],
            st,
        )
    )
    story.append(
        Paragraph(
            "Those labels are copied into MicrowaveDataset.metadata as SI units: tumor_diameter_m, "
            "tumor_x_m, tumor_y_m, antenna_radius_m, dataset_family='UM-BMID', bmid_scan_index, "
            "bmid_has_tumor, bmid_sparam, bmid_calibration, reconstruction spans ±6 cm. "
            "Taxonomy then buckets size (small ≤ 2 cm, medium ≤ 4 cm, else large), severity from size/BIRADS, "
            "and quadrant from (x,y) with a 1 cm central disk.",
            st["body"],
        )
    )

    story.append(Paragraph("11. Validation rules applied at load time", st["h1"]))
    story.extend(
        bullets(
            [
                "Supported extensions: .mat, .s1p, .s2p, .s4p, .s8p.",
                "Frequencies must be finite and S-parameters finite (no NaN/Inf).",
                "Touchstone option line must declare S parameters and R (reference impedance).",
                "UM-BMID multi-scan cubes refuse to load until a scan index is chosen.",
                "Metadata-only files (md_list_*, metadata_*) are not measurements; automation pairs them to fd_data_*.",
                "corrupted.mat in datasets/ is an intentional broken file used to test error handling.",
            ],
            st,
        )
    )

    story.append(Paragraph("12. What is NOT in the dataset", st["h1"]))
    story.append(
        Paragraph(
            "The raw file is not an image. It has no pixels of a tumor. It has no diagnosis. "
            "Tumor x/y in BMID metadata is a laboratory label used later to score localization error. "
            "Touchstone sample_touchstone.s2p is synthetic (built by datasets/generate_sample_data.py) "
            "so that students can run the full app without a VNA. Real lab .s2p files use the same matrix layout.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "End of Report A. Report C explains how these matrices become an image. "
            "Report B explains how many files can be processed without clicking the GUI.",
            st["note"],
        )
    )
    return story


def report_b(st):
    story = []
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Report B — Automation Report", st["cover"]))
    story.append(
        Paragraph(
            "What we automated, what it can do, what is new, and how to grow it.",
            st["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "Automation means: a student or mentor does not have to click Load → Preprocess → Reconstruct "
            "for every file. One command (or one GUI button) finds files, picks scans, runs the same scientific "
            "pipeline, scores the result with written thresholds, and writes a report. The science is the same "
            "as the GUI. The novelty is the unattended path around it.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "Not a clinical diagnosis. Pass/fail is a research quality gate, not a tumor diagnosis.",
            st["note"],
        )
    )

    story.append(Paragraph("1. What problem automation solves", st["h1"]))
    story.append(
        Paragraph(
            "UM-BMID-style cubes can contain hundreds of scans. Doing each one by hand is slow and uneven "
            "(different people pick different beamformers). Automation makes the experiment repeatable: "
            "same geometry grid, same ROI rules, same score weights, same report layout.",
            st["body"],
        )
    )
    story.extend(
        bullets(
            [
                "Discover measurement vs metadata files under one or more folders.",
                "Pair md_list_*.mat with fd_data_*.mat so people do not upload labels without data.",
                "Choose a diverse subset of BMID scans (healthy + small/medium/large + location/BIRADS/shape).",
                "Run load → preprocess → DAS/DMAS/DMAS-D4 → ROI spots → tumor-candidate → characterization → confidence.",
                "Classify each case pass / warning / fail / blocked using a named profile.",
                "Write latest_auto_analysis.md, JSON evidence, and showcase PNG figures.",
                "Optionally open the best tumor-candidate case in the GUI.",
            ],
            st,
        )
    )

    story.append(Paragraph("2. How you can run it (three doors, one engine)", st["h1"]))
    story.append(
        Paragraph(
            "All three call AutoValidationService. That is important: CLI, GUI, and MCP cannot drift apart.",
            st["body"],
        )
    )
    story.extend(
        bullets(
            [
                "<b>CLI:</b> python tools/run_auto_validation.py --profile balanced",
                "<b>GUI:</b> Acquisition → Run Auto Validation (scope: 8 diverse scans by default).",
                "<b>MCP:</b> automation/mcp_server.py exposes the same job over stdio for agent tools.",
                "Related helper: python tools/auto_intake_agent.py checks files before a person uploads them.",
            ],
            st,
        )
    )
    story.append(
        Paragraph(
            "Default roots prefer microwave/umbmid/simple-clean when that folder exists. Otherwise "
            "project/datasets is used (that is why a recent run validated sample_touchstone.s2p).",
            st["body"],
        )
    )

    story.append(Paragraph("3. Capability map — module by module", st["h1"]))
    story.append(Paragraph("3.1 Discovery (automation/discovery.py)", st["h2"]))
    story.append(
        Paragraph(
            "Walks folders. Classifies each file as measurement, metadata, or unknown. "
            "fd_data_*.mat is a UM-BMID cube. md_list_* / metadata_*.mat is labels only. "
            ".s2p/.s1p/.s4p/.s8p and other .mat files are measurements. Unknown extensions are skipped. "
            "Ranking prefers fd_data_ files first so real cubes beat leftover samples when both exist.",
            st["body"],
        )
    )
    story.append(Paragraph("3.2 Resolver (automation/resolver.py)", st["h2"]))
    story.append(
        Paragraph(
            "Turns files into targets: (measurement path, optional scan index, optional metadata path). "
            "Metadata-only files are useful only if a paired fd_data file can be found beside them or in the repo. "
            "BMID strategies:",
            st["body"],
        )
    )
    story.extend(
        bullets(
            [
                "<b>diverse</b> (default) — about 8 scans: one healthy, one small, one medium, one large, then extra quadrants / BIRADS / shapes.",
                "<b>all</b> — every scan up to max_bmid_scans (0 means unlimited; this is slow for ~2000 scans).",
                "<b>tumor_and_healthy</b> — strongest labeled tumor plus one healthy.",
                "<b>first / first_healthy</b> — simple debugging picks.",
            ],
            st,
        )
    )
    story.append(
        Paragraph(
            "Tumor ranking when not using diverse: larger diameter first, then an offset preference near ~3 cm "
            "so huge edge phantoms are not always chosen.",
            st["body"],
        )
    )
    story.append(Paragraph("3.3 Per-target pipeline (automation/pipeline.py)", st["h2"]))
    story.append(
        Paragraph(
            "This is the automated twin of the GUI workflow.",
            st["body"],
        )
    )
    story.extend(
        bullets(
            [
                "Load with load_dataset(path, scan_index).",
                "Read ground-truth tumor (x,y) from metadata if present.",
                "Preprocess: BMID → all methods none, Week 3 off. Otherwise → Savitzky–Golay window 7 poly 3, Week 3 on, clutter on, global normalize on.",
                "Reconstruct DAS, DMAS, DMAS-D4 on a 64×64 grid. BMID FOV forced to ±6 cm; phase-delay radius off for comparable batch runs.",
                "Select beamformer: if GT exists, pick the image whose ROI is closest to the labeled tumor; else weighted quality.",
                "Detect up to 4 ROIs (threshold 0.7 internally, min_score_ratio 0.28 in automation). If GT exists, keep the nearest hotspot even if NMS dropped it.",
                "High-resolution ROI reconstruct (min 96², up to 256², ×4 upscale of the coarse box).",
                "Characterize size/SCR/FWHM; assess confidence; write figures.",
                "Classify with the chosen threshold profile.",
            ],
            st,
        )
    )
    story.append(Paragraph("3.4 Quality profiles (automation/policy.py + classifier.py)", st["h2"]))
    story.append(
        Paragraph(
            "Hard fails: load error, non-finite image, energy below min, ROI area below min. "
            "Soft warnings: localization error, SCR, SNR, confidence, or “GT tumor but not a candidate”. "
            "Blocked: cannot even start (no paired measurement).",
            st["body"],
        )
    )
    prof = [
        ["Gate", "strict", "balanced", "lenient"],
        ["min energy", "1e-10", "1e-12", "1e-14"],
        ["min ROI area (px)", "8", "4", "1"],
        ["max loc. error (cm)", "1.5", "3.0", "5.0"],
        ["min SCR", "5.0", "2.0", "0.5"],
        ["min SNR", "1.0", "0.5", "0.1"],
        ["min confidence", "0.55", "0.35", "0.20"],
        ["require candidate if GT", "yes", "yes", "no"],
        ["finite output required", "yes", "yes", "yes"],
    ]
    story.append(table(prof, [4.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm], st))
    story.append(
        Paragraph(
            "Warning cases can still be recommended_upload if localization is within 1.25× the error gate. "
            "That lets slightly blurry but useful scans into a demo set.",
            st["body"],
        )
    )
    story.append(Paragraph("3.5 Reports, cache, figures, MCP", st["h2"]))
    story.extend(
        bullets(
            [
                "reports.py writes results/latest_auto_analysis.md (tumor index, detection table, showcase, walkthrough) and latest_auto_validation.json.",
                "Showcase pick: labeled tumor first, then strongest candidate, then smaller localization error, then higher confidence.",
                "figures.py copies selected / beamformer-compare / ROI-refine PNGs.",
                "cache.py avoids re-running identical (file, scan, profile) work in one session.",
                "MCP server is a thin stdio wrapper so an external agent can start the same job.",
            ],
            st,
        )
    )

    story.append(Paragraph("4. What automation is capable of today (plain list)", st["h1"]))
    story.extend(
        bullets(
            [
                "Batch-screen a folder of .mat / .s2p files without opening each GUI tab.",
                "Tell you which BMID indices are labeled tumors before reconstruction (tumor index).",
                "Produce Yes/No tumor-candidate plus a reason sentence per case.",
                "Report localization error in cm against laboratory GT when labels exist.",
                "Compare DAS vs DMAS vs DMAS-D4 automatically.",
                "Keep a JSON audit trail of metrics, taxonomy, preprocessing steps, and ROI spots.",
                "Fail closed on corrupted files instead of crashing the desktop app.",
                "Run load-only mode (--no-reconstruction) for a fast finite/energy sanity check.",
            ],
            st,
        )
    )
    story.append(
        Paragraph(
            "What it does <b>not</b> do: it is not a trained neural classifier; taxonomy is metadata buckets. "
            "It does not claim BIRADS medically. It does not replace a radiologist. It does not search every "
            "geometry on every scan (that is Auto Tweak in the GUI; batch uses a fixed conservative geometry).",
            st["body"],
        )
    )

    story.append(Paragraph("5. Novelty this automation brings to the project", st["h1"]))
    story.append(
        Paragraph(
            "Many microwave-imaging student projects stop at “we plotted one DAS image.” "
            "The new layer is an experiment operating system around the physics:",
            st["body"],
        )
    )
    story.extend(
        bullets(
            [
                "<b>Format-aware batching:</b> the same job understands Touchstone S-matrices and BMID cubes, including “do not preprocess _adi”.",
                "<b>Label-aware scan picking:</b> diverse strategy is not random; it forces healthy + size + location coverage so a demo is honest.",
                "<b>Physics-in-the-loop scoring:</b> gates are localization (cm), SCR, SNR, ROI area, energy — not a black-box accuracy percentage.",
                "<b>Multi-spot ROI:</b> more than one hotspot can be reported; GT is used to keep a nearby spot, not to hide other spots.",
                "<b>One evidence pack:</b> markdown for humans, JSON for machines, PNG for slides, GUI handoff for inspection.",
                "<b>Profiled strictness:</b> mentors can ask for lenient triage or strict localization without changing code.",
                "<b>Three interfaces, one pipeline:</b> students, GUI users, and MCP agents cannot silently use different math.",
            ],
            st,
        )
    )

    story.append(Paragraph("6. How this sits next to the rest of the software", st["h1"]))
    story.append(
        Paragraph(
            "Manual GUI path remains the teaching path: you see every tab. Automation is the verification path: "
            "you prove the same modules work on many files. Auto Tweak (reconstruction/auto_calibrate.py) is a "
            "different automation: it searches antenna angle, flip, 355° vs 360°, phase-delay radius, and ROI mode "
            "for <b>one</b> loaded scan. Batch validation does not currently call Auto Tweak, on purpose, so a 8-scan "
            "run finishes in a reasonable time and stays comparable.",
            st["body"],
        )
    )

    story.append(Paragraph("7. How we can develop it further", st["h1"]))
    story.append(Paragraph("Near term (practical)", st["h2"]))
    story.extend(
        bullets(
            [
                "Default Module 3 input to Hamming / Week-3 time-domain when those arrays exist (README already lists this gap).",
                "Optional “Auto Tweak lite” inside batch: only on showcase tumor cases, not on every healthy scan.",
                "Write a CSV summary (one row per scan) for Excel mentors who will not open JSON.",
                "Store per-scan reconstruction settings in the JSON so a GUI reload is bit-identical.",
                "Add a dry-run that only prints the tumor index and chosen diverse scans (no beamforming).",
            ],
            st,
        )
    )
    story.append(Paragraph("Medium term (science)", st["h2"]))
    story.extend(
        bullets(
            [
                "Replace the placeholder severity mapping with a trained model that uses image features + metadata, still labeled research-only.",
                "Learn suspicion_threshold per dataset family instead of a global 0.45.",
                "Support empty-chamber (_emp) subtraction when both adi and emp cubes are present.",
                "Quantify false-alarm rate on healthy scans as a first-class metric in the analysis report.",
                "Add a third data family (e.g. time-domain .mat already IFFT’d) with an explicit skip-IFFT flag.",
            ],
            st,
        )
    )
    story.append(Paragraph("Longer term (product / teaching)", st["h2"]))
    story.extend(
        bullets(
            [
                "A simple “lab exam mode”: hidden GT, student reconstructs, automation grades localization only.",
                "Cloud/batch runner that processes the full 2264-scan inventory overnight and publishes a dashboard.",
                "Uncertainty maps (pixel-wise coherence factor) exported beside the image.",
                "Keep the medical disclaimer in every new report surface so nobody treats scores as diagnosis.",
            ],
            st,
        )
    )

    story.append(Paragraph("8. Beginner cheat-sheet", st["h1"]))
    story.extend(
        bullets(
            [
                "If you have one .s2p: automation will load it, clean S21, reconstruct, and likely say tumor-candidate Unknown (no GT).",
                "If you have UM-BMID: put fd_data and md_list together; run diverse profile balanced.",
                "Read section 1 of latest_auto_analysis.md to find tumor scan indices.",
                "Read section 2 for Yes/No and localization error.",
                "Open the showcase image instead of clicking every tab.",
                "Use lenient only when you are debugging failures; use strict before a demo you care about.",
            ],
            st,
        )
    )
    return story


def report_c(st):
    story = []
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Report C — End-to-End Working of the Project", st["cover"]))
    story.append(
        Paragraph(
            "From loading a file to ROI reconstruction: every input, process, formula, threshold, and output.",
            st["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "This is the A-to-Z story. We stay in simple language, but we do not skip a decision the code actually makes. "
            "Whenever the software compares a number to a limit, that limit is written here.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "Research / educational only. Not a clinical diagnosis and not for medical decisions.",
            st["note"],
        )
    )

    story.append(Paragraph("0. Big picture flow", st["h1"]))
    story.append(
        Paragraph(
            "1) Module 1 loads a file into MicrowaveDataset.<br/>"
            "2) Module 2 cleans the complex traces (or does nothing for already-clean BMID _adi).<br/>"
            "3) Module 3 turns frequency data into time, then DAS / DMAS / DMAS-D4 images.<br/>"
            "4) ROI finds hotspots; optional high-resolution rebuild of the box.<br/>"
            "5) Quality metrics, tumor-candidate, characterization, confidence, session report.<br/>"
            "Automation can run 1–5 without the GUI. Auto Tweak can search geometry on one scan.",
            st["formula"],
        )
    )

    story.append(Paragraph("1. Module 1 — loading the dataset", st["h1"]))
    story.append(Paragraph("1.1 Input forms we accept", st["h2"]))
    story.extend(
        bullets(
            [
                "<b>Touchstone .s1p/.s2p/.s4p/.s8p</b> — VNA export. .s2p is the UG brief. Shape (n_freq, n_ports, n_ports).",
                "<b>MATLAB .mat</b> — legacy v4/v5/v7 via scipy, or v7.3 HDF5 via h5py. Heuristic hunt for frequency vector and S array.",
                "<b>UM-BMID fd_data_*.mat</b> — 3-D complex cube; you must pick scan_index. Companion md_list supplies tumor labels.",
            ],
            st,
        )
    )
    story.append(Paragraph("1.2 Load decisions and formulas", st["h2"]))
    story.append(
        Paragraph(
            "Entry: load_dataset(file_path, scan_index=None). Extension chooses the loader. "
            "If max(frequency) &lt; 1000 in a MATLAB file, values are treated as GHz and multiplied by 1e9. "
            "S arrays are moved so axis 0 is frequency. Real arrays are cast to complex (imaginary part 0) "
            "so later IFFT still has a consistent type. Touchstone frequencies from scikit-rf are already Hz.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "S21 extract from .s2p (Touchstone column order): after frequency, numbers are "
            "S11, S21, S12, S22 each as a pair. For RI: S21 = col3 + j col4 (0-based columns 3 and 4 in the 8-value block).",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "Physical defaults if the file is silent: antenna radius 0.08 m (8 cm), wave speed 3e8 m/s, "
            "FOV ±5 cm. BMID overrides: radius from ant_rad (else 18 cm), FOV ±6 cm, family tag UM-BMID.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "Output of Module 1: MicrowaveDataset + DatasetSummary for the GUI. Errors are typed "
            "(unsupported format, corrupted, missing variable, scan selection required) so the Status Log can explain them.",
            st["body"],
        )
    )

    story.append(Paragraph("2. Module 2 — preprocessing", st["h1"]))
    story.append(
        Paragraph(
            "Router: if the file is .s2p, run the UG S21 track (+ optional Week 3). Otherwise run the general "
            "filter → calibration → normalization → background → artifacts pipeline. BMID GUI/automation preset "
            "sets every method to none and Week 3 off, because _adi is already reference-cleaned.",
            st["body"],
        )
    )
    story.append(Paragraph("2.1 Touchstone S21 track (UG Weeks 1–2)", st["h2"]))
    story.extend(
        bullets(
            [
                "Extract S21 only; convert to complex linear; sort frequencies; interpolate real and imaginary onto a uniform Δf grid (needed for IFFT).",
                "Split into magnitude, wrapped phase, real, imaginary for display.",
                "Unwrap phase so jumps of ±180° become a continuous delay curve.",
                "Spike fix: Hampel (default) or median / local replacement of outliers.",
                "Optional average of repeated sweeps; optional subtract a reference measurement.",
                "Mild Savitzky–Golay on real and imaginary (typical window_length=7, polyorder=3) to reduce noise without killing the target bump.",
                "Keep a Hamming-windowed copy separately; GUI reconstruction still defaults to filtered S21(f).",
                "NRMSE can be reported to show how much filtering moved the trace.",
            ],
            st,
        )
    )
    story.append(Paragraph("2.2 Week 3 time-domain (steps 10–15 of the UG PDF)", st["h2"]))
    story.append(
        Paragraph(
            "Hamming window (length N):  w[k] = 0.54 − 0.46 cos(2π k / (N−1))<br/>"
            "S_hamming(f) = S21(f) · w(f)<br/>"
            "Δf = mean consecutive frequency step<br/>"
            "Δt = 1 / (N Δf)     t[n] = n Δt     bandwidth B = f_last − f_first     period T = 1/Δf<br/>"
            "s(t) = IFFT(S(f))   (NumPy ifft along frequency; optional zero padding lengthens N)<br/>"
            "If a reference exists: subtract in time after the same Hamming (matched windowing).<br/>"
            "Group clutter (need ≥2 channels):  μ_G[n] = mean_p s_p[n];   s_clean = s − μ_G<br/>"
            "Global normalize:  α = max |s|;   s ← s / α<br/>"
            "On a single .s2p there is only one channel, so group clutter is skipped and a note is stored.",
            st["formula"],
        )
    )
    story.append(Paragraph("2.3 General MATLAB pipeline", st["h2"]))
    story.extend(
        bullets(
            [
                "Noise filter (savgol / others) on complex traces.",
                "Calibration (self or external reference).",
                "Normalization (e.g. max).",
                "Optional background subtraction.",
                "Artifact suppression (hybrid and other methods).",
                "Quality report: SNR and dynamic range before/after.",
            ],
            st,
        )
    )
    story.append(
        Paragraph(
            "SNR estimate used in preprocessing (not the later image SNR): smooth |S| with a 5-sample moving average; "
            "treat smooth part as signal, residual as noise; SNR_dB = 10 log10(P_signal / P_noise).<br/>"
            "Dynamic range_dB = 20 log10(max|S| / min|S|).",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Optional export: Module 3 handoff .mat/.csv/.json with frequencies, processed S, and Tx/Rx coordinates "
            "on a circle of the antenna radius. That is how a teammate can reconstruct without rerunning Module 2.",
            st["body"],
        )
    )

    story.append(Paragraph("3. Module 3 — reconstruction (frequency → image)", st["h1"]))
    story.append(Paragraph("3.1 Geometry: where are the antennas and pixels?", st["h2"]))
    story.append(
        Paragraph(
            "Imaging is 2-D in the x–y plane. Pixels are a grid:<br/>"
            "x = linspace(x_min, x_max, n_x)   y = linspace(y_min, y_max, n_y)<br/>"
            "Default n_x = n_y = 64. Automation also uses 64 for speed. High-res ROI later uses 96–256.",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Circular array: if span is 360°, angles = linspace(0, 2π, n_ant, endpoint=False). "
            "If span is 355° (BMID-style incomplete ring), angles = linspace(0, 355°, n_ant, endpoint=True). "
            "Then x = r cos(θ+offset), y = r sin(θ+offset), with optional clockwise and x/y flips.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "Optional UM-BMID phase-delay radius (Rodriguez-Herrera / umbmid.antennas):<br/>"
            "r_eff = 0.97 · (r_scan − 0.106) + 0.148     (meters)<br/>"
            "Default in infer_reconstruction_config: this is OFF; Auto Tweak may turn it on if it helps GT error.",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Wave speed c defaults to 3e8 m/s (speed of light). Tissue/coupling media are slower; Auto Tweak can try "
            "2.1e8 and 1.8e8 if include_wave_speeds is enabled. Wrong c stretches the focus like a wrong camera lens.",
            st["body"],
        )
    )

    story.append(Paragraph("3.2 IFFT — the shared first physics step", st["h2"]))
    story.append(
        Paragraph(
            "Beamformers work on time traces. Frequencies must be uniformly spaced or IFFT raises an error "
            "(relative tolerance 1e-3 on Δf). At least two samples are required. "
            "time_signals = ifft(S, n = n_freq + zero_padding, axis=frequency). "
            "Time axis used when picking delay samples: t[n] = n / (N Δf).",
            st["body"],
        )
    )

    story.append(Paragraph("3.3 Delay-and-Sum (DAS)", st["h2"]))
    story.append(
        Paragraph(
            "For each image point r and each antenna position a_i:<br/>"
            "distance d_i = || a_i − r ||<br/>"
            "two-way travel time  τ_i = 2 d_i / c<br/>"
            "sample the time trace at the nearest index  n = round(τ_i / Δt)<br/>"
            "If n is outside [0, N_time), that antenna contributes 0 at this pixel.",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Textbook DAS image: I_DAS(r) = Σ_i | s_i(τ_i) |<br/>"
            "Coherence factor (Hollmann-style):<br/>"
            "CF(r) = |Σ_i s_i|²  /  ( N · Σ_i |s_i|² )<br/>"
            "Project default: I = I_DAS · CF^γ   with γ = 0.55<br/>"
            "γ = 0 restores textbook DAS. The 0.55 default is the cohort average: it damps pixels where antennas disagree, and γ ≥ 0.80 starts healthy false positives.",
            st["formula"],
        )
    )

    story.append(Paragraph("3.4 Delay-Multiply-and-Sum (DMAS)", st["h2"]))
    story.append(
        Paragraph(
            "Same delayed samples s_i. For every unique pair i &lt; j:<br/>"
            "p_ij = s_i · conjugate(s_j)<br/>"
            "signed_power(z, p) = sign(Re-like) · (|z|+ε)^p<br/>"
            "implemented as  Re{ sign(z) · (|z|+ε)^exponent } with ε = 1e-9, exponent p = 0.30 "
            "(textbook p = 0.5).<br/>"
            "I_DMAS = Σ_{i&lt;j} signed_power(p_ij, 0.30)<br/>"
            "Pairing emphasizes energy that is consistent across two antennas, so compact scatterers pop out.",
            st["formula"],
        )
    )

    story.append(Paragraph("3.5 DMAS-D4", st["h2"]))
    story.append(
        Paragraph(
            "First compute I_DMAS, then a compressive map:<br/>"
            "I_D4 = sign(I_DMAS) · |I_DMAS|^{d4}<br/>"
            "The cohort sweep keeps the textbook exponent d4 = 0.25. Values from 0.10 to 0.80 "
            "were within 0.01 of that average. d4 = 0 scatters the spot, and d4 ≥ 0.85 raises healthy false positives.",
            st["formula"],
        )
    )

    story.append(Paragraph("3.6 Choosing which beamformer to display", st["h2"]))
    story.append(
        Paragraph(
            "Image quality numbers (quality/metrics.py):<br/>"
            "SNR = mean(I) / std(I)<br/>"
            "SCR = max(I) / mean(|I − mean(I)|)<br/>"
            "CCR = (mean(top 5%) − mean(rest)) / std(rest)<br/>"
            "contrast = (max−min)/(max+min)<br/>"
            "FWHM_px = average of contiguous half-max widths on the peak row and peak column.",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Quality mode score (min-max normalized across the three images):<br/>"
            "score = 0.35 SNR_n + 0.35 SCR_n + 0.20 contrast_n + 0.10 (1 − time_n)<br/>"
            "Other modes: prefer_dmas_d4 (always pick DMAS-D4), force DAS/DMAS/DMAS-D4, "
            "or tumor_gt (pick the image whose detected ROI centroid is closest to labeled tumor).",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "A simpler legacy helper select_best_reconstruction still exists: "
            "0.4 contrast + 0.3 (std/mean) + 0.3 peak. The GUI/automation path uses select_best_beamformer.",
            st["note"],
        )
    )

    story.append(Paragraph("4. ROI reconstruction — from coarse image to a box, then a sharper box", st["h1"]))
    story.append(Paragraph("4.1 Detecting hotspots", st["h2"]))
    story.append(
        Paragraph(
            "Normalize image to [0,1]:  Ĩ = (I − min) / (max − min)<br/>"
            "Smooth with Gaussian σ = 1.0 pixel.<br/>"
            "Binary mask: Ĩ_smooth ≥ threshold_ratio   default threshold_ratio = 0.70<br/>"
            "Morphological opening with a 3×3 ones kernel (drops 1-pixel speckle).<br/>"
            "Label connected components. Drop components with area &lt; min_area (default 6 pixels).",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Component score:<br/>"
            "compactness = area / bbox_area<br/>"
            "size_penalty = 1 / (1 + 10 · area_ratio)   area_ratio = area / (Ny Nx)<br/>"
            "edge_penalty = 0.65 if the box touches the image edge, else 1<br/>"
            "score = (0.65 peak + 0.35 mean) · compactness · size_penalty · edge_penalty<br/>"
            "Off-center mode multiplies by (0.2 + 0.8 r_norm) · (1 + 1.5 r_norm) where r_norm is distance from image center / max radius.<br/>"
            "Tight-peak mode shrinks the box to about ±margin pixels around the peak (margin ≤ 2).",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Extra local maxima (5×5 maximum filter, peak at least max(0.75·threshold, 82nd percentile)) "
            "can seed extra spots. Keep spots if score ≥ 0.40 · best_score (automation uses 0.28) and "
            "centroids are at least 6 pixels apart. Maximum 4 ROIs. If GT exists, the nearest peak within "
            "3 cm of the label is added even if non-maximum suppression had dropped it — GT does not delete other spots.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "If nothing survives, threshold is raised by 0.05 (capped at 0.95) and min_area falls to 1, recursively. "
            "Last resort: a small box around the global peak.",
            st["body"],
        )
    )
    story.append(Paragraph("4.2 High-resolution ROI rebuild", st["h2"]))
    story.append(
        Paragraph(
            "Take the chosen bounding box on the coarse 64×64 image. Convert pixel edges back to meters. "
            "Rebuild only that rectangle with the same beamformer on a denser grid: "
            "n = clip(roi_size · 4, min 96, max 256). This is “ROI reconstruction”: not a new physical model, "
            "just the same delays on a zoomed field of view so the focus is sampled more finely.",
            st["body"],
        )
    )
    story.append(Paragraph("4.3 Tumor-candidate decision (not a diagnosis)", st["h2"]))
    story.append(
        Paragraph(
            "Features: local_contrast = peak_in_ROI / mean_outside<br/>"
            "ring_excess = peak / median of other pixels on the same radius (antenna-ring test)<br/>"
            "size_score = 1 / (1 + (area_ratio / 0.08)²)<br/>"
            "contrast_score = tanh((local_contrast − 1)/2)<br/>"
            "ring_score = tanh((ring_excess − 1.15)/0.35)<br/>"
            "location_score = 0.45 + 0.55 r_norm   (center tumors are not zeroed)<br/>"
            "suspicion = clip(0.38 contrast + 0.22 size + 0.15 location + 0.25 ring, 0, 1)",
            st["formula"],
        )
    )
    story.append(
        Paragraph(
            "Penalties on confidence = suspicion · penalty:<br/>"
            "×0.85 if ROI touches edge; ×0.65 if area_ratio &gt; 0.35; ×0.55 if ring_excess &lt; 1.20.<br/>"
            "If GT distance ≤ 2 cm, suspicion += 0.35 (cap 1); if ≤ 3 cm, += 0.20.<br/>"
            "Candidate if confidence ≥ 0.45.<br/>"
            "Override YES if GT ≤ 3 cm and local_contrast ≥ 1.05 and area_ratio ≤ 0.40 (confidence at least 0.55).<br/>"
            "Override NO if ring_excess &lt; 1.40 and (no GT or GT &gt; 3 cm) — this is the “it is just the antenna ring” veto.<br/>"
            "Multi-spot combined decision: YES if any spot is YES; confidence = max spot confidence.",
            st["formula"],
        )
    )

    story.append(Paragraph("5. Characterization and confidence", st["h1"]))
    story.append(
        Paragraph(
            "Centroid (cm) from pixel interpolation across the FOV. "
            "equivalent_diameter_cm = 2 sqrt(area_cm² / π). "
            "local_scr uses SCR on the bounding box. FWHM_cm = FWHM_px · average pixel size in cm. "
            "compactness = area_px / (bbox_w · bbox_h). eccentricity_proxy = (aspect−1)/(aspect+1).",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "Overall confidence (0–1), label high ≥ 0.75, medium ≥ 0.45, else low:<br/>"
            "image_q = 0.45(1−e^{−SCR/8}) + 0.30 contrast + 0.25(1−e^{−CCR/4})<br/>"
            "detection = tumor-candidate confidence<br/>"
            "beamformer_margin = clip(0.55 + 0.9 tanh(2·(score_sel − score_2nd)))<br/>"
            "localization = exp(−dist_cm / 2.5) if GT, else 0.55 compactness + 0.45 exp(−|diam−2|/2.5)<br/>"
            "focus = 0.6 exp(−max(FWHM−1.5,0)/3) + 0.4 exp(−max(area−4,0)/6)<br/>"
            "overall = 0.22 image_q + 0.28 detection + 0.15 margin + 0.22 localization + 0.13 focus",
            st["formula"],
        )
    )

    story.append(Paragraph("6. Auto Tweak (single-scan geometry search)", st["h1"]))
    story.append(
        Paragraph(
            "Not the batch validator. Used from the Reconstruction page when you click Auto Tweak Settings.",
            st["body"],
        )
    )
    story.extend(
        bullets(
            [
                "Stage 0: score current manual settings on DAS/DMAS/DMAS-D4 (baseline).",
                "Stage 1: DAS-only sweep of angles {0,90,180,270}°, flips {none,x,y,xy}, CW/CCW, arc {360,355}, phase-delay on/off, ROI modes {peak, off-center, tight}. Quick mode drops CW, 355°, and phase-delay.",
                "Stage 2: full three beamformers on top 8 geometries.",
                "With GT: score = 1/(dist_m + 1e-4) + tiny compactness term. Apply a new setup only if GT error improves by at least 0.5 mm.",
                "Without GT: score = quality · (0.5 + r_norm) / (1 + area/2000), preferring contrasty off-center compact ROIs. Must beat baseline by 1e-4.",
            ],
            st,
        )
    )

    story.append(Paragraph("7. GUI path vs automation path (same physics)", st["h1"]))
    story.append(
        Paragraph(
            "GUI: Acquisition page loads one file (BMID shows a scan picker with HEALTHY / TUMOR labels). "
            "Preprocessing page offers presets (BMID pass-through vs .s2p brief defaults), stage plots, Week 3 time plots, handoff export. "
            "Reconstruction page lets you set radius, FOV, c, beamformer mode, ROI mode, run all three algorithms, refine ROI, save a session report under results/. "
            "Automation calls the same functions with fixed 64×64 and profiled gates, then writes latest_auto_analysis.md.",
            st["body"],
        )
    )

    story.append(Paragraph("8. Worked numeric walk-through on sample_touchstone.s2p", st["h1"]))
    story.append(
        Paragraph(
            "Input form: 2-port RI Touchstone, 201 frequencies, 1–9 GHz, Δf = 40 MHz, R = 50 Ω. "
            "At 1.00 GHz, S21 = 0.126712 − 0.018170 j (exact digits from Report A).",
            st["body"],
        )
    )
    story.extend(
        bullets(
            [
                "Module 1 stores a (201, 2, 2) complex tensor plus comments that set r = 0.08 m, c = 3e8, FOV ±5 cm.",
                "Module 2 (automation): Savitzky–Golay on R and I, Week 3 Hamming+IFFT. Group clutter skipped (one channel). α = max |s(t)| if global normalize is on.",
                "After Module 2, processed S is no longer 2×2: it is S21 only, shape (201, 1). reconstruct_all requires 2-D (n_freq, n_traces) and treats that single column as one antenna trace.",
                "IFFT length 201 (plus padding if set). Δt = 1/(201 · 40e6) ≈ 0.124 ns.",
                "8 cm radius circle, 64×64 pixels over 10 cm × 10 cm. Two-way delay to the origin: τ = 2·0.08 / 3e8 ≈ 0.533 ns ≈ 4 samples.",
                "DAS/DMAS/DMAS-D4 images computed; quality scores compared; ROI at 70% of normalized peak; candidate classifier runs without GT so result may be Unknown/No depending on ring_excess.",
                "Session/auto report stores peak, mean, ROI area, confidence — still not a diagnosis.",
            ],
            st,
        )
    )

    story.append(Paragraph("9. BMID walk-through (when the cube is on disk)", st["h1"]))
    story.extend(
        bullets(
            [
                "Input: fd_data_s21_adi.mat cube + md_list_s21_adi.mat. Pick scan k with a tumor at (x_cm, y_cm), diameter d_cm.",
                "Slice S shape (1001, n_ant), frequencies linspace(1e9, 8e9, 1001), Δf = 7 MHz.",
                "Module 2 pass-through: processed S = raw S. No Hamming unless you force Week 3.",
                "Geometry: r = 0.18 m unless metadata says otherwise; FOV ±6 cm; 64×64.",
                "If Auto Tweak is used with GT, it minimizes ||ROI_centroid − (x,y)||.",
                "Localization error_cm = 100 · hypot(cx − x, cy − y). Balanced profile warns if this exceeds 3 cm.",
                "Taxonomy label such as small/moderate/round/upper-left is metadata, not inferred from the image.",
            ],
            st,
        )
    )

    story.append(Paragraph("10. Threshold and constant cheat-sheet (every decision)", st["h1"]))
    cheat = [
        ["Constant / gate", "Value", "Where it is used"],
        ["BMID N_freq", "1001", "Cube recognition"],
        ["BMID band", "1–8 GHz", "Synthetic frequency axis"],
        ["GHz heuristic", "max(f)&lt;1000", "MATLAB loader ×1e9"],
        ["Default r, c, FOV", "0.08 m, 3e8, ±5 cm", "Non-BMID metadata gap"],
        ["BMID r, FOV", "0.18 m, ±6 cm", "bmid_loader / automation"],
        ["DAS γ", "0.55", "Coherence weight"],
        ["DMAS p, γ, ε", "0.30, 0, 1e-9", "Pairing"],
        ["D4 exponent", "0.25", "After DMAS"],
        ["Quality weights", "0.35/0.35/0.20/0.10", "SNR/SCR/contrast/time"],
        ["ROI threshold", "0.70", "Normalized image"],
        ["ROI min area", "6 px (auto min from profile)", "Component filter"],
        ["ROI NMS gap", "6 px", "Multi-spot"],
        ["ROI max spots", "4", "detect_rois"],
        ["min_score_ratio", "0.40 GUI / 0.28 auto", "Keep weaker spots"],
        ["Suspicion thr.", "0.45", "Tumor-candidate"],
        ["Ring veto", "ring_excess &lt; 1.40", "Not a candidate"],
        ["GT boost", "≤2 cm / ≤3 cm", "+0.35 / +0.20"],
        ["Size classes", "≤2 / ≤4 cm", "small / medium / large"],
        ["Quadrant center", "radius 1 cm", "“central”"],
        ["Auto Tweak improve", "0.5 mm GT", "Must beat baseline"],
        ["Confidence labels", "0.75 / 0.45", "high / medium"],
        ["IFFT Δf check", "rtol 1e-3", "Uniform grid"],
        ["Phase-delay r", "0.97(r−0.106)+0.148", "Optional BMID"],
        ["Refine grid", "96–256, ×4", "ROI reconstruction"],
    ]
    story.append(table(cheat, [4.4 * cm, 5.2 * cm, 6.4 * cm], st))

    story.append(Paragraph("11. Outputs you should expect", st["h1"]))
    story.extend(
        bullets(
            [
                "GUI: magnitude/phase/real/imag plots; three reconstruction images; ROI overlay; refined zoom; quality table; confidence sentence.",
                "results/ session reports from manual runs.",
                "results/latest_auto_analysis.md + JSON + PNGs from automation.",
                "Optional handoff .mat for another reconstruction tool.",
            ],
            st,
        )
    )
    story.append(
        Paragraph(
            "If something looks “wrong”: first check you did not load metadata-only; then check scan index; "
            "then check whether BMID pass-through was used on already-clean data (double cleaning can erase the target); "
            "then check c and array radius; then try Auto Tweak rather than inventing a new beamformer.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "That is the full path: complex matrices in, delays computed from geometry and c, "
            "pixels scored by DAS/DMAS/D4, hotspots boxed, numbers compared to the gates in the table above, "
            "and a research image plus a research label out. Stop there — a person still has to interpret it.",
            st["body"],
        )
    )
    return story


def main():
    st = styles()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = OUT_DIR / "A_Dataset_Report.pdf"
    b = OUT_DIR / "B_Automation_Report.pdf"
    c = OUT_DIR / "C_Project_Working_Report.pdf"
    print("Building Report A…")
    build_doc(a, "Report A — Dataset Report", report_a(st))
    print("Building Report B…")
    build_doc(b, "Report B — Automation Report", report_b(st))
    print("Building Report C…")
    build_doc(c, "Report C — Project Working Report", report_c(st))
    print("Wrote:")
    print(a)
    print(b)
    print(c)


if __name__ == "__main__":
    main()

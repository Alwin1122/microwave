"""Plain-language PDF report of the final, fair tumor classification results.

Reads results/classification/final_summary.json, geometry_calibration.json,
background_removal_comparison.json and features_twin_gen1_gen3.npz, and writes
results/tumor_classification_report.pdf.

Usage:
    python tools/write_classification_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.literature import PUBLISHED  # noqa: E402
from classification_walkthrough import append_steps  # noqa: E402

OUT_DIR = PROJECT / "results" / "classification"
REPORTS = PROJECT / "results" / "reports" / "classification"
METHODS_PATH = REPORTS / "how_classification_works.pdf"
GUIDE_PATH = REPORTS / "guide_progress.pdf"
NAVY = colors.HexColor("#1b365d")
FEATURE_LABELS = {
    "signal": "signal (empty chamber removed)",
    "signal_rot": "signal (empty chamber + same-from-every-angle part removed)",
    "breast": "breast time window (0.71–5.71 ns)",
    "image": "image",
    "all": "signal + image",
}
VARIANT_LABELS = {
    "rot": "Empty chamber + same-from-every-angle part removed, 1-8 GHz",
    "svd1": "Remove the strongest shared pattern",
    "svd2": "Remove the 2 strongest shared patterns",
    "svd3": "Remove the 3 strongest shared patterns",
    "rot_gate0.5": "First row + cut echoes before 0.5 ns (skin)",
    "rot_gate1": "First row + cut echoes before 1 ns",
    "rot_gate1.5": "First row + cut echoes before 1.5 ns",
    "rot_band1-4": "First row, low band only (1-4 GHz)",
    "rot_band2-6": "First row, middle band only (2-6 GHz)",
    "rot_band4-8": "First row, high band only (4-8 GHz)",
}


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def cm_text(value: float) -> str:
    return f"{value:.1f} cm"


def score(value: float) -> str:
    return f"{100 * value:.0f} / 100"


def mean(block: dict, key: str) -> float:
    return block["nested"][key]["mean"]


def spread(block: dict, key: str) -> str:
    return f"{100 * block['nested'][key]['mean']:.0f} &plusmn; {100 * block['nested'][key]['std']:.0f}"


def alarm(block: dict) -> str:
    return f"{pct(mean(block, 'tuned_sensitivity'))} found / {pct(mean(block, 'tuned_specificity'))} cleared"


def most_chosen(block: dict) -> str:
    counts = block["chosen_candidates"]
    fs, model, pool = max(counts, key=counts.get).split("|")
    pool_text = "fair scans only" if pool == "fair" else "fair scans plus all other scans"
    return f"{FEATURE_LABELS[fs]}, {model.replace('_', ' ')}, trained on {pool_text}"


def load() -> dict:
    data = {
        "final": json.loads((OUT_DIR / "final_summary.json").read_text(encoding="utf-8")),
        "geometry": json.loads((OUT_DIR / "geometry_calibration.json").read_text(encoding="utf-8")),
        "background": json.loads((OUT_DIR / "background_removal_comparison.json").read_text(encoding="utf-8")),
    }
    math_path = OUT_DIR / "math_analysis.json"
    data["math"] = json.loads(math_path.read_text(encoding="utf-8")) if math_path.exists() else None
    twin = np.load(OUT_DIR / "features_twin_upto3_signal.npz", allow_pickle=False)
    names = [str(n) for n in twin["feature_names"]]
    change = twin["X"][:, names.index("sig_cal_to_raw_log")]
    gen3 = twin["generation"] == "gen3"
    healthy = gen3 & (twin["has_tumor"] == 0)
    tumor = gen3 & (twin["has_tumor"] == 1)
    diam = twin["tum_diam_cm"].astype(float)
    data["dose"] = [("Healthy (repeat scan)", float(np.median(10 ** change[healthy])))] + [
        (f"Tumour {d:g} cm", float(np.median(10 ** change[tumor & (diam == d)]))) for d in sorted(set(diam[tumor]))
    ]
    return data


def bar_chart(bars, ylabel, path, ref=None, ref_label=None, ylim=None) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars])
    top = max(b[1] for b in bars)
    for i, (_, value, _) in enumerate(bars):
        ax.text(i, value + top * 0.02, f"{value:.0f}" if top > 10 else f"{value:.1f}", ha="center", fontsize=9)
    if ref is not None:
        ax.axhline(ref, color="#555555", linestyle="--", linewidth=1, label=ref_label)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylim(0, ylim or top * 1.18)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def detection_chart(final: dict) -> Path:
    det = final["main"]["detection"]
    bars = [
        ("Current app\nchecker", 100 * det["rule_based"]["auc"], "#8fa8c8"),
        ("Recording\ndetails only", 100 * det["setup_only"]["auc"]["mean"], "#c0504d"),
        ("Single scan\n(fair)", 100 * mean(det, "auc"), "#2e8b57"),
    ]
    if "twin_baseline" in final:
        bars.append(("Compared with an\nearlier healthy scan", 100 * mean(final["twin_baseline"]["detection"], "auc"), "#1f6f43"))
    if "reference_gen3_fat_scan" in final:
        bars.append(("gen3 with a\nfat-only scan", 100 * mean(final["reference_gen3_fat_scan"]["detection"], "auc"), "#6b8e7f"))
    return bar_chart(bars, "Tumour-vs-healthy score (/100)", OUT_DIR / "report_detection.png", 50, "coin flip = 50", 108)


def position_chart(geometry: dict, background: dict) -> Path:
    gens = [g for g in ("gen1", "gen2", "gen3") if g in geometry["generations"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    labels = ["App as it was", "Guess the centre", "New image, 1-8 GHz", "New image, best band"]
    shades = ["#c0504d", "#8fa8c8", "#7fbf9a", "#2e8b57"]
    width = 0.2
    for g, gen in enumerate(gens):
        block = geometry["generations"][gen]
        values = [
            block["app_default_cm"],
            block["centre_guess_cm"],
            block["modes"]["emp_rot"]["held_out_mean_cm"],
            background[gen]["held_out_mean_cm"],
        ]
        for k, value in enumerate(values):
            x = g + (k - 1.5) * width
            ax.bar(x, value, width, color=shades[k], label=labels[k] if g == 0 else None)
            ax.text(x, value + 0.08, f"{value:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels(gens)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_ylabel("Average miss (cm), lower is better")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "report_position.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write(data: dict) -> None:
    final, geometry, background = data["final"], data["geometry"], data["background"]
    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5, leading=14.5, alignment=TA_LEFT, spaceAfter=7)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, textColor=NAVY, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, textColor=NAVY, spaceBefore=6, spaceAfter=6)
    note = ParagraphStyle("n", parent=body, fontSize=9, leading=12, textColor=colors.HexColor("#555555"))
    h3 = ParagraphStyle("h3", parent=h2, fontName="Helvetica-Bold", fontSize=12, spaceBefore=8, spaceAfter=4)
    formula = ParagraphStyle("f", parent=body, fontName="Courier", fontSize=8.2, leading=11, backColor=colors.HexColor("#f4f7fb"), borderPadding=5, spaceAfter=7)
    io = ParagraphStyle("io", parent=body, fontSize=9.5, leading=13, backColor=colors.HexColor("#f4f7fb"), borderPadding=5, spaceAfter=7)
    bullet = ParagraphStyle("bu", parent=body, fontSize=10.5, leading=14, leftIndent=8, spaceAfter=3)
    cell = ParagraphStyle("c", parent=body, fontSize=9, leading=11.5, spaceAfter=0)
    head = ParagraphStyle("hd", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)

    def table(rows, widths, highlight=()):
        cells = [[Paragraph(str(v), head if r == 0 else cell) for v in row] for r, row in enumerate(rows)]
        tbl = Table(cells, colWidths=widths, repeatRows=1)
        cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d0de")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        cmds += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#d9efe4")) for r in highlight]
        tbl.setStyle(TableStyle(cmds))
        return tbl

    main = final["main"]
    twin = final["twin_baseline"]
    ref = final.get("reference_gen3_fat_scan")
    det, size, diam, pos = main["detection"], main["size_class"], main["diameter"], main["position"]
    tdet, tsize, tdiam = twin["detection"], twin["size_class"], twin["diameter"]
    g3 = geometry["generations"]["gen3"]
    bg3 = background["gen3"]
    best_variant = bg3["chosen_per_half"][0]
    reps = final["protocol"]["repeats"]

    story = [
        Paragraph("How tumour classification works", h1),
        Paragraph(
            "Internal process, cut-offs and formulas, in plain language, each one tied to the measured result. "
            "Research and educational use only, not a medical diagnosis. "
            "The short results sheet for a project guide is guide_progress.pdf in the same folder.",
            note,
        ),
        Paragraph("1. The short answer", h2),
        Paragraph(
            "The program was asked: <b>is there a tumour</b>, <b>how big is it</b> and <b>where is it</b>. Every number comes from "
            "phantoms the program had never seen, in recording sessions where the session itself gives no hint. "
            "Two situations were tested: a <b>single scan</b> (a first-ever visit) and a <b>follow-up scan compared with earlier "
            "healthy scans of the same breast</b> (monitoring).",
            body,
        ),
        table(
            [
                ["Question", "Program", "Recording details only", "Simple guessing"],
                ["Tumour? single scan (score /100)", score(mean(det, "auc")), score(det["setup_only"]["auc"]["mean"]), "50 / 100"],
                ["Tumour? compared with earlier healthy scan", f"<b>{score(mean(tdet, 'auc'))}</b>", score(tdet["setup_only"]["auc"]["mean"]), "50 / 100"],
                ["Alarm, single scan", alarm(det), "-", "-"],
                ["Alarm, compared with earlier scan", alarm(tdet), "-", "-"],
                ["Size class, single scan", pct(mean(size, "balanced_accuracy")), pct(size["setup_only"]["balanced_accuracy"]["mean"]), pct(size["chance_balanced_accuracy"])],
                ["Size class, compared with earlier scan", pct(mean(tsize, "balanced_accuracy")), pct(tsize["setup_only"]["balanced_accuracy"]["mean"]), pct(tsize["chance_balanced_accuracy"])],
                ["Where? gen3 image spot (no training)", f"<b>{cm_text(bg3['held_out_mean_cm'])}</b>", "-", cm_text(bg3["centre_guess_cm"]) + " (the centre)"],
            ],
            [5.8 * cm, 3.6 * cm, 3.4 * cm, 4.6 * cm],
            highlight=(2, 7),
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "\"/100\": pick one tumour scan and one healthy scan at random; how often does the program give the tumour scan the higher "
            "score? 50 = coin flip, 100 = perfect. \"Alarm\": with the yes/no cut-off chosen on training phantoms only, the share of "
            "tumours found and of healthy scans correctly cleared. \"Recording details only\": a model that sees just session, "
            "antenna distance and generation - how much could be guessed without looking at the scan.",
            note,
        ),
        Paragraph(
            f"<b>The official first-scan score is {score(mean(det, 'auc'))}.</b> "
            "That number is the nested test in section 3: the program was allowed to choose features, model and training pool "
            "on training phantoms only, then scored once. A later side check used only the signal features and only the fair training scans, "
            "and that narrower check scored about 69. It did not replace this result. Section 9 is that side check.",
            body,
        ),
        Paragraph("2. How the test was kept fair", h2),
    ]
    for text in (
        "<b>No session shortcut.</b> Tumour and healthy phantoms were mostly scanned in different sessions, and the machine drifts. "
        "Only sessions with both kinds were used for testing, each trimmed to equal tumour and healthy counts "
        f"({main['n_fair']} single-scan test scans from {main['n_fair_phantoms']} phantoms). Recording details alone score about 50.",
        "<b>Never the same phantom in training and test.</b>",
        "<b>No picking the best of many tries.</b> Feature set, model type, training data and the alarm cut-off were chosen on the "
        f"training phantoms only, then tested once (\"nested\" testing), repeated {reps} times with different splits.",
        "<b>No help from the answer.</b> Tumour positions are never given to the program. Settings for the image (antenna layout, "
        "background removal) were chosen on one half of the phantoms and checked on the other half.",
        "<b>Earlier healthy scans are picked blindly.</b> Every scan, tumour or healthy, is compared with the same number of "
        "<i>other</i> healthy scans of the same phantom from the same session: as many as the scan that has the fewest, and at most 3. "
        "A tumour is never given a cleaner baseline than the healthy scans, so the count cannot give away the answer. "
        "Scans with only one earlier healthy scan are included.",
    ):
        story.append(Paragraph("&bull; " + text, body))

    story += [
        PageBreak(),
        Paragraph("3. Is there a tumour? Single scan", h2),
        Image(str(detection_chart(final)), width=17 * cm, height=7.2 * cm),
        table(
            [
                ["Single scan (fair test scans)", "Result"],
                ["Score /100 (all / gen1 / gen3)", f"{spread(det, 'auc')} / {100 * mean(det, 'auc_gen1'):.0f} / {100 * mean(det, 'auc_gen3'):.0f}"],
                ["Alarm with the default cut-off (0.5)", f"{pct(mean(det, 'sensitivity'))} found / {pct(mean(det, 'specificity'))} cleared"],
                ["Alarm with the cut-off chosen on training phantoms", alarm(det)],
                ["Current app checker without training", score(det["rule_based"]["auc"])],
                ["Choice the program made most often", most_chosen(det)],
            ],
            [7.4 * cm, 10.0 * cm],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "From a single scan the tumour signal is real but weak. Choosing the cut-off properly balances the alarms "
            "but cannot add information that is not in the scan.",
            body,
        ),
        Paragraph("4. Comparing with an earlier healthy scan (monitoring)", h2),
        Paragraph(
            "The biggest problem is the background: skin, fat and inner tissue echo much more strongly than a tumour. "
            "If an earlier healthy scan of the same breast exists, subtracting it removes almost all of that background, "
            "leaving only what changed. Gen3 has several healthy scans of each phantom in the same session as its tumour scans, "
            "taken alternately. Gen1 is included too, down to phantoms that have only one healthy scan in that sitting. "
            "Both kinds of scan are handled the same way.",
            body,
        ),
        Paragraph("<b>Is the change really the tumour?</b> Bigger tumours should change the scan more; handling noise would not care "
                  "about tumour size. Median amount of change in gen3 (relative to the scan's total signal):", body),
        table(
            [["Scan", "Median change"]] + [[name, f"{value:.1e}"] for name, value in data["dose"]],
            [8.0 * cm, 5.0 * cm],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "The change grows steadily with tumour size, so it is caused by the tumour. It also shows the limit of this system: "
            "a 1 cm tumour changes the scan no more than repeating a healthy scan does.",
            body,
        ),
        table(
            [
                [f"Compared with earlier healthy scan ({twin['n_fair']} test scans, {twin['n_fair_phantoms']} phantoms)", "Program", "Recording details only", "Guessing"],
                ["Tumour? (score /100)", spread(tdet, "auc"), score(tdet["setup_only"]["auc"]["mean"]), "50"],
                ["Tumour? older / newest / 2 cm or smaller", f"{100 * mean(tdet, 'auc_gen1'):.0f} / {100 * mean(tdet, 'auc_gen3'):.0f} / {100 * mean(tdet, 'auc_small'):.0f}", "-", "-"],
                ["Alarm (cut-off chosen on training phantoms)", alarm(tdet), "-", "-"],
                ["Size class (balanced accuracy, three sizes)", pct(mean(tsize, "balanced_accuracy")), pct(tsize["setup_only"]["balanced_accuracy"]["mean"]), pct(tsize["chance_balanced_accuracy"])],
                ["Size class, newest phantoms only", pct(mean(tsize, "accuracy_gen3")), "-", "-"],
                ["Diameter, average error", cm_text(mean(tdiam, "mae_cm")), cm_text(tdiam["setup_only"]["mae_cm"]["mean"]), cm_text(tdiam["average_guess_mae_cm"])],
                ["Diameter, newest phantoms only", cm_text(mean(tdiam, "mae_cm_gen3")), "-", "-"],
            ],
            [6.6 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "<b>Catch:</b> this answers \"did something new appear since the last healthy scan?\", not \"is there a tumour in a "
            "first-ever scan?\". It needs a previous healthy scan of the same person in the same position. The difference image "
            "did not locate the tumour better, because small shifts of the inner tissue between scans also show up in it.",
            body,
        ),
        PageBreak(),
        Paragraph("5. How big is it? (single scan)", h2),
        table(
            [
                ["Tumour scans of the fair set", "Program", "Recording details only", "Guessing"],
                ["Size class: small (&le;2 cm) / medium (&le;4 cm) / large", pct(mean(size, "balanced_accuracy")), pct(size["setup_only"]["balanced_accuracy"]["mean"]), pct(size["chance_balanced_accuracy"])],
                ["Diameter, average error", cm_text(mean(diam, "mae_cm")), cm_text(diam["setup_only"]["mae_cm"]["mean"]), cm_text(diam["average_guess_mae_cm"])],
            ],
            [7.0 * cm, 2.8 * cm, 4.0 * cm, 3.6 * cm],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "Most of the size skill comes from knowing the generation (gen1 used 2, 4 and 6 cm tumours, gen3 used 1-3 cm). "
            "The scan itself adds only a few points.",
            body,
        ),
        Paragraph("6. Where is it? The image step", h2),
        Paragraph(
            "The app's original image missed the tumour by more than simply guessing the centre. Three changes fixed it for gen3: "
            "remove the part of the signal that looks the same from every antenna angle (skin, fat shell, chamber); "
            "add the echoes up with their wave phase (coherent delay-and-sum); and fit the antenna layout per generation. "
            "Then several further background-removal ideas were compared (gen3, average miss; settings chosen on one half of the phantoms):",
            body,
        ),
        table(
            [["Background removal (gen3)", "Average miss", "Within 2 cm"]]
            + [[VARIANT_LABELS.get(k, k), cm_text(v), pct(bg3["within_2cm"][k])] for k, v in bg3["mean_miss_cm"].items()],
            [10.4 * cm, 3.4 * cm, 3.6 * cm],
            highlight=(list(bg3["mean_miss_cm"]).index(best_variant) + 1,),
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            f"Using only the low frequencies (1-4 GHz) worked best and was picked by both halves; on the held-out half it misses by "
            f"{cm_text(bg3['held_out_mean_cm'])} against {cm_text(bg3['centre_guess_cm'])} for the centre. "
            "Removing the strongest shared patterns removed the tumour as well, and cutting the early skin echo changed nothing.",
            body,
        ),
        Image(str(position_chart(geometry, background)), width=17 * cm, height=7.0 * cm),
        Paragraph(
            "In gen1 and gen2 nothing beat guessing the centre: their dense inner tissue sits off-centre and outshines the tumour. "
            "Gen3 has only four tumour positions, so part of its skill is picking the right quarter.",
            body,
        ),
        table(
            [
                ["Position (all fair single-scan tumours)", "Average miss", "Within 2 cm"],
                ["Guess the centre", cm_text(pos["centre_guess"]["mean_cm"]), pct(pos["centre_guess"]["within_2cm"])],
                ["<b>Image spot, no training (official)</b>", cm_text(pos["image_spot"]["mean_cm"]), pct(pos["image_spot"]["within_2cm"])],
                ["Trained program", cm_text(mean(pos, "mean_cm")), pct(mean(pos, "within_2cm"))],
                ["Recording details only (warning sign)", cm_text(pos["setup_only"]["mean_cm"]["mean"]), pct(pos["setup_only"]["within_2cm"]["mean"])],
            ],
            [8.0 * cm, 4.6 * cm, 4.8 * cm],
            highlight=(2,),
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "<b>Why the trained program is not used for position:</b> within a session the tumour sat in the same spot, so a model "
            "that sees only the session places it about as well. The image spot never sees labels or sessions, so it is the honest result.",
            body,
        ),
    ]

    if ref:
        rd, rs, rdm = ref["detection"], ref["size_class"], ref["diameter"]
        story += [
            Paragraph("7. Best case for a single scan: a fat-only scan (gen3)", h2),
            Paragraph(
                "Every gen3 phantom was also scanned with only its fat layer. A real patient never has such a scan, so this is a "
                "ceiling for single-scan background removal, not a usable method.",
                body,
            ),
            table(
                [
                    [f"gen3, {ref['n_fair']} scans, {ref['n_fair_phantoms']} phantoms", "With fat-only scan"],
                    ["Tumour? (score /100)", spread(rd, "auc")],
                    ["Alarm", alarm(rd)],
                    ["Size class (balanced accuracy)", pct(mean(rs, "balanced_accuracy"))],
                    ["Diameter average error", f"{cm_text(mean(rdm, 'mae_cm'))} (recording details only: {cm_text(rdm['setup_only']['mae_cm']['mean'])})"],
                ],
                [9.0 * cm, 8.4 * cm],
            ),
            Spacer(1, 0.2 * cm),
        ]

    story.append(Paragraph("8. What this means", h2))
    for text in (
        "<b>Single first-time scan:</b> tumours are detected better than chance but with many false alarms; not good enough for decisions. "
        "Subtracting an average of other breasts that share the same fat shell made this worse, so a first visit still uses the empty chamber only.",
        "<b>Follow-up against an earlier healthy scan:</b> this is the result to use. Small tumours (2 cm or less) score about 97/100, "
        "the newest phantoms about 99/100, and the older phantoms about 86/100.",
        "<b>Location:</b> works in gen3 with the new image steps (best with 1-4 GHz); not in gen1/gen2.",
        "<b>Size:</b> on a follow-up the newest phantoms are sized well. The older phantoms are not, and a 1 cm tumour still changes the raw scan about as much as repeating a healthy scan.",
        "<b>Generation 2</b> cannot be added. Its tumour scans and healthy scans were taken in different sittings, and the sitting number alone gives the answer away.",
    ):
        story.append(Paragraph("&bull; " + text, body))
    story.extend(math_sections(data.get("math"), body, h2, note, table))
    story.append(Paragraph("12. Limits", h2))
    for text in (
        f"Few phantoms ({main['n_fair_phantoms']} single-scan, {twin['n_fair_phantoms']} follow-up), so numbers can move by several points (see &plusmn;).",
        "Follow-up uses 46 older-phantom scans and 200 newest-phantom scans. Generation 2 is excluded on purpose.",
        "Large (6 cm) tumours exist only in gen1; gen3 has only four tumour positions.",
        "The gen3 antenna layout was fitted on tumour positions; the gen3 image-spot figure uses the half-and-half check, the "
        "all-tumours figure uses the final layout and may be slightly flattering.",
        "Size and diameter may borrow a little from the recording session, as position did; the \"recording details only\" column shows that ceiling.",
        "No benign vs malignant labels exist in this dataset. \"BI-RADS\" in the metadata is breast density, not cancer risk. "
        "Criticality in this report is the Born volume index defined in section 10, not a clinical category.",
    ):
        story.append(Paragraph("&bull; " + text, body))
    story.append(PageBreak())
    story.append(Paragraph(
        "The sections above are the measured results. The steps below are the internal process, in the order the code runs, "
        "with the formula and the number that formula produced.",
        body,
    ))
    step_styles = {"h1": h1, "h2": h2, "h3": h3, "body": body, "formula": formula, "io": io, "note": note, "bullet": bullet}
    append_steps(story, step_styles, lambda text, styles, key="body": Paragraph(text, styles[key]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Full numbers: results/classification/final_summary.json, math_analysis.json, geometry_calibration.json and background_removal_comparison.json.", note))

    REPORTS.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(METHODS_PATH), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="How tumour classification works",
    )
    doc.build(story)
    write_guide(data, body, h1, h2, note, table)


def _pm(block: dict, key: str, scale: float = 100.0) -> str:
    item = block.get(key) or {}
    mean = item.get("mean") if isinstance(item, dict) else None
    if mean is None:
        return "n/a"
    spread = item.get("std") or 0.0
    return f"{scale * mean:.0f} &plusmn; {scale * spread:.0f}"


def math_sections(math: dict | None, body, h2, note, table) -> list:
    """Physics layer, type/criticality, and the published-study comparison."""
    story = [
        PageBreak(),
        Paragraph("9. Mathematical analysis of a first visit (side check, not a new official score)", h2),
        Paragraph(
            "The official first-scan score stays the nested result in section 3. "
            "This section asks a different question: if the same forest is given only the signal samples, "
            "does adding the formulas below raise the score? The signal-only run of this check is about 69, "
            "because it is not allowed the image features or the extra training scans that the nested test used.",
            body,
        ),
        Paragraph(
            "A first visit has no earlier healthy scan, so the skin and fat shell cannot be subtracted. "
            "After the empty-chamber scan is removed, the part that looks the same from every antenna is subtracted. "
            "What remains is summarised by three formulas, and those formulas are handed to the same forest that already reads the time samples.",
            body,
        ),
        Paragraph(
            "<b>Scattering spectrum.</b> The breast-window signals form a matrix B, one column per antenna. "
            "Its singular values s give L_i = s_i^2 / (sum of s_j^2). L_1 is the share of energy in the strongest pattern, "
            "and the effective rank is 1 / (sum of L_i^2). A compact inclusion raises L_1. Tissue that fills the breast spreads the energy.",
            body,
        ),
        Paragraph(
            "<b>Frequency slope.</b> p(f) is the average leftover magnitude. Over 2–6 GHz the fit is "
            "log10 p(f) = a + b log10(f / 1 GHz). A small sphere in the Born approximation has b near 2, because the "
            "scattered field grows with frequency squared. A larger inclusion pulls b down. b does not depend on how hard "
            "the antennas were driven, so generation 1 and generation 3 can share it.",
            body,
        ),
        Paragraph(
            "<b>Compactness.</b> Excess kurtosis of the energy-weighted delay is high for a short echo and low for a smear. "
            "The angular resultant length R is high when one antenna direction carries the leftover.",
            body,
        ),
        Paragraph(
            "<b>Where, without training.</b> On the brightest leftover antenna, d = R_antenna - c t_peak / 2, "
            "and the antenna angle supplies the direction. This is the radar round-trip, not a fitted image.",
            body,
        ),
    ]
    if not math:
        story.append(Paragraph("The phantom-grouped comparison file has not been written yet (tools/eval_scattering_math.py).", note))
        return story

    det = math["first_visit"]["detection_random_forest"]
    base, plus = det["baseline"], det["with_physics"]
    only = math["first_visit"]["physics_only_logistic"]
    follow = math["followup"]["detection_logistic_on_rotational_signal"]
    size = math["first_visit"]["size_class_logistic"]
    crit = math["first_visit"]["criticality_logistic"]
    follow_crit = math["followup"]["criticality_logistic"]
    story += [
        Paragraph(
            "The comparison below was fixed before looking at the result: the same model, the same phantom-grouped folds "
            f"(5 folds, 3 seeds, {math['first_visit']['n_fair']} fair first-visit scans), with and without the physics features. "
            "No other model was tried.",
            body,
        ),
        table(
            [
                ["First visit", "Signal only", "Signal + physics", "Physics formulas only"],
                ["Tumour score /100", _pm(base, "auc"), _pm(plus, "auc"), _pm(only, "auc")],
                ["Older phantoms (gen1)", _pm(base, "auc_gen1"), _pm(plus, "auc_gen1"), _pm(only, "auc_gen1")],
                ["Newest phantoms (gen3)", _pm(base, "auc_gen3"), _pm(plus, "auc_gen3"), _pm(only, "auc_gen3")],
                ["2 cm or smaller", _pm(base, "auc_small"), _pm(plus, "auc_small"), _pm(only, "auc_small")],
                ["Found / cleared at the 0.5 cut", f"{_pm(base, 'sensitivity')} / {_pm(base, 'specificity')}", f"{_pm(plus, 'sensitivity')} / {_pm(plus, 'specificity')}", f"{_pm(only, 'sensitivity')} / {_pm(only, 'specificity')}"],
            ],
            [4.2 * cm, 4.2 * cm, 4.4 * cm, 4.6 * cm],
        ),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "The formulas do not improve the first visit. Signal samples alone score "
            f"{_pm(base, 'auc')}; adding the formulas moves that to {_pm(plus, 'auc')}. "
            f"The formulas with no signal samples score {_pm(only, 'auc')}, a coin flip. "
            "On a first visit the tumour does not change the scattering spectrum, the frequency slope or the compactness "
            "more than healthy fibroglandular tissue does. Comparing the bright antennas with the quiet ones inside the same scan, "
            "and the group-delay roughness of the brightest antenna, also sat near 52/100, and that check saw every fair scan at once. "
            "The weak first-visit result still lives in the individual time samples, "
            "which a 15-number summary throws away. Newest phantoms stay ahead of older ones "
            f"({_pm(plus, 'auc_gen3')} against {_pm(plus, 'auc_gen1')}).",
            body,
        ),
        Paragraph(
            "Follow-up keeps the published logistic model on the rotationally cleaned difference signal. "
            "Physics features are computed on that same difference, so a gain on the first visit is not allowed to spoil the follow-up.",
            body,
        ),
        table(
            [
                [f"Follow-up ({math['followup']['n_fair']} fair scans)", "Difference signal", "Difference + physics"],
                ["Tumour score /100", _pm(follow["baseline"], "auc"), _pm(follow["with_physics"], "auc")],
                ["Older / newest phantoms", f"{_pm(follow['baseline'], 'auc_gen1')} / {_pm(follow['baseline'], 'auc_gen3')}", f"{_pm(follow['with_physics'], 'auc_gen1')} / {_pm(follow['with_physics'], 'auc_gen3')}"],
                ["2 cm or smaller", _pm(follow["baseline"], "auc_small"), _pm(follow["with_physics"], "auc_small")],
            ],
            [6.0 * cm, 5.5 * cm, 5.9 * cm],
        ),
        Spacer(1, 0.2 * cm),
    ]
    weights = math.get("physics_logistic_coefficients") or {}
    names = weights.get("features") or []
    means = weights.get("mean") or []
    if names and means:
        order = sorted(range(len(names)), key=lambda i: abs(means[i]), reverse=True)[:6]
        story.append(Paragraph(
            "On the physics-only logistic model, a positive weight raises the tumour score. "
            "The six largest standardised weights, averaged over the training folds, are: "
            + "; ".join(f"{names[i].replace('math_', '')} {means[i]:+.2f}" for i in order)
            + ".",
            body,
        ))
    ident = math.get("identification") or {}
    physics = ident.get("physics") or {}
    image = ident.get("image_spot_same_scans") or {}
    if physics.get("n"):
        by_gen = ident.get("physics_by_generation") or {}
        gen_text = ", ".join(
            f"{gen} {by_gen[gen].get('mean_cm', float('nan')):.1f} cm" for gen in ("gen1", "gen3") if gen in by_gen and by_gen[gen].get("n")
        )
        story.append(Paragraph(
            f"Closed-form range and bearing, on the same fair tumour scans: average miss {physics['mean_cm']:.1f} cm "
            f"({100 * physics['within_2cm']:.0f}% within 2 cm). {gen_text}. "
            f"The trained-free image spot on those scans misses by {image.get('mean_cm', float('nan')):.1f} cm. "
            "The formula is the identification baseline; the image remains the localiser where it already works.",
            body,
        ))

    story += [
        Paragraph("10. Type and criticality", h2),
        Paragraph(
            "The dataset records shape (sphere or spiculated) and diameter. It does not record benign versus malignant, "
            "and its BI-RADS number is breast density. Type here means that shape. Criticality is a research index from the Born volume law:",
            body,
        ),
        Paragraph("C = (d / 2 cm)^3, multiplied by 1.5 when the shape is spiculated.", body),
        Paragraph(
            "A 2 cm sphere has C = 1 (low). A 2 cm spiculated inclusion has C = 1.5 (moderate). A 4 cm sphere has C = 8 (still moderate). "
            "Anything above 8 is high. Healthy scans are not scored. "
            f"Only {math['first_visit'].get('n_spiculated_fair', 0)} of {math['first_visit'].get('n_fair_tumour', 0)} "
            "fair tumour scans are spiculated, all in the older phantoms, so a separate shape classifier is not reported: "
            "the sample cannot support one. Criticality is reported because it is defined for every tumour scan.",
            body,
        ),
        table(
            [
                ["Balanced accuracy (/100)", "Signal only", "Signal + physics", "Chance"],
                ["Size class, first visit", _pm(size["baseline"], "balanced_accuracy"), _pm(size["with_physics"], "balanced_accuracy"), "33"],
                ["Criticality, first visit", _pm(crit["baseline"], "balanced_accuracy"), _pm(crit["with_physics"], "balanced_accuracy"), "33"],
                ["Criticality, follow-up", _pm(follow_crit["baseline"], "balanced_accuracy"), _pm(follow_crit["with_physics"], "balanced_accuracy"), "33"],
            ],
            [5.5 * cm, 4.0 * cm, 4.5 * cm, 3.4 * cm],
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "Balanced accuracy gives each class the same weight, so a model cannot look good by always saying \"small\" or \"low\". "
            "Chance for three classes is 33. "
            f"Follow-up criticality is uneven: older phantoms {_pm(follow_crit['with_physics'], 'accuracy_gen1')}, "
            f"newest phantoms {_pm(follow_crit['with_physics'], 'accuracy_gen3')} correct. "
            "The balanced accuracy is the number for the whole follow-up set.",
            note,
        ),
        Paragraph("11. Comparison with published studies", h2),
        Paragraph(
            "These papers answer related questions on different data and under different rules. "
            "The point of the table is where this study adds a result they did not report: a first visit and a follow-up "
            "kept separate, a phantom that is never in both training and test, and a criticality index on measured scans "
            "rather than on a simulated tumour response.",
            body,
        ),
    ]
    rows = [["Study", "What they reported", "What was different"]]
    for item in PUBLISHED:
        rows.append([item["study"], f"{item['task']}: {item['result']}", item["restriction"]])
    first_auc = _pm(plus, "auc")
    follow_auc = _pm(follow["with_physics"], "auc")
    rows.append([
        "This study",
        f"First visit, signal + physics, ROC score {first_auc}/100. Follow-up {follow_auc}/100. Criticality balanced accuracy {_pm(crit['with_physics'], 'balanced_accuracy')}.",
        "Experimental UM-BMID. Fair sessions only. Same phantom never in training and test. Not a clinical benign/malignant label.",
    ])
    story.append(table(rows, [4.4 * cm, 6.6 * cm, 6.4 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "Reimer and Pistorius (IEEE J-ERM, 2022) remain the strict published detection number on this dataset: "
        "a convolutional network reached 78 &plusmn; 3 under leave-one-fat-shell-out, and 90 &plusmn; 3 only when the test breast was inside the training volume and the tumour sat at antenna height. "
        "Their logistic regression did not generalise under that split. The EuCAP 2020 accuracy of 85 &plusmn; 4 was a subset and is not the same test. "
        "Davis et al. (2008) and Conceição et al. (2010) show that size and shape can be read from a radar signature when the tumour response is simulated; "
        "those accuracies are not available as a target on UM-BMID, which has skin, fat, fibroglandular tissue and only six spiculated tumours in the fair set. "
        "Image-based microwave studies reviewed in 2023 report sensitivity from 63 to 100 percent, but specificity, when it is reported at all, is 20 to 65 percent.",
        body,
    ))
    story.append(Paragraph(
        "This study does not beat the 2022 network on the network's own split, and the new formulas do not beat the signal forest on a first visit. "
        "What it adds is the split between a first visit and a follow-up in which both classes are given the same number of earlier healthy scans, "
        "the follow-up score near 98/100 under that rule, a checked statement that the Born slope and the scattering spectrum are clutter on a first visit, "
        "and a criticality index that can be audited because the diameter and the shape are known. "
        "On these measured scans that reading is only partly available: criticality clears chance, and a three-way size label does not.",
        body,
    ))
    return story


def write_guide(data: dict, body, h1, h2, note, table) -> None:
    """Short results sheet for a project guide. The long process note is the other PDF."""
    final = data["final"]
    main = final["main"]
    twin = final["twin_baseline"]
    det, size, diam = main["detection"], main["size_class"], main["diameter"]
    tdet, tsize, tdiam = twin["detection"], twin["size_class"], twin["diameter"]
    math = data.get("math") or {}
    side = (math.get("first_visit") or {}).get("detection_random_forest") or {}
    crit = ((math.get("first_visit") or {}).get("criticality_logistic") or {}).get("with_physics") or {}
    story = [
        Paragraph("Tumour classification — results and progress", h1),
        Paragraph(
            "Sheet for a project guide. Research scores on UM-BMID phantoms, not a clinical diagnosis. "
            "Formulas, cut-offs and the step-by-step test are in how_classification_works.pdf.",
            note,
        ),
        Paragraph("1. Where the work stands", h2),
        Paragraph(
            f"A first scan, with no earlier healthy scan of the same breast, scores <b>{score(mean(det, 'auc'))}</b> "
            f"({spread(det, 'auc')} across {final['protocol']['repeats']} nested repeats). "
            f"A follow-up that subtracts earlier healthy scans of the same breast scores <b>{score(mean(tdet, 'auc'))}</b>. "
            "The first-scan number has not been lowered. A later side check scored about 69 because it was a smaller test: "
            "signal features only, and training scans from the fair set only. The official model still uses signal plus image, "
            "and it may train on every other scan of phantoms that are not in the test fold.",
            body,
        ),
        table(
            [
                ["Question", "Official result", "What was compared"],
                ["Tumour on a first scan", f"<b>{spread(det, 'auc')}</b>", f"Recording details only {score(det['setup_only']['auc']['mean'])}; chance 50"],
                ["First scan, older / newest phantoms", f"{100 * mean(det, 'auc_gen1'):.0f} / {100 * mean(det, 'auc_gen3'):.0f}", "Same nested test, split by generation"],
                ["First-scan alarm (cut-off from training phantoms)", alarm(det), "Default 0.5 cut is less balanced"],
                ["Tumour on a follow-up", f"<b>{spread(tdet, 'auc')}</b>", f"Older {100 * mean(tdet, 'auc_gen1'):.0f}, newest {100 * mean(tdet, 'auc_gen3'):.0f}, ≤2 cm {100 * mean(tdet, 'auc_small'):.0f}"],
                ["Follow-up alarm", alarm(tdet), "Same number of earlier healthy scans for both classes, at most 3"],
                ["Size class, first scan / follow-up", f"{pct(mean(size, 'balanced_accuracy'))} / {pct(mean(tsize, 'balanced_accuracy'))}", "Chance 33%. Cuts at 2 cm and 4 cm"],
                ["Diameter error, first scan / follow-up", f"{cm_text(mean(diam, 'mae_cm'))} / {cm_text(mean(tdiam, 'mae_cm'))}", f"Newest follow-up phantoms {cm_text(mean(tdiam, 'mae_cm_gen3'))}"],
                ["Side check: signal only, then signal + formulas", f"{_pm(side.get('baseline') or {}, 'auc')} then {_pm(side.get('with_physics') or {}, 'auc')}", "Does not replace the official first-scan row"],
                ["Criticality, first scan (volume index)", _pm(crit, "balanced_accuracy") if crit else "n/a", "Chance 33. Not clinical BI-RADS"],
            ],
            [6.2 * cm, 5.4 * cm, 5.8 * cm],
            highlight=(1, 4),
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("2. Progress since the reconstruction-only stage", h2),
    ]
    for text in (
        "Detection and characterisation now have a fair test: equal tumour and healthy counts inside a session, and no phantom in both training and test.",
        f"The saved first-scan model is {most_chosen(det)}. Its yes/no cut-off is the Youden point (highest true-positive rate minus false-positive rate) on training phantoms only.",
        f"The saved follow-up model is {most_chosen(tdet)}. The reference is the mean of other healthy scans from the same sitting, and the tumour is never given more of those scans than the healthy cases.",
        "Things tried for the first scan that did not raise the official score: pasting a measured tumour echo onto other training breasts, subtracting an average fat shell from other breasts, and adding the scattering-spectrum and frequency-slope formulas. The follow-up score stayed near 98 when the formulas were added.",
        "A convolutional network on the breast-window sinogram scored 50/100 on this fair test, and a PCA-then-logistic classifier on that sinogram scored 45/100. The network fit the training breasts and did not carry that fit to a new phantom. Neither was adopted. Details are in how_classification_works.pdf, Step 59.",
        "Type, meaning sphere versus spiculated, is not reported as a classifier. Only 6 of the fair tumour scans are spiculated. Criticality is reported instead, from diameter and that shape flag.",
        "Closed-form radar location (round-trip delay of the brightest antenna) misses by more than the coherent image spot, so position stays with the image.",
    ):
        story.append(Paragraph("&bull; " + text, body))
    story.append(Paragraph("3. Comparison with published studies", h2))
    story.append(Paragraph(
        "Protocols differ, so this is context, not a ranking. Reimer and Pistorius, IEEE J-ERM 2022, report a CNN at 78 ± 3 "
        "on a leave-one-fat-shell-out split of this dataset, and 90 ± 3 only when the test breast is inside the training volume "
        "and the tumour is at antenna height. Their logistic regression did not generalise on that split. "
        "The EuCAP 2020 accuracy of 85 ± 4 was a subset. Davis et al. 2008 and Conceição et al. 2010 report size and shape "
        "accuracies above 70–97 percent on simulated tumour echoes, without a measured skin and fat shell. "
        "The 2023 review gives image-based sensitivity 63–100 percent and specificity 20–65 percent in the few studies that stated it.",
        body,
    ))
    story.append(Paragraph(
        f"What those papers do not report is the follow-up number under a matched reference count: {score(mean(tdet, 'auc'))}. "
        "The first-scan number here is the stricter nested score, not a claim to have beaten the 2022 network on its own split.",
        body,
    ))
    story.append(Paragraph("4. Limits to state with the numbers", h2))
    for text in (
        f"{main['n_fair_phantoms']} phantoms in the first-scan test and {twin['n_fair_phantoms']} in the follow-up test. The ± is the spread across repeats.",
        "No benign versus malignant label. Metadata BI-RADS is breast density. Criticality is a research volume index.",
        "Generation 2 is excluded: tumour and healthy scans were taken in different sittings, and the sitting number alone gives the answer.",
        "Large 6 cm tumours exist only in the older phantoms. The newest phantoms have four tumour positions.",
    ):
        story.append(Paragraph("&bull; " + text, body))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Numbers: results/classification/final_summary.json. Process note: results/reports/classification/how_classification_works.pdf.",
        note,
    ))
    REPORTS.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(GUIDE_PATH), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title="Tumour classification — results and progress",
    )
    doc.build(story)


def main() -> None:
    write(load())
    print(f"saved {METHODS_PATH}")
    print(f"saved {GUIDE_PATH}")


if __name__ == "__main__":
    main()

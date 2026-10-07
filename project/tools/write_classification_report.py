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

OUT_DIR = PROJECT / "results" / "classification"
PDF_PATH = PROJECT / "results" / "tumor_classification_report.pdf"
NAVY = colors.HexColor("#1b365d")
FEATURE_LABELS = {
    "signal": "signal (empty chamber removed)",
    "signal_rot": "signal (empty chamber + same-from-every-angle part removed)",
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
    twin = np.load(OUT_DIR / "features_twin_gen1_gen3.npz", allow_pickle=False)
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
        Paragraph("Tumour detection and characterisation - final fair results", h1),
        Paragraph("Plain-language summary. Research and educational use only, not a medical diagnosis.", note),
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
        "<b>Earlier healthy scans are picked blindly.</b> Every scan, tumour or healthy, is compared with 3 randomly chosen "
        "<i>other</i> healthy scans of the same phantom from the same session, so how the baseline is chosen says nothing about the answer.",
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
            "leaving only what changed. In this dataset gen3 (and part of gen1) has several healthy scans of each phantom in the "
            "same session as its tumour scans, taken alternately (tumour, healthy, tumour, healthy...), so both kinds are handled "
            "the same way between scans.",
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
                ["Alarm (cut-off chosen on training phantoms)", alarm(tdet), "-", "-"],
                ["Size class (balanced accuracy)", pct(mean(tsize, "balanced_accuracy")), pct(tsize["setup_only"]["balanced_accuracy"]["mean"]), pct(tsize["chance_balanced_accuracy"])],
                ["Diameter, average error", cm_text(mean(tdiam, "mae_cm")), cm_text(tdiam["setup_only"]["mae_cm"]["mean"]), cm_text(tdiam["average_guess_mae_cm"])],
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
        "<b>Single first-time scan:</b> tumours are detected better than chance but with many false alarms; not good enough for decisions.",
        "<b>Follow-up against an earlier healthy scan:</b> clearly better, because the background cancels. This is the most promising use of this system.",
        "<b>Location:</b> works in gen3 with the new image steps (best with 1-4 GHz); not in gen1/gen2.",
        "<b>Size:</b> tumours of 1 cm are below what this system can see; larger tumours change the signal clearly.",
        "<b>Biggest remaining fix is more data:</b> more phantoms recorded with tumour and healthy scans in the same sessions.",
    ):
        story.append(Paragraph("&bull; " + text, body))
    story.append(Paragraph("9. Limits", h2))
    for text in (
        f"Few phantoms ({main['n_fair_phantoms']} single-scan, {twin['n_fair_phantoms']} follow-up), so numbers can move by several points (see &plusmn;).",
        "The follow-up test relies mostly on gen3; gen1 has only a few tumour scans with same-session healthy repeats.",
        "Large (6 cm) tumours exist only in gen1; gen3 has only four tumour positions.",
        "The gen3 antenna layout was fitted on tumour positions; the gen3 image-spot figure uses the half-and-half check, the "
        "all-tumours figure uses the final layout and may be slightly flattering.",
        "Size and diameter may borrow a little from the recording session, as position did; the \"recording details only\" column shows that ceiling.",
        "No benign vs malignant labels exist in this dataset. \"BI-RADS\" here means breast density, not cancer risk.",
    ):
        story.append(Paragraph("&bull; " + text, body))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Full numbers: results/classification/final_summary.json, geometry_calibration.json and background_removal_comparison.json.", note))

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="Tumour detection and characterisation - final fair results",
    )
    doc.build(story)


def main() -> None:
    write(load())
    print(f"saved {PDF_PATH}")


if __name__ == "__main__":
    main()

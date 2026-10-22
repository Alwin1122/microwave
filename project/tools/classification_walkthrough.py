"""Step-by-step tumour-classification chapter shared by two reports.

Report C (project working) and how_classification_works.pdf both call
append_steps so the formulas and the measured scores cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SUMMARY = PROJECT / "results" / "classification" / "final_summary.json"
MATH = PROJECT / "results" / "classification" / "math_analysis.json"


def _load() -> tuple[dict, dict]:
    final = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    math = json.loads(MATH.read_text(encoding="utf-8")) if MATH.exists() else {}
    return final, math


def _spread(block: dict, key: str) -> str:
    item = block["nested"][key]
    return f"{100 * item['mean']:.0f} +/- {100 * item['std']:.0f}"


def _one(block: dict, key: str) -> str:
    return f"{100 * block['nested'][key]['mean']:.0f}"


def _cm(block: dict, key: str) -> str:
    return f"{block['nested'][key]['mean']:.1f} cm"


def append_steps(story: list, st: dict, P) -> None:
    """Append Steps 45-60. ``P(text, st, style_key)`` builds a paragraph."""
    final, math = _load()
    main = final.get("main") or {}
    twin = final.get("twin_baseline") or {}
    det = main.get("detection") or {}
    tdet = twin.get("detection") or {}
    size = main.get("size_class") or {}
    tsize = twin.get("size_class") or {}
    diam = main.get("diameter") or {}
    tdiam = twin.get("diameter") or {}
    pos = main.get("position") or {}
    if not det or not tdet:
        story.append(P("Classification scores were not found in results/classification/final_summary.json.", st, "note"))
        return

    first = _spread(det, "auc")
    follow = _spread(tdet, "auc")
    side = ((math.get("first_visit") or {}).get("detection_random_forest") or {})
    side_base = ((side.get("baseline") or {}).get("auc") or {}).get("mean")
    side_plus = ((side.get("with_physics") or {}).get("auc") or {}).get("mean")
    side_only = (((math.get("first_visit") or {}).get("physics_only_logistic") or {}).get("auc") or {}).get("mean")
    crit_block = ((math.get("first_visit") or {}).get("criticality_logistic") or {})
    crit = ((crit_block.get("baseline") or {}).get("balanced_accuracy") or {})
    crit_plus = ((crit_block.get("with_physics") or {}).get("balanced_accuracy") or {})
    ident = math.get("identification") or {}

    story.append(P("Module — Tumour classification (the learned reading)", st, "h1"))
    story.append(P(
        "Steps 1-44 turn one file into a picture and a rule-based YES or NO (confidence at least 0.45). "
        "That rule does not learn from other phantoms. On the fair UM-BMID test it scores about "
        f"{100 * det['rule_based']['auc']:.0f}/100, which is a coin flip. "
        "This module is the separate learned reading. It asks three questions of phantoms it has never seen: "
        "is there a tumour, how big is it, and where is it. "
        f"Official first-scan score: <b>{first}/100</b>. Official follow-up score: <b>{follow}/100</b>. "
        "Neither number is a medical diagnosis.",
        st,
    ))

    story.append(P("Step 45. Two situations, two subtractions", st, "h3"))
    story.append(P(
        "<b>Input:</b> one UM-BMID scan S(f, antenna), shape (1001, n_antennas), "
        "f = linspace(1 GHz, 8 GHz, 1001), plus the empty-chamber scan of that sitting.<br/>"
        "<b>First visit</b> has no earlier healthy scan of the same breast. "
        "Only the empty chamber is removed.<br/>"
        "<b>Follow-up</b> also subtracts earlier healthy scans of the same phantom and the same sitting.",
        st, "io",
    ))
    story.append(P(
        "S_cal = S - S_empty<br/>"
        "S_rot = S_cal - mean_over_antennas(S_cal)<br/>"
        "S_follow = S_cal - mean(S_healthy_1 ... S_healthy_k)",
        st, "formula",
    ))
    story.append(P(
        "The rotational mean is the part that looks the same from every antenna: skin, fat shell, chamber. "
        "A compact inclusion does not look the same from every angle, so it survives in S_rot. "
        "S_follow is stronger when a previous healthy scan exists, because that scan carries the fibroglandular tissue as well as the shell. "
        f"That is why the follow-up score is {follow}/100 and the first-scan score is {first}/100.",
        st,
    ))

    story.append(P("Step 46. Breast time window", st, "h3"))
    story.append(P(
        "The frequency samples are turned into time with an IFFT. The delay of sample n is t_n = n / (N * df). "
        "Echoes before 0.71 ns are the antenna. Echoes after 5.71 ns are past the breast. "
        "This is the same window used by the published logistic-regression baseline on this dataset.",
        st,
    ))
    story.append(P(
        "dt = 1 / (N * mean(diff(f)))<br/>"
        "keep samples with 0.71e-9 &lt;= t &lt;= 5.71e-9<br/>"
        "about 36 time samples on the 1-8 GHz, 1001-point grid",
        st, "formula",
    ))

    story.append(P("Step 47. Signal features (prefix sig_ and rsig_)", st, "h3"))
    story.append(P(
        "<b>Input:</b> S_cal for sig_, and S_rot for rsig_. No tumour coordinate is used.<br/>"
        "<b>Output:</b> one number per formula below. The forest and the logistic model see only these numbers, not the raw matrix.",
        st, "io",
    ))
    story.append(P(
        "cal_to_raw = log10( sum|S_cal|^2 / sum|S|^2 )<br/>"
        "rot_residual = sum|S - mean_antenna|^2 / sum|S|^2<br/>"
        "For each 1 GHz band from 1-2 through 7-8:<br/>"
        "  band_log = log10(mean power)<br/>"
        "  band_cv = std(power per antenna) / mean(power)<br/>"
        "  band_rot = rotational energy in that band / band energy<br/>"
        "For each kept time sample k:<br/>"
        "  w_log = log10(mean over antennas of |s(t_k)|^2)<br/>"
        "  w_cv = std / mean of that time sample<br/>"
        "Angular harmonics h1..h4 = |FFT of antenna power|[k] / |FFT|[0]<br/>"
        "ant_cv = std(antenna energy) / mean(antenna energy)<br/>"
        "ant_max_over_median = max(antenna energy) / median(antenna energy)",
        st, "formula",
    ))
    story.append(P(
        "sig_ is the empty-chamber-cleaned scan. rsig_ is the same list after the rotational mean is removed. "
        "The breast-only candidate used later keeps just the w_ numbers (the time samples inside 0.71-5.71 ns) "
        "and drops the frequency-band numbers, because those bands still contain the antenna reflection.",
        st,
    ))

    story.append(P("Step 48. Image features (prefix img_ and imgl_)", st, "h3"))
    story.append(P(
        "Coherent delay-and-sum is run on S_rot. The grid is 64 by 64 over +/- 6 cm. "
        "Wave speed and antenna layout are the fitted values for that generation "
        "(gen1: 2.7e8 m/s, gen3: 2.1e8 m/s). imgl_ uses only 1-4 GHz, which was the band that localised gen3 best. "
        "DMAS-D4 features are stored but are not allowed into the model search: they were built before a baseband fix.",
        st,
    ))
    story.append(P(
        "peak_over_mean = max(image) / mean(image)<br/>"
        "ccr, FWHM_cm, entropy, top-1-percent / median<br/>"
        "peak (x, y) of a lightly smoothed image, and its radius<br/>"
        "From the brightest blind ROI (no ground truth):<br/>"
        "  equivalent diameter = 2 * sqrt(area / pi)<br/>"
        "  aspect, compactness = area / bounding box<br/>"
        "  eccentricity = (aspect - 1) / (aspect + 1)<br/>"
        "  local contrast, ring excess, suspicion, rule confidence<br/>"
        "The rule confidence is a feature. It is not the classification answer.",
        st, "formula",
    ))

    story.append(P("Step 49. Physics summaries (the side check)", st, "h3"))
    story.append(P(
        "These 15 numbers describe the rotational residual with formulas, instead of handing every time sample to the forest. "
        "They were tested after the official model. They did not replace it.",
        st,
    ))
    story.append(P(
        "Breast-window matrix B (time by antenna), singular values s_i<br/>"
        "L_i = s_i^2 / sum(s_j^2)<br/>"
        "effective rank = 1 / sum(L_i^2)<br/>"
        "On 2-6 GHz, p(f) = mean over antennas of |S_rot|<br/>"
        "log10 p(f) = a + b * log10(f / 1 GHz)    (ordinary least squares)<br/>"
        "A small sphere in the Born approximation has b near 2.<br/>"
        "Excess kurtosis of the energy-weighted delay<br/>"
        "Angular resultant R = |sum p_k exp(i theta_k)| / sum p_k<br/>"
        "Radar range of the brightest antenna:<br/>"
        "d = R_antenna - c * t_peak / 2",
        st, "formula",
    ))
    if side_base is not None and side_plus is not None and side_only is not None:
        story.append(P(
            f"<b>Measured on the same fair first-visit scans.</b> "
            f"Signal features alone in this narrower test: {100 * side_base:.0f}/100. "
            f"Signal plus these formulas: {100 * side_plus:.0f}/100. "
            f"Formulas alone: {100 * side_only:.0f}/100. "
            f"The official nested first-scan score remains {first}/100, because that test is allowed the image features "
            "and the extra training scans. On a first visit the tumour does not move these summaries more than healthy fibroglandular tissue does.",
            st,
        ))

    story.append(P("Step 50. The fair test set", st, "h3"))
    story.append(P(
        f"<b>Why.</b> Tumour scans and healthy scans were often taken in different sittings, and the hardware drifts. "
        f"A model can look clever by recognising the sitting. "
        f"<b>Rule.</b> Keep only sittings that contain both classes. Inside each sitting, draw the same number of tumour scans "
        f"and healthy scans (the smaller of the two counts, random seed 0). "
        f"<b>Resulting test:</b> {main.get('n_fair', '?')} first-scan scans from {main.get('n_fair_phantoms', '?')} phantoms "
        f"({main.get('n_fair_tumor', '?')} tumour). "
        f"Follow-up: {twin.get('n_fair', '?')} scans from {twin.get('n_fair_phantoms', '?')} phantoms. "
        "Generation 2 is left out of the follow-up test: its tumour scans and healthy scans were different sittings, "
        "so the sitting number alone gives the answer.",
        st,
    ))
    setup = det.get("setup_only", {}).get("auc", {}).get("mean")
    twin_setup = tdet.get("setup_only", {}).get("auc", {}).get("mean")
    if setup is not None and twin_setup is not None:
        story.append(P(
            "A second model sees only antenna radius, sitting number and generation. "
            f"On the first-scan test it scores {100 * setup:.0f}/100. "
            f"On the follow-up test it scores {100 * twin_setup:.0f}/100. "
            "Both must stay near 50. If they do not, the sitting is leaking the label.",
            st,
        ))

    story.append(P("Step 51. Phantom-grouped nested testing", st, "h3"))
    story.append(P(
        "<b>Input:</b> the fair scans, grouped by phantom name.<br/>"
        "<b>Technique:</b> StratifiedGroupKFold. A phantom is never in both the training fold and the test fold. "
        "Outer loop: 5 folds. Inner loop, on the training phantoms only: 4 folds. Repeated 3 times with different seeds.<br/>"
        "<b>What is chosen inside the inner loop, and nowhere else:</b> which feature family, which model, which training pool, and the alarm cut-off.",
        st, "io",
    ))
    story.append(P(
        "Feature families:<br/>"
        "  signal = sig_<br/>"
        "  signal_rot = rsig_<br/>"
        "  breast = time-window samples only (sig_w and rsig_w)<br/>"
        "  image = img_ and imgl_<br/>"
        "  all = signal + signal_rot + image<br/>"
        "Models: logistic, or a forest of 400 trees<br/>"
        "Pools: fair scans only, or fair scans plus every other scan of phantoms that are not in the test fold<br/>"
        "The test fold is scored once, with the choice the inner loop made. "
        "The reported +/- is the spread of that score across the 3 repeats.",
        st, "formula",
    ))
    story.append(P(
        f"<b>What the inner loop picked most often.</b> "
        f"First scan: {det.get('final_choice', 'signal + image, random forest, fair plus other scans')}. "
        f"Follow-up: {tdet.get('final_choice', 'rsig_, logistic, fair scans')}. "
        "The saved models in results/classification/ are those choices, fit afterwards on all training phantoms. "
        "The scores above are the nested test, not the fit on every phantom.",
        st,
    ))

    story.append(P("Step 52. The two learners, written out", st, "h3"))
    story.append(P(
        "Logistic, used for the official follow-up:<br/>"
        "  z_j = (x_j - mean_j) / std_j     (means and stds from the training phantoms)<br/>"
        "  p = 1 / (1 + exp(-(b0 + sum b_j z_j)))<br/>"
        "  penalty: inverse strength C = 0.5, class_weight = balanced<br/>"
        "  balanced means each class is given the same total weight, so a larger class cannot dominate<br/><br/>"
        "Random forest, used for the official first scan:<br/>"
        "  400 trees, a leaf must hold at least 3 scans<br/>"
        "  class_weight = balanced, random_state = 0<br/>"
        "  the score is the fraction of trees that vote tumour<br/><br/>"
        "Diameter uses ridge regression (penalty alpha = 3) or a forest regressor "
        "(400 trees, min leaf 3, max_features 0.33). "
        "Size class uses the same logistic or forest, with three labels.",
        st, "formula",
    ))

    story.append(P("Step 53. The yes/no cut-off", st, "h3"))
    story.append(P(
        "The model outputs a probability p. A scan is called tumour when p is at least t. "
        "t = 0.5 is reported, but it is not the official alarm. "
        "The official t is chosen on the training phantoms only, at the point where "
        "the true-positive rate minus the false-positive rate is largest (the Youden point).",
        st,
    ))
    story.append(P(
        "t = argmax over thresholds of (TPR - FPR)<br/>"
        "TPR = tumours found / all tumours<br/>"
        "FPR = healthy scans called tumour / all healthy scans<br/>"
        "sensitivity = TPR, specificity = 1 - FPR",
        st, "formula",
    ))
    thr = det.get("final_threshold")
    tthr = tdet.get("final_threshold")
    story.append(P(
        f"Saved first-scan cut-off: {thr:.3f}. "
        f"At that cut the nested test finds {_one(det, 'tuned_sensitivity')}% of tumours and clears {_one(det, 'tuned_specificity')}% of healthy scans. "
        f"At the default 0.5 it finds {_one(det, 'sensitivity')}% and clears {_one(det, 'specificity')}% "
        "(more tumours, more false alarms). "
        f"Saved follow-up cut-off: {tthr:.3f}. "
        f"It finds {_one(tdet, 'tuned_sensitivity')}% and clears {_one(tdet, 'tuned_specificity')}%.",
        st,
    ))

    story.append(P("Step 54. How a follow-up reference is chosen", st, "h3"))
    story.append(P(
        "Inside one phantom and one sitting, list the healthy scans. "
        "For every scan in that sitting, tumour or healthy, the other healthy scans are the allowed references. "
        "Let m be the smallest number of other healthy scans any one of those scans has. Use k = min(3, m). "
        "If m is 0, the sitting is dropped. The k scans are drawn with a random generator seeded by the scan id, "
        "the same way for both classes. The reference is their average. "
        "A tumour is never given a cleaner baseline than the healthy scans, so k cannot give the label away.",
        st,
    ))
    story.append(P(
        f"<b>Measured.</b> Follow-up score {follow}/100 "
        f"(older phantoms {_one(tdet, 'auc_gen1')}, newest {_one(tdet, 'auc_gen3')}, "
        f"tumours of 2 cm or smaller {_one(tdet, 'auc_small')}). "
        "The change in the scan grows with tumour diameter on the newest phantoms, "
        "which is the check that the subtracted signal is the tumour and not a processing accident. "
        "A 1 cm tumour changes the scan about as much as repeating a healthy scan.",
        st,
    ))

    story.append(P("Step 55. Size, diameter, type and criticality", st, "h3"))
    story.append(P(
        "size = small if d &lt;= 2 cm<br/>"
        "       medium if d &lt;= 4 cm<br/>"
        "       large otherwise<br/>"
        "Balanced accuracy = average of the three class recalls.<br/>"
        "Chance for three equal classes = 1/3.<br/>"
        "Diameter error = mean |predicted cm - true cm|<br/><br/>"
        "Type would be spherical versus spiculated. "
        "Only 6 of 134 fair tumour scans are spiculated, all in the older phantoms, "
        "so no shape classifier is reported.<br/><br/>"
        "Criticality is a research index, not clinical BI-RADS "
        "(the metadata BI-RADS number is breast density):<br/>"
        "C = (d / 2 cm)^3<br/>"
        "C = 1.5 * C if the shape string contains spiculated<br/>"
        "low if C &lt;= 1, moderate if C &lt;= 8, high if C &gt; 8<br/>"
        "A 2 cm sphere is low (C = 1). A 2 cm spiculated inclusion is moderate (C = 1.5). "
        "A 4 cm sphere is moderate (C = 8). A 6 cm sphere is high (C = 27).",
        st, "formula",
    ))
    crit_txt = ""
    if crit.get("mean") is not None:
        crit_txt = (
            f" The volume-index classifier on the first-scan fair tumours scores "
            f"{100 * crit['mean']:.0f} +/- {100 * crit.get('std', 0):.0f}/100 balanced accuracy (chance 33)."
        )
        if crit_plus.get("mean") is not None:
            crit_txt += f" Adding the physics summaries leaves it at {100 * crit_plus['mean']:.0f}/100."
    story.append(P(
        f"First-scan size class: {_spread(size, 'balanced_accuracy')}/100 balanced accuracy "
        f"(recording-details-only {100 * size['setup_only']['balanced_accuracy']['mean']:.0f}). "
        f"First-scan diameter error: {_cm(diam, 'mae_cm')} "
        f"(older {_cm(diam, 'mae_cm_gen1')}, newest {_cm(diam, 'mae_cm_gen3')}). "
        f"Follow-up size class: {_spread(tsize, 'balanced_accuracy')}/100, "
        f"and {_one(tsize, 'accuracy_gen3')}% correct on the newest phantoms. "
        f"Follow-up diameter error: {_cm(tdiam, 'mae_cm')}, newest phantoms {_cm(tdiam, 'mae_cm_gen3')}."
        + crit_txt,
        st,
    ))

    story.append(P("Step 56. Where the tumour is", st, "h3"))
    story.append(P(
        "A learned position model is trained during the nested test and then not saved. "
        "Inside one sitting the tumour sits in the same place, so a model that can tell sittings apart "
        "places the tumour without reading the image. The honest localiser never sees a label or a sitting: "
        "it is the brightest spot of the 1-4 GHz coherent image (imgl_peak_x_cm, imgl_peak_y_cm).",
        st,
    ))
    story.append(P(
        "miss_cm = hypot(x_peak - x_true, y_peak - y_true)<br/>"
        "radar alternative, not used as the official localiser:<br/>"
        "d = R_antenna - c * t_peak / 2<br/>"
        "(x, y) = d * (cos theta, sin theta) at the brightest residual antenna",
        st, "formula",
    ))
    spot = pos.get("image_spot") or {}
    trained = pos.get("nested", {}).get("mean_cm", {})
    radar = (ident.get("physics") or {})
    radar_txt = ""
    if radar.get("mean_cm") is not None and (ident.get("image_spot_same_scans") or {}).get("mean_cm") is not None:
        same = ident["image_spot_same_scans"]
        gen3 = (ident.get("physics_by_generation") or {}).get("gen3") or {}
        radar_txt = (
            f" The closed-form radar range, on the same {radar.get('n', '')} tumour scans, misses by "
            f"{radar['mean_cm']:.2f} cm ({100 * radar['within_2cm']:.0f}% within 2 cm). "
            f"The image spot on those scans misses by {same['mean_cm']:.2f} cm "
            f"({100 * same['within_2cm']:.0f}% within 2 cm)."
        )
        if gen3.get("mean_cm") is not None:
            radar_txt += (
                f" On gen3 the radar range misses by {gen3['mean_cm']:.2f} cm "
                f"({100 * gen3['within_2cm']:.0f}% within 2 cm)."
            )
    story.append(P(
        f"On the fair tumour scans the image spot misses by {spot.get('mean_cm', float('nan')):.2f} cm on average "
        f"({100 * spot.get('within_2cm', float('nan')):.0f}% within 2 cm). "
        f"The nested trained position model misses by {trained.get('mean', float('nan')):.2f} cm, which is worse, "
        "and it is the one that can borrow the sitting."
        + radar_txt
        + " Gen1 images still lose to guessing the centre: the dense inner tissue is off-centre and brighter than the tumour. "
        "Gen3 is the generation where the image spot is the result to quote. "
        "The gen3 antenna layout was fitted using tumour positions, and the held-out half is the number that should be quoted for that layout.",
        st,
    ))

    story.append(P("Step 57. What happens for one new scan", st, "h3"))
    story.append(P(
        "<b>Input:</b> the scan, its empty-chamber scan (or the follow-up healthy average), antenna radius, generation.<br/>"
        "<b>Steps inside predict.py:</b> S_cal, S_rot, signal features, coherent DAS, low-band DAS, image features, physics summaries.<br/>"
        "<b>Then:</b> the saved forest (first scan) or the saved logistic model (follow-up) turns that vector into p. "
        "Compare p with the saved Youden cut-off. Size class and diameter come from their own saved models. "
        "Criticality is computed from the predicted diameter with the spherical factor, unless a shape string is supplied.<br/>"
        "<b>Output:</b> tumour probability, yes/no, size class, diameter in cm, criticality, and the image-spot (x, y). "
        "The image spot is marked reliable only for gen3.",
        st, "io",
    ))

    story.append(P("Step 58. How the 71 and the 69 fit together", st, "h3"))
    story.append(P(
        f"71/100 is {_spread(det, 'auc')}: nested test, feature family chosen inside the training phantoms, "
        "and that choice was signal plus image, random forest, trained with the extra scans. "
        "69/100 is the later side check: the forest was locked to signal features and to the fair training scans only, "
        "before the formulas were added. Adding the formulas made 68. "
        "Same phantoms, smaller menu of inputs. The 69 did not replace the 71. "
        f"Oldest-phantom first scans are the weak part ({_one(det, 'auc_gen1')}/100). "
        f"Newest-phantom first scans are {_one(det, 'auc_gen3')}/100. "
        f"Follow-up on those same newest phantoms is {_one(tdet, 'auc_gen3')}/100.",
        st,
    ))

    story.append(P("Step 59. What was tried and did not raise the first scan", st, "h3"))
    story.extend([
        P("Pasting a tumour echo measured on a training breast onto other training breasts. The fair test breasts were never touched. The forest score fell.", st, "bullet"),
        P("Subtracting an average fat shell built from other breasts. That removed the tumour as well as the shell, and the first visit stayed with the empty chamber only.", st, "bullet"),
        P("Removing the strongest shared patterns (SVD) and cutting the early skin echo. The shared-pattern removal took the tumour with it. The time cut changed nothing.", st, "bullet"),
        P("The physics summaries in Step 49. They describe the scan, and they do not separate tumour from healthy on a first visit.", st, "bullet"),
        P(
            "A convolutional network on the breast-window sinogram, which is the input behind the published 78 ROC figure, "
            "and a PCA-then-logistic classifier on that same sinogram, which is the simulation papers' step. "
            "On this fair phantom-held-out test the network scored 50/100 and the PCA logistic 45/100. "
            "The network could fit the breasts it trained on (about 71/100 on those breasts) and the score fell to about 58/100 on a held-out training phantom, then to 50/100 on the fair test. "
            "Neither replaced the forest.",
            st, "bullet",
        ),
    ])

    story.append(P("Step 60. Scores next to the step that produces them", st, "h3"))
    story.append(P(
        f"Step 45-48 features, nested, first scan: {first}/100 (Step 51-53).<br/>"
        f"Step 45 follow-up subtraction, nested: {follow}/100 (Step 54).<br/>"
        f"Step 38 rule checker, same first-scan scans, no training: {100 * det['rule_based']['auc']:.0f}/100.<br/>"
        f"Step 53 cut-off, first scan: find {_one(det, 'tuned_sensitivity')}%, clear {_one(det, 'tuned_specificity')}%.<br/>"
        f"Step 55 size, first scan / follow-up: {_spread(size, 'balanced_accuracy')} / {_spread(tsize, 'balanced_accuracy')} balanced accuracy out of 100.<br/>"
        f"Step 55 diameter, first scan / follow-up: {_cm(diam, 'mae_cm')} / {_cm(tdiam, 'mae_cm')}.<br/>"
        f"Step 56 image spot: {spot.get('mean_cm', float('nan')):.1f} cm average miss.<br/>"
        "Sinogram network / PCA logistic, same fair first-scan test: 50/100 / 45/100. Not adopted.<br/>"
        "Files: results/classification/final_summary.json holds every nested number. "
        "results/reports/classification/guide_progress.pdf is the short sheet for a project guide.",
        st, "formula",
    ))

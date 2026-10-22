# Beamformer constant sweep

Research and educational comparison only. These scores are not a clinical diagnosis.

## How the average was defined

Every constant was sampled from **0.00 to 1.00 in steps of 0.05** (21 values). Values outside that interval are not part of the standard definitions: a coherence power above 1 over-suppresses any pixel that is not perfectly coherent, and a negative exponent blows up noise.

The cohort is a stratified UM-BMID sample, not all 2,264 scans: each generation contributes one healthy scan and small, medium, and large tumors in different quadrants when those labels exist. This run used **9 tumor scans and 3 healthy scans**.

For each setting the pipeline's own ROI detector picks the top spot **without the tumor label**. The label is used afterwards only to measure distance. A tumor score is

`0.55 exp(-distance_cm / 2.5) + 0.15 ring_focus + 0.15 compact_size + 0.15 SCR`.

A healthy score is 1 when that same blind rule rejects the scan, and lower when it calls a false tumor. The reported average is **0.70 × mean tumor score + 0.30 × mean healthy score**.

DAS has one knob, so its winner is the gamma with the best average. DMAS and DMAS-D4 share the pair exponent and the DMAS coherence gamma in the code, and D4 adds its own exponent. The shared triple maximizes `0.5 × DMAS score + 0.5 × DMAS-D4 score`. The default is that highest average. A second peak stays a second peak even when it is within 0.01, so a lower mode is not promoted just because it is nearer the textbook constants. The textbook value replaces the peak only when it is one grid step away and the score drop is under 0.001.

## Chosen defaults

| Knob | Textbook | Average-best |
| --- | --- | --- |
| DAS coherence γ | 0.00 | **0.55** |
| DMAS pair exponent | 0.50 | **0.30** |
| DMAS coherence γ | 0.00 | **0.00** |
| DMAS-D4 exponent | 0.25 | **0.25** |

DAS average at the winner: **0.644**. Textbook γ = 0 scored 0.566.

Shared DMAS + D4 average at the winner: **0.618**. Textbook (0.50, 0, 0.25) scored 0.591.

Optimizing DMAS alone, ignoring D4, would pick pair **0.00** and γ **0.05** (score 0.589). Optimizing the D4 image alone would pick pair **0.30**, γ **0.00**, exponent **0.25** (score 0.655). The defaults below are the shared compromise, because one configuration feeds both algorithms.

## Cohort

| Generation | Scan | Label | This scan's best DAS / pair / DMAS γ / D4 | Cohort DAS cm | Textbook DAS cm | Cohort D4 cm | Textbook D4 cm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gen1 | 4 | healthy | 0.00 / 0.00 / 0.00 / 0.00 | healthy | healthy | healthy | healthy |
| gen1 | 3 | small/low/sphere/lower-right | 0.15 / 0.00 / 0.75 / 0.05 | 2.35 | 1.67 | 2.19 | 2.19 |
| gen1 | 7 | medium/moderate/sphere/upper-left | 0.20 / 0.85 / 0.90 / 1.00 | 6.73 | 9.82 | 6.67 | 6.67 |
| gen1 | 12 | large/high/sphere/upper-right | 0.25 / 1.00 / 0.00 / 0.15 | 7.61 | 9.24 | 7.26 | 7.61 |
| gen2 | 0 | healthy | 0.00 / 0.00 / 0.00 / 0.00 | healthy | healthy | healthy | healthy |
| gen2 | 106 | small/high/sphere/lower-left | 0.40 / 0.20 / 0.00 / 0.15 | 2.98 | 6.27 | 2.48 | 4.59 |
| gen2 | 104 | medium/high/sphere/central | 0.50 / 0.85 / 0.00 / 0.15 | 0.86 | 4.89 | 0.85 | 1.92 |
| gen2 | 105 | medium/high/sphere/upper-left | 0.55 / 0.75 / 0.00 / 0.25 | 0.31 | 6.55 | 0.29 | 0.31 |
| gen3 | 1 | healthy | 0.00 / 0.00 / 0.00 / 0.00 | healthy | healthy | healthy | healthy |
| gen3 | 6 | small/low/sphere/upper-right | 1.00 / 0.00 / 0.00 / 0.15 | 3.05 | 3.28 | 3.05 | 3.05 |
| gen3 | 16 | medium/moderate/sphere/upper-left | 1.00 / 1.00 / 1.00 / 1.00 | 3.18 | 3.02 | 3.32 | 3.32 |
| gen3 | 48 | medium/moderate/sphere/lower-right | 1.00 / 1.00 / 1.00 / 1.00 | 3.18 | 3.29 | 3.05 | 3.05 |

The fourth column is the best setting **for that scan alone**. It is there to show the spread. Healthy rows often tie across the whole grid, and the table then shows the first tied cell. The project default is the cohort average, not any one of those per-scan winners. Localization columns are the distance from the **blind top-ranked spot** to the label when the cohort-best or textbook constants are used.

## DAS coherence γ

| Value | Average | Tumor score | Healthy score | Tumor loc. cm | Tumor area | Tumor ring | Healthy FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.566 | 0.381 | 1.000 | 5.34 | 0.024 | 1.06 | 0.00 |
| 0.05 | 0.571 | 0.387 | 1.000 | 5.30 | 0.015 | 1.06 | 0.00 |
| 0.10 | 0.573 | 0.390 | 1.000 | 5.31 | 0.010 | 1.07 | 0.00 |
| 0.15 | 0.575 | 0.392 | 1.000 | 5.40 | 0.007 | 1.06 | 0.00 |
| 0.20 | 0.579 | 0.399 | 1.000 | 4.97 | 0.005 | 1.11 | 0.00 |
| 0.25 | 0.606 | 0.437 | 1.000 | 4.20 | 0.008 | 1.11 | 0.00 |
| 0.30 | 0.585 | 0.407 | 1.000 | 4.54 | 0.007 | 1.14 | 0.00 |
| 0.35 | 0.585 | 0.406 | 1.000 | 4.54 | 0.008 | 1.15 | 0.00 |
| 0.40 | 0.591 | 0.416 | 1.000 | 4.30 | 0.008 | 1.15 | 0.00 |
| 0.45 | 0.591 | 0.416 | 1.000 | 4.27 | 0.008 | 1.16 | 0.00 |
| 0.50 | 0.611 | 0.445 | 1.000 | 3.94 | 0.008 | 1.17 | 0.00 |
|  **0.55** |  **0.644** | 0.492 | 1.000 | 3.36 | 0.008 | 1.18 | 0.00 |
| 0.60 | 0.643 | 0.490 | 1.000 | 3.36 | 0.008 | 1.19 | 0.00 |
| 0.65 | 0.643 | 0.490 | 1.000 | 3.35 | 0.008 | 1.19 | 0.00 |
| 0.70 | 0.643 | 0.489 | 1.000 | 3.35 | 0.008 | 1.20 | 0.00 |
| 0.75 | 0.642 | 0.488 | 1.000 | 3.35 | 0.008 | 1.21 | 0.00 |
| 0.80 | 0.574 | 0.487 | 0.777 | 3.36 | 0.008 | 1.22 | 0.33 |
| 0.85 | 0.563 | 0.474 | 0.771 | 3.66 | 0.008 | 1.23 | 0.33 |
| 0.90 | 0.561 | 0.473 | 0.766 | 3.66 | 0.008 | 1.24 | 0.33 |
| 0.95 | 0.554 | 0.465 | 0.761 | 3.71 | 0.014 | 1.25 | 0.33 |
| 1.00 | 0.552 | 0.464 | 0.756 | 3.71 | 0.014 | 1.26 | 0.33 |

### Why the other DAS values lost

- 0.00–0.50 loses on localization. Mean distance of the top-ranked spot is 4.74 cm, against 3.36 cm for the winner. The energy is not locked on the labeled tumor.
- 0.60–0.75 sits on the same plateau as the winner (average score 0.643 vs 0.644). The gap is under 0.01, so these are not a separate result. The default is the highest point on that plateau.
- 0.80–1.00 raises healthy false positives to 0.33 against 0.00 for the winner.

Named alternatives on this axis:

- 0.00: overall 0.566 vs winner 0.644; mean localization 5.34 cm vs 3.36 cm; ROI area fraction 0.024 vs 0.008; ring excess 1.06 vs 1.18; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- 0.25: overall 0.606 vs winner 0.644; mean localization 4.20 cm vs 3.36 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.11 vs 1.18; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- 0.35: overall 0.585 vs winner 0.644; mean localization 4.54 cm vs 3.36 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.15 vs 1.18; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- 0.50: overall 0.611 vs winner 0.644; mean localization 3.94 cm vs 3.36 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.17 vs 1.18; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- **0.55 is the average-best DAS γ.**
- 0.75: overall 0.642 vs winner 0.644; mean localization 3.35 cm vs 3.36 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.21 vs 1.18; healthy false-positive rate 0.00 vs 0.00. Within 0.01 of the best average, so this is the same plateau, not a separate result. The default stays on the highest point.
- 1.00: overall 0.552 vs winner 0.644; mean localization 3.71 cm vs 3.36 cm; ROI area fraction 0.014 vs 0.008; ring excess 1.26 vs 1.18; healthy false-positive rate 0.33 vs 0.00. Healthy scans are called tumor candidates more often.

## DMAS pair exponent

This slice holds DMAS γ and the D4 exponent at the winning companions. The average column is the shared DMAS + D4 score. Localization, area, and ring are measured on the D4 image, because the D4 image is where the compression effect shows most.

| Value | Average | Tumor score | Healthy score | Tumor loc. cm | Tumor area | Tumor ring | Healthy FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.590 | 0.414 | 1.000 | 4.54 | 0.007 | 1.01 | 0.00 |
| 0.05 | 0.554 | 0.402 | 0.906 | 5.38 | 0.008 | 1.04 | 0.00 |
| 0.10 | 0.587 | 0.451 | 0.903 | 4.15 | 0.008 | 1.04 | 0.00 |
| 0.15 | 0.586 | 0.452 | 0.900 | 4.16 | 0.008 | 1.04 | 0.00 |
| 0.20 | 0.588 | 0.455 | 0.897 | 4.08 | 0.008 | 1.04 | 0.00 |
| 0.25 | 0.585 | 0.453 | 0.894 | 4.28 | 0.008 | 1.05 | 0.00 |
|  **0.30** |  **0.618** | 0.501 | 0.892 | 3.24 | 0.008 | 1.05 | 0.00 |
| 0.35 | 0.610 | 0.490 | 0.889 | 3.37 | 0.008 | 1.05 | 0.00 |
| 0.40 | 0.604 | 0.483 | 0.886 | 3.41 | 0.008 | 1.06 | 0.00 |
| 0.45 | 0.596 | 0.473 | 0.883 | 3.43 | 0.008 | 1.06 | 0.00 |
| 0.50 | 0.591 | 0.467 | 0.881 | 3.63 | 0.008 | 1.06 | 0.00 |
| 0.55 | 0.588 | 0.464 | 0.878 | 3.71 | 0.008 | 1.06 | 0.00 |
| 0.60 | 0.608 | 0.493 | 0.876 | 3.39 | 0.008 | 1.06 | 0.00 |
| 0.65 | 0.610 | 0.497 | 0.874 | 3.32 | 0.008 | 1.06 | 0.00 |
| 0.70 | 0.608 | 0.496 | 0.871 | 3.34 | 0.008 | 1.07 | 0.00 |
| 0.75 | 0.602 | 0.487 | 0.869 | 3.34 | 0.008 | 1.07 | 0.00 |
| 0.80 | 0.587 | 0.467 | 0.867 | 3.34 | 0.008 | 1.07 | 0.00 |
| 0.85 | 0.582 | 0.460 | 0.866 | 3.32 | 0.008 | 1.07 | 0.00 |
| 0.90 | 0.571 | 0.445 | 0.864 | 3.32 | 0.008 | 1.07 | 0.00 |
| 0.95 | 0.581 | 0.461 | 0.863 | 3.33 | 0.008 | 1.08 | 0.00 |
| 1.00 | 0.565 | 0.438 | 0.861 | 3.90 | 0.008 | 1.08 | 0.00 |

### Why the other pair exponents lost

- 0.00–0.25 loses on localization. Mean distance of the top-ranked spot is 4.43 cm, against 3.24 cm for the winner. The energy is not locked on the labeled tumor.
- 0.35 sits on the same plateau as the winner (average score 0.610 vs 0.618). The gap is under 0.01, so these are not a separate result. The default is the highest point on that plateau.
- 0.40–0.60 is simply lower on the combined average (0.597 vs 0.618). Localization 3.51 cm, area fraction 0.008. The loss is spread across the score, not one dramatic failure.
- 0.65–0.70 sits on the same plateau as the winner (average score 0.609 vs 0.618). The gap is under 0.01, so these are not a separate result. The default is the highest point on that plateau.
- 0.75–0.95 is simply lower on the combined average (0.585 vs 0.618). Localization 3.33 cm, area fraction 0.008. The loss is spread across the score, not one dramatic failure.
- 1.00 loses on localization. Mean distance of the top-ranked spot is 3.90 cm, against 3.24 cm for the winner. The energy is not locked on the labeled tumor.

Named alternatives:

- 0.25: overall 0.585 vs winner 0.618; mean localization 4.28 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.05 vs 1.05; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- 0.35: overall 0.610 vs winner 0.618; mean localization 3.37 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.05 vs 1.05; healthy false-positive rate 0.00 vs 0.00. Within 0.01 of the best average, so this is the same plateau, not a separate result. The default stays on the highest point.
- 0.50: overall 0.591 vs winner 0.618; mean localization 3.63 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.06 vs 1.05; healthy false-positive rate 0.00 vs 0.00. No single failure dominates; the average of localization, focus, and healthy rejection is lower.
- 0.55: overall 0.588 vs winner 0.618; mean localization 3.71 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.06 vs 1.05; healthy false-positive rate 0.00 vs 0.00. No single failure dominates; the average of localization, focus, and healthy rejection is lower.
- 0.75: overall 0.602 vs winner 0.618; mean localization 3.34 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.07 vs 1.05; healthy false-positive rate 0.00 vs 0.00. No single failure dominates; the average of localization, focus, and healthy rejection is lower.
- 1.00: overall 0.565 vs winner 0.618; mean localization 3.90 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.08 vs 1.05; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.

The same pair axis scored on the DMAS image only (D4 not applied), still at the winning DMAS γ:

| Value | Average | Tumor score | Healthy score | Tumor loc. cm | Tumor area | Tumor ring | Healthy FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.588 | 0.411 | 1.000 | 4.63 | 0.008 | 1.05 | 0.00 |
| 0.05 | 0.536 | 0.418 | 0.812 | 4.24 | 0.008 | 1.18 | 0.33 |
| 0.10 | 0.574 | 0.475 | 0.806 | 3.34 | 0.008 | 1.18 | 0.33 |
| 0.15 | 0.574 | 0.477 | 0.801 | 3.33 | 0.008 | 1.18 | 0.33 |
| 0.20 | 0.573 | 0.479 | 0.795 | 3.33 | 0.008 | 1.19 | 0.33 |
| 0.25 | 0.573 | 0.480 | 0.789 | 3.31 | 0.008 | 1.20 | 0.33 |
|  **0.30** |  **0.581** | 0.495 | 0.783 | 3.24 | 0.008 | 1.22 | 0.33 |
| 0.35 | 0.572 | 0.484 | 0.778 | 3.37 | 0.008 | 1.24 | 0.33 |
| 0.40 | 0.563 | 0.474 | 0.772 | 3.59 | 0.008 | 1.26 | 0.33 |
| 0.45 | 0.549 | 0.456 | 0.767 | 3.85 | 0.008 | 1.26 | 0.33 |
| 0.50 | 0.546 | 0.453 | 0.762 | 3.90 | 0.008 | 1.27 | 0.33 |
| 0.55 | 0.543 | 0.452 | 0.756 | 3.98 | 0.008 | 1.28 | 0.33 |
| 0.60 | 0.566 | 0.486 | 0.752 | 3.57 | 0.008 | 1.29 | 0.33 |
| 0.65 | 0.567 | 0.489 | 0.747 | 3.21 | 0.039 | 1.32 | 0.33 |
| 0.70 | 0.564 | 0.488 | 0.743 | 3.24 | 0.039 | 1.33 | 0.33 |
| 0.75 | 0.551 | 0.471 | 0.739 | 3.34 | 0.071 | 1.37 | 0.33 |
| 0.80 | 0.522 | 0.431 | 0.735 | 3.70 | 0.069 | 1.35 | 0.33 |
| 0.85 | 0.510 | 0.415 | 0.731 | 4.58 | 0.036 | 1.30 | 0.33 |
| 0.90 | 0.489 | 0.386 | 0.728 | 4.93 | 0.036 | 1.31 | 0.33 |
| 0.95 | 0.510 | 0.418 | 0.725 | 4.46 | 0.008 | 1.28 | 0.33 |
| 1.00 | 0.509 | 0.418 | 0.722 | 3.93 | 0.036 | 1.33 | 0.33 |

## DMAS coherence γ

Pair exponent and D4 exponent are held at the winning companions.

| Value | Average | Tumor score | Healthy score | Tumor loc. cm | Tumor area | Tumor ring | Healthy FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  **0.00** |  **0.618** | 0.501 | 0.892 | 3.24 | 0.008 | 1.05 | 0.00 |
| 0.05 | 0.615 | 0.498 | 0.889 | 3.27 | 0.008 | 1.05 | 0.00 |
| 0.10 | 0.581 | 0.450 | 0.887 | 4.30 | 0.008 | 1.05 | 0.00 |
| 0.15 | 0.580 | 0.450 | 0.885 | 4.30 | 0.008 | 1.05 | 0.00 |
| 0.20 | 0.581 | 0.451 | 0.883 | 4.10 | 0.008 | 1.05 | 0.00 |
| 0.25 | 0.580 | 0.451 | 0.881 | 4.10 | 0.008 | 1.06 | 0.00 |
| 0.30 | 0.579 | 0.450 | 0.879 | 4.10 | 0.008 | 1.06 | 0.00 |
| 0.35 | 0.577 | 0.448 | 0.877 | 4.17 | 0.008 | 1.06 | 0.00 |
| 0.40 | 0.576 | 0.448 | 0.875 | 4.15 | 0.008 | 1.06 | 0.00 |
| 0.45 | 0.575 | 0.447 | 0.873 | 4.16 | 0.008 | 1.06 | 0.00 |
| 0.50 | 0.570 | 0.441 | 0.872 | 4.16 | 0.008 | 1.07 | 0.00 |
| 0.55 | 0.569 | 0.441 | 0.870 | 4.16 | 0.008 | 1.07 | 0.00 |
| 0.60 | 0.569 | 0.440 | 0.869 | 4.16 | 0.008 | 1.07 | 0.00 |
| 0.65 | 0.568 | 0.440 | 0.868 | 4.16 | 0.008 | 1.07 | 0.00 |
| 0.70 | 0.572 | 0.446 | 0.866 | 4.09 | 0.008 | 1.07 | 0.00 |
| 0.75 | 0.558 | 0.427 | 0.865 | 4.12 | 0.008 | 1.07 | 0.00 |
| 0.80 | 0.558 | 0.427 | 0.864 | 4.12 | 0.008 | 1.08 | 0.00 |
| 0.85 | 0.558 | 0.427 | 0.863 | 4.12 | 0.008 | 1.08 | 0.00 |
| 0.90 | 0.547 | 0.412 | 0.861 | 4.12 | 0.008 | 1.08 | 0.00 |
| 0.95 | 0.546 | 0.411 | 0.860 | 4.17 | 0.008 | 1.08 | 0.00 |
| 1.00 | 0.546 | 0.411 | 0.859 | 4.17 | 0.008 | 1.08 | 0.00 |

### Why the other DMAS γ values lost

- 0.05 sits on the same plateau as the winner (average score 0.615 vs 0.618). The gap is under 0.01, so these are not a separate result. The default is the highest point on that plateau.
- 0.10–1.00 loses on localization. Mean distance of the top-ranked spot is 4.16 cm, against 3.24 cm for the winner. The energy is not locked on the labeled tumor.

Named alternatives:

- **0.00 is the average-best DMAS γ.**
- 0.25: overall 0.580 vs winner 0.618; mean localization 4.10 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.06 vs 1.05; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- 0.35: overall 0.577 vs winner 0.618; mean localization 4.17 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.06 vs 1.05; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- 0.50: overall 0.570 vs winner 0.618; mean localization 4.16 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.07 vs 1.05; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- 0.60: overall 0.569 vs winner 0.618; mean localization 4.16 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.07 vs 1.05; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- 0.75: overall 0.558 vs winner 0.618; mean localization 4.12 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.07 vs 1.05; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.
- 1.00: overall 0.546 vs winner 0.618; mean localization 4.17 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.08 vs 1.05; healthy false-positive rate 0.00 vs 0.00. The top-ranked spot sits farther from the labeled tumor.

DMAS-image-only γ curve at the winning pair exponent:

| Value | Average | Tumor score | Healthy score | Tumor loc. cm | Tumor area | Tumor ring | Healthy FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  **0.00** |  **0.581** | 0.495 | 0.783 | 3.24 | 0.008 | 1.22 | 0.33 |
| 0.05 | 0.579 | 0.494 | 0.778 | 3.24 | 0.008 | 1.23 | 0.33 |
| 0.10 | 0.567 | 0.478 | 0.774 | 3.33 | 0.008 | 1.22 | 0.33 |
| 0.15 | 0.565 | 0.477 | 0.769 | 3.33 | 0.008 | 1.23 | 0.33 |
| 0.20 | 0.562 | 0.475 | 0.765 | 3.36 | 0.008 | 1.25 | 0.33 |
| 0.25 | 0.560 | 0.474 | 0.761 | 3.36 | 0.008 | 1.26 | 0.33 |
| 0.30 | 0.559 | 0.474 | 0.757 | 3.36 | 0.008 | 1.27 | 0.33 |
| 0.35 | 0.557 | 0.473 | 0.754 | 3.36 | 0.008 | 1.28 | 0.33 |
| 0.40 | 0.556 | 0.473 | 0.750 | 3.36 | 0.008 | 1.29 | 0.33 |
| 0.45 | 0.555 | 0.473 | 0.747 | 3.37 | 0.008 | 1.30 | 0.33 |
| 0.50 | 0.546 | 0.461 | 0.744 | 3.58 | 0.008 | 1.31 | 0.33 |
| 0.55 | 0.545 | 0.461 | 0.741 | 3.58 | 0.008 | 1.32 | 0.33 |
| 0.60 | 0.544 | 0.461 | 0.738 | 3.58 | 0.008 | 1.33 | 0.33 |
| 0.65 | 0.543 | 0.461 | 0.735 | 3.58 | 0.008 | 1.34 | 0.33 |
| 0.70 | 0.547 | 0.468 | 0.732 | 3.51 | 0.008 | 1.35 | 0.33 |
| 0.75 | 0.522 | 0.433 | 0.730 | 3.96 | 0.008 | 1.37 | 0.33 |
| 0.80 | 0.522 | 0.433 | 0.727 | 3.96 | 0.008 | 1.39 | 0.33 |
| 0.85 | 0.521 | 0.434 | 0.725 | 3.96 | 0.008 | 1.40 | 0.33 |
| 0.90 | 0.500 | 0.405 | 0.723 | 4.33 | 0.008 | 1.42 | 0.33 |
| 0.95 | 0.499 | 0.405 | 0.721 | 4.39 | 0.008 | 1.43 | 0.33 |
| 1.00 | 0.499 | 0.405 | 0.719 | 4.39 | 0.008 | 1.45 | 0.33 |

## DMAS-D4 exponent

Pair exponent and DMAS γ are held at the winning companions. This table is the D4 image alone, so a bad exponent cannot hide behind a good DMAS image.

| Value | Average | Tumor score | Healthy score | Tumor loc. cm | Tumor area | Tumor ring | Healthy FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.522 | 0.317 | 1.000 | 9.39 | 0.004 | 1.00 | 0.00 |
| 0.05 | 0.630 | 0.471 | 1.000 | 3.70 | 0.008 | 1.01 | 0.00 |
| 0.10 | 0.655 | 0.507 | 1.000 | 3.24 | 0.008 | 1.02 | 0.00 |
| 0.15 | 0.655 | 0.508 | 1.000 | 3.24 | 0.008 | 1.03 | 0.00 |
| 0.20 | 0.656 | 0.508 | 1.000 | 3.24 | 0.008 | 1.04 | 0.00 |
|  **0.25** |  **0.655** | 0.508 | 1.000 | 3.24 | 0.008 | 1.05 | 0.00 |
| 0.30 | 0.655 | 0.507 | 1.000 | 3.24 | 0.008 | 1.06 | 0.00 |
| 0.35 | 0.654 | 0.506 | 1.000 | 3.24 | 0.008 | 1.07 | 0.00 |
| 0.40 | 0.653 | 0.505 | 1.000 | 3.24 | 0.008 | 1.08 | 0.00 |
| 0.45 | 0.652 | 0.503 | 1.000 | 3.24 | 0.008 | 1.09 | 0.00 |
| 0.50 | 0.651 | 0.502 | 1.000 | 3.24 | 0.008 | 1.10 | 0.00 |
| 0.55 | 0.650 | 0.501 | 1.000 | 3.24 | 0.008 | 1.11 | 0.00 |
| 0.60 | 0.650 | 0.500 | 1.000 | 3.24 | 0.008 | 1.12 | 0.00 |
| 0.65 | 0.649 | 0.498 | 1.000 | 3.24 | 0.008 | 1.13 | 0.00 |
| 0.70 | 0.648 | 0.498 | 1.000 | 3.24 | 0.008 | 1.14 | 0.00 |
| 0.75 | 0.648 | 0.497 | 1.000 | 3.24 | 0.008 | 1.16 | 0.00 |
| 0.80 | 0.647 | 0.496 | 1.000 | 3.24 | 0.008 | 1.17 | 0.00 |
| 0.85 | 0.588 | 0.496 | 0.804 | 3.24 | 0.008 | 1.18 | 0.33 |
| 0.90 | 0.586 | 0.495 | 0.796 | 3.24 | 0.008 | 1.19 | 0.33 |
| 0.95 | 0.583 | 0.495 | 0.790 | 3.24 | 0.008 | 1.20 | 0.33 |
| 1.00 | 0.581 | 0.495 | 0.783 | 3.24 | 0.008 | 1.22 | 0.33 |

### Why the other D4 exponents lost

- 0.00 loses on localization. Mean distance of the top-ranked spot is 9.39 cm, against 3.24 cm for the winner. The energy is not locked on the labeled tumor.
- 0.05 is simply lower on the combined average (0.630 vs 0.655). Localization 3.70 cm, area fraction 0.008. The loss is spread across the score, not one dramatic failure.
- 0.10–0.20 sits on the same plateau as the winner (average score 0.655 vs 0.655). The gap is under 0.01, so these are not a separate result. The default is the highest point on that plateau.
- 0.30–0.80 sits on the same plateau as the winner (average score 0.651 vs 0.655). The gap is under 0.01, so these are not a separate result. The default is the highest point on that plateau.
- 0.85–1.00 raises healthy false positives to 0.33 against 0.00 for the winner.

Named alternatives:

- **0.25 is the average-best D4 exponent.**
- 0.35: overall 0.654 vs winner 0.655; mean localization 3.24 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.07 vs 1.05; healthy false-positive rate 0.00 vs 0.00. Within 0.01 of the best average, so this is the same plateau, not a separate result. The default stays on the highest point.
- 0.50: overall 0.651 vs winner 0.655; mean localization 3.24 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.10 vs 1.05; healthy false-positive rate 0.00 vs 0.00. Within 0.01 of the best average, so this is the same plateau, not a separate result. The default stays on the highest point.
- 0.55: overall 0.650 vs winner 0.655; mean localization 3.24 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.11 vs 1.05; healthy false-positive rate 0.00 vs 0.00. Within 0.01 of the best average, so this is the same plateau, not a separate result. The default stays on the highest point.
- 0.75: overall 0.648 vs winner 0.655; mean localization 3.24 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.16 vs 1.05; healthy false-positive rate 0.00 vs 0.00. Within 0.01 of the best average, so this is the same plateau, not a separate result. The default stays on the highest point.
- 1.00: overall 0.581 vs winner 0.655; mean localization 3.24 cm vs 3.24 cm; ROI area fraction 0.008 vs 0.008; ring excess 1.22 vs 1.05; healthy false-positive rate 0.33 vs 0.00. Healthy scans are called tumor candidates more often.

## What this does not claim

- The grid step is 0.05. A neighbor 0.02 away was not scored, and the tie rule treats gaps under 0.01 as noise.
- The cohort is stratified across generations, sizes, and locations. It is not every UM-BMID scan. A different mix can move the average slightly.
- One scan can prefer a different value. The default is the cohort mean. Per-scan retuning is a separate search and is not what these defaults are.
- Geometry, wave speed, and the numerical floor ε = 1e-9 were held at the values the app already uses. This sweep does not retune those.

Plots: `results/beamformer_sweep/das_gamma.png`, `dmas_pair.png`, `dmas_gamma.png`, `dmas_d4.png`.


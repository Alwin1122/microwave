"""Shared status text so the dataset, automation and project reports stay aligned."""

HEADING = "Tumour classification status"

PARAGRAPHS = (
    "Official first-scan score: <b>71/100</b>. This is the nested test on UM-BMID "
    "(signal plus image, extra training scans from phantoms that are not in the test fold, "
    "cut-off chosen on training phantoms only). It has not been replaced.",
    "Official follow-up score: <b>97/100</b>. The scan is compared with earlier healthy scans "
    "of the same breast from the same sitting. Tumour and healthy cases are given the same number "
    "of those scans, at most 3, so the count cannot give the answer away.",
    "A later side check scored about <b>69/100</b>. That check used only the signal features and only "
    "the fair training scans. Adding scattering-spectrum and frequency-slope formulas did not raise it. "
    "It is recorded as a side check, not as a new official score.",
    "A convolutional network on the breast-window sinogram, and a PCA-then-logistic classifier on that "
    "same sinogram, were then scored on the fair phantom-held-out test. They scored <b>50/100</b> and "
    "<b>45/100</b>. The network could fit the breasts it trained on and did not carry that fit to a new phantom. "
    "Neither replaced the official first-scan model.",
    "Two write-ups live in results/reports/classification/: "
    "<b>how_classification_works.pdf</b> (process, cut-offs and formulas, in plain language) and "
    "<b>guide_progress.pdf</b> (numbers and progress for a project guide). "
    "Machine-readable scores stay in results/classification/final_summary.json.",
)

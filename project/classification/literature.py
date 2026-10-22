"""Published breast-microwave results used as context for this study.

Numbers below are the figures stated in the cited abstracts or tables. They
are not recomputed here. Protocols differ (simulation versus phantom, image
reading versus a classifier, whether the same breast can appear in training
and test), so the table is a context for the contribution, not a leaderboard.
"""

from __future__ import annotations

# Each row is something a paper reported, plus the restriction that stops a
# reader treating the number as the same experiment as ours.
PUBLISHED = (
    {
        "id": "reimer_eucap_2020",
        "study": "Reimer, Krenkevich and Pistorius, EuCAP 2020",
        "citation": "T. Reimer, J. Krenkevich and S. Pistorius, \"An open-access experimental dataset for breast microwave imaging,\" Proc. 14th Eur. Conf. Antennas Propag. (EuCAP), 2020. doi:10.23919/EuCAP48036.2020.9135659",
        "task": "Detection",
        "data": "UM-BMID subset, experimental phantoms",
        "method": "Logistic regression on the sinogram",
        "result": "Accuracy 85 ± 4%",
        "restriction": "Subset accuracy. The later leave-one-shell-out study found logistic regression did not generalise.",
    },
    {
        "id": "reimer_jerm_2022",
        "study": "Reimer and Pistorius, IEEE J-ERM 2022",
        "citation": "T. Reimer and S. Pistorius, \"The diagnostic performance of machine learning in breast microwave sensing on an experimental dataset,\" IEEE J. Electromagn. RF Microw. Med. Biol., vol. 6, no. 1, pp. 139–145, 2022. doi:10.1109/JERM.2021.3076100",
        "task": "Detection",
        "data": "1008 UM-BMID scans, 66 phantoms",
        "method": "CNN on the 0.71–5.71 ns sinogram; leave-one-adipose-shell-out",
        "result": "ROC AUC 78 ± 3%; 90 ± 3% if the test breast volume is inside the training range and the tumour is at antenna height",
        "restriction": "First-visit detection. Logistic regression and a dense network did not generalise under the same split. No follow-up subtraction and no size or criticality task.",
    },
    {
        "id": "reimer_sensors_2023",
        "study": "Reimer and Pistorius, Sensors 2023 (review)",
        "citation": "T. Reimer and S. Pistorius, \"Review and analysis of tumour detection and image quality analysis in experimental breast microwave sensing,\" Sensors, vol. 23, no. 11, 5123, 2023. doi:10.3390/s23115123",
        "task": "Detection from images",
        "data": "Published phantom and patient studies",
        "method": "Image reading, several groups",
        "result": "Sensitivity 63–100% across 11 studies; specificity 20–65% in the 4 studies that reported it",
        "restriction": "Specificity is rarely reported. The review treats most machine-learning estimates other than the 2022 leave-one-shell-out result as likely optimistic.",
    },
    {
        "id": "davis_tbme_2008",
        "study": "Davis et al., IEEE TBME 2008",
        "citation": "S. K. Davis, B. D. Van Veen, S. C. Hagness and F. Kelcz, \"Breast tumor characterization based on ultrawideband microwave backscatter,\" IEEE Trans. Biomed. Eng., vol. 55, no. 1, pp. 237–246, 2008. doi:10.1109/TBME.2007.900564",
        "task": "Size and shape",
        "data": "360 numerical Gaussian-random-sphere targets, 1–11 GHz, 10 dB SNR",
        "method": "Local discriminant bases or PCA, then a linear classifier",
        "result": "Size accuracy above 97%; shape accuracy above 70%",
        "restriction": "Simulation with a known tumour response and added noise. No skin, fat shell or measured instrument drift.",
    },
    {
        "id": "conceicao_pier_2010",
        "study": "Conceição, O'Halloran, Jones and Glavin, PIER 2010",
        "citation": "R. C. Conceição, M. O'Halloran, E. Jones and M. Glavin, \"Investigation of classifiers for early-stage breast cancer based on radar target signatures,\" Progress In Electromagnetics Research, vol. 105, pp. 295–311, 2010. doi:10.2528/PIER10051904",
        "task": "Size, shape, then benign versus malignant from shape",
        "data": "Numerical breast model, Gaussian-random-sphere tumours",
        "method": "PCA of the radar target signature, then LDA or QDA",
        "result": "Coarse size 93.05% (LDA); coarse shape 80.90% (LDA); best size-then-shape cascade 81.10% (LDA)",
        "restriction": "Simulation. Benign versus malignant is assigned from the sphere's surface, not from a measured pathology label.",
    },
)

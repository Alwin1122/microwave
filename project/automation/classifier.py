"""Classify pipeline evidence into pass / warning / fail / blocked."""

from __future__ import annotations

from automation.policy import ThresholdProfile
from automation.types import DatasetVerdict, MetricEvidence


def _gate(
    name: str,
    value: float | None,
    threshold: float | None,
    comparison: str,
) -> MetricEvidence:
    if value is None or threshold is None:
        return MetricEvidence(name=name, value=value, threshold=threshold, comparison=comparison, ok=None)
    if comparison == ">=":
        ok = float(value) >= float(threshold)
    elif comparison == "<=":
        ok = float(value) <= float(threshold)
    else:
        ok = bool(value)
    return MetricEvidence(name=name, value=float(value), threshold=float(threshold), comparison=comparison, ok=ok)


def classify_evidence(
    *,
    profile: ThresholdProfile,
    load_ok: bool,
    load_error: str | None,
    finite_ok: bool | None,
    energy: float | None,
    roi_area: int | None,
    localization_error_cm: float | None,
    scr: float | None,
    snr: float | None,
    confidence: float | None,
    is_tumor_candidate: bool | None,
    has_tumor_gt: bool,
    blocked_reason: str | None = None,
) -> tuple[DatasetVerdict, list[MetricEvidence], list[str], list[str], bool]:
    """Return status, metrics, reasons, remediation, recommended_upload."""
    metrics: list[MetricEvidence] = []
    reasons: list[str] = []
    remediation: list[str] = []

    if blocked_reason:
        return (
            DatasetVerdict.BLOCKED,
            metrics,
            [blocked_reason],
            ["Provide a paired measurement file or choose a supported dataset."],
            False,
        )

    if not load_ok:
        return (
            DatasetVerdict.FAIL,
            metrics,
            [load_error or "Dataset failed to load."],
            ["Repair/replace the file or pick another scan index for BMID cubes."],
            False,
        )

    hard_fail = False
    soft_warn = False

    if profile.require_finite:
        metrics.append(MetricEvidence(name="finite_output", value=1.0 if finite_ok else 0.0, threshold=1.0, comparison=">=", ok=bool(finite_ok)))
        if not finite_ok:
            hard_fail = True
            reasons.append("Reconstructed output contains NaN/Inf.")
            remediation.append("Check frequency grid, wave speed, and preprocessing settings.")

    energy_gate = _gate("energy", energy, profile.min_energy, ">=")
    metrics.append(energy_gate)
    if energy_gate.ok is False:
        hard_fail = True
        reasons.append(f"Output energy {energy_gate.value:.3g} is below minimum {profile.min_energy:.3g}.")
        remediation.append("Verify calibration path and avoid over-suppressing target energy.")

    area_gate = _gate("roi_area", float(roi_area) if roi_area is not None else None, float(profile.min_roi_area), ">=")
    metrics.append(area_gate)
    if area_gate.ok is False:
        hard_fail = True
        reasons.append(f"ROI area {roi_area} is below minimum {profile.min_roi_area}.")
        remediation.append("Lower ROI threshold or inspect reconstruction FOV/geometry.")

    loc_gate = _gate("localization_error_cm", localization_error_cm, profile.max_localization_error_cm, "<=")
    metrics.append(loc_gate)
    if loc_gate.ok is False:
        soft_warn = True
        reasons.append(
            f"Localization error {localization_error_cm:.2f} cm exceeds "
            f"{profile.max_localization_error_cm:.2f} cm."
        )
        remediation.append("Try antenna angle offset / axis flip before changing wave speed.")

    scr_gate = _gate("scr", scr, profile.min_scr, ">=")
    metrics.append(scr_gate)
    if scr_gate.ok is False:
        soft_warn = True
        reasons.append(f"SCR {scr:.3g} is below gate {profile.min_scr:.3g}.")
        remediation.append("Prefer DMAS-D4 or adjust clutter settings for cleaner focus.")

    snr_gate = _gate("snr", snr, profile.min_snr, ">=")
    metrics.append(snr_gate)
    if snr_gate.ok is False:
        soft_warn = True
        reasons.append(f"SNR {snr:.3g} is below gate {profile.min_snr:.3g}.")

    conf_gate = _gate("confidence", confidence, profile.min_confidence, ">=")
    metrics.append(conf_gate)
    if conf_gate.ok is False:
        soft_warn = True
        reasons.append(f"Overall confidence {confidence:.2f} is below {profile.min_confidence:.2f}.")
        remediation.append("Review ROI compactness and beamformer selection mode.")

    if (
        profile.require_tumor_candidate_if_gt
        and has_tumor_gt
        and is_tumor_candidate is False
    ):
        soft_warn = True
        reasons.append("Ground-truth tumor scan was not marked as a tumor candidate.")
        remediation.append("Retune ROI mode (off-center/tight) or suspicion threshold.")

    if hard_fail:
        status = DatasetVerdict.FAIL
        recommended = False
    elif soft_warn:
        status = DatasetVerdict.WARNING
        recommended = localization_error_cm is None or (
            profile.max_localization_error_cm is None
            or localization_error_cm <= profile.max_localization_error_cm * 1.25
        )
        if not reasons:
            reasons.append("Completed with soft quality warnings.")
    else:
        status = DatasetVerdict.PASS
        recommended = True
        reasons.append("All configured quality gates passed.")

    if not remediation and status != DatasetVerdict.PASS:
        remediation.append("Open the session report details and inspect ROI / beamformer plots.")

    return status, metrics, reasons, remediation, recommended

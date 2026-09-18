"""Operational automation analysis report (not the mentor/beginner guides)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from automation.types import ValidationBatchResult

_DISCLAIMER = (
    "**Important:** Model / reconstruction output is **not a clinical diagnosis**. "
    "Results are research/educational estimates under stated assumptions and must not "
    "be used for medical decision-making."
)


def pick_showcase_result(batch: ValidationBatchResult):
    """Choose the GUI case: labeled tumor first, then strongest candidate."""
    if not batch.results:
        return None

    def _rank(item):
        details = item.details or {}
        cand = details.get("tumor_candidate") or {}
        has_gt = 1 if details.get("has_tumor_gt") else 0
        is_yes = 1 if cand.get("is_tumor_candidate") else 0
        loc = details.get("localization_error_cm")
        loc_rank = -float(loc) if isinstance(loc, (int, float)) else -1e3
        conf = float(cand.get("confidence") or 0.0)
        return (has_gt, is_yes, loc_rank, conf)

    return max(batch.results, key=_rank)


def write_automation_reports(batch: ValidationBatchResult, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    showcase = pick_showcase_result(batch)
    copied = _copy_showcase_figures(showcase, out)
    payload = batch.to_dict()
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["disclaimer"] = _DISCLAIMER
    payload["showcase"] = None
    if showcase is not None:
        payload["showcase"] = {
            "measurement_path": showcase.target.measurement_path,
            "scan_index": showcase.target.scan_index,
            "scan_label": showcase.target.scan_label,
            "tumor_candidate": (showcase.details or {}).get("tumor_candidate"),
            "has_tumor_gt": (showcase.details or {}).get("has_tumor_gt"),
        }

    analysis_md = _build_analysis_report(batch, showcase, copied)
    paths = {
        "analysis": out / "latest_auto_analysis.md",
        "json": out / "latest_auto_validation.json",
    }
    paths["analysis"].write_text(analysis_md, encoding="utf-8")
    paths["json"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    for key, src in copied.items():
        dest = out / f"latest_auto_{key}.png"
        paths[f"figure_{key}"] = dest
    return {k: str(v) for k, v in paths.items()}


def _copy_showcase_figures(showcase, output_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    if showcase is None:
        return copied
    figures = (showcase.details or {}).get("figures") or {}
    for key in ("selected", "beamformers", "roi_refine"):
        src = figures.get(key)
        if not src or not Path(src).is_file():
            continue
        dest = output_dir / f"latest_auto_{key}.png"
        shutil.copy2(src, dest)
        copied[key] = str(dest)
    return copied


def _yn(value: bool | None) -> str:
    if value is None:
        return "Unknown"
    return "Yes" if value else "No"


def _counts_table(batch: ValidationBatchResult) -> str:
    c = batch.counts or {}
    n_yes = sum(
        1
        for item in batch.results
        if ((item.details or {}).get("tumor_candidate") or {}).get("is_tumor_candidate")
    )
    return (
        "| Status | Count |\n|---|---:|\n"
        f"| pass | {c.get('pass', 0)} |\n"
        f"| warning | {c.get('warning', 0)} |\n"
        f"| fail | {c.get('fail', 0)} |\n"
        f"| blocked | {c.get('blocked', 0)} |\n"
        f"| tumor-candidate Yes | {n_yes} |\n"
    )


def _detection_table(batch: ValidationBatchResult) -> str:
    lines = [
        "| File | Scan | GT tumor | Type | Predicted candidate | Spots | Loc. error (cm) | Beamformer | Why |",
        "|---|---:|---|---|---|---:|---:|---|---|",
    ]
    for item in batch.results:
        details = item.details or {}
        cand = details.get("tumor_candidate") or {}
        tax = details.get("tumor_taxonomy") or {}
        type_txt = tax.get("short_label")
        if not type_txt:
            parts = [
                part
                for part in (
                    tax.get("size_class"),
                    tax.get("severity_class"),
                    tax.get("shape"),
                    tax.get("location_quadrant"),
                )
                if part
            ]
            type_txt = "/".join(parts) if parts else ("healthy" if tax.get("presence") == "healthy" else "n/a")
        name = Path(item.target.measurement_path).name
        scan = "" if item.target.scan_index is None else str(item.target.scan_index)
        loc = details.get("localization_error_cm")
        loc_txt = f"{loc:.2f}" if isinstance(loc, (int, float)) else "n/a"
        n_rois = details.get("n_rois") or cand.get("n_rois") or (details.get("reconstruction") or {}).get("n_rois") or 1
        reason = str(cand.get("reason") or (item.reasons[0] if item.reasons else "")).replace("|", "/")
        lines.append(
            f"| `{name}` | {scan} | {_yn(details.get('has_tumor_gt'))} | {type_txt} | "
            f"{_yn(cand.get('is_tumor_candidate'))} | {n_rois} | {loc_txt} | "
            f"{details.get('selected_beamformer') or 'n/a'} | {reason} |"
        )
    if not batch.results:
        lines.append("| _(none)_ |  |  |  |  |  |  |  |  |")
    return "\n".join(lines)


def _tumor_index_table(batch: ValidationBatchResult) -> str:
    index = batch.tumor_index or []
    if not index:
        return "No UM-BMID metadata inventory was available for this run."
    blocks = []
    for entry in index:
        examples = entry.get("examples") or []
        example_txt = "; ".join(
            f"scan {ex.get('index')} d={ex.get('tum_diam_cm')}cm @ "
            f"({ex.get('tum_x_cm')},{ex.get('tum_y_cm')}) cm"
            for ex in examples[:5]
        ) or "(no labeled tumor examples listed)"
        indices = ", ".join(str(i) for i in (entry.get("tumor_indices") or [])[:20])
        if len(entry.get("tumor_indices") or []) > 20:
            indices += ", …"
        blocks.append(
            "\n".join(
                [
                    f"### `{entry.get('file_name')}`",
                    f"- Total scans: {entry.get('n_scans')}",
                    f"- Labeled tumor scans: **{entry.get('n_tumor')}**",
                    f"- Healthy scans: {entry.get('n_healthy')}",
                    f"- Tumor scan indices: {indices or '(none)'}",
                    f"- Examples: {example_txt}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _case_walkthrough(batch: ValidationBatchResult, limit: int = 10) -> str:
    if not batch.results:
        return "No datasets were reconstructed in this run."
    sections = []
    for item in batch.results[:limit]:
        details = item.details or {}
        cand = details.get("tumor_candidate") or {}
        recon = details.get("reconstruction") or {}
        char = details.get("characterization") or {}
        name = Path(item.target.measurement_path).name
        loc = details.get("localization_error_cm")
        loc_txt = f"{loc:.2f} cm" if isinstance(loc, (int, float)) else "n/a"
        sections.append(
            "\n".join(
                [
                    f"#### {name} (scan {item.target.scan_index if item.target.scan_index is not None else 'n/a'})",
                    f"- Scan label: {item.target.scan_label or 'n/a'}",
                    f"- Status: **{item.status.value}**",
                    f"- GT tumor: {_yn(details.get('has_tumor_gt'))} at "
                    f"({details.get('tumor_gt_x_cm')}, {details.get('tumor_gt_y_cm')}) cm",
                    f"- Type/severity base: {(details.get('tumor_taxonomy') or {}).get('short_label', 'n/a')}",
                    f"- Predicted tumor-candidate: **{_yn(cand.get('is_tumor_candidate'))}** "
                    f"(score={cand.get('confidence')})",
                    f"- Detected spots: {details.get('n_rois') or cand.get('n_rois') or 1}",
                    f"- Beamformer: {recon.get('beamformer')}, ROI area={recon.get('roi_area_px')} px, "
                    f"centroid=({recon.get('centroid_x_cm')}, {recon.get('centroid_y_cm')}) cm",
                    f"- Localization error: {loc_txt}; FWHM={char.get('fwhm_cm')}; "
                    f"local SCR={char.get('local_scr')}",
                    f"- Interpretation: {cand.get('reason') or (item.reasons[0] if item.reasons else '')}",
                ]
            )
        )
    return "\n\n".join(sections)


def _build_analysis_report(batch: ValidationBatchResult, showcase, copied: dict[str, str]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    showcase_block = "No showcase case was selected."
    if showcase is not None:
        cand = (showcase.details or {}).get("tumor_candidate") or {}
        showcase_block = (
            f"- File: `{Path(showcase.target.measurement_path).name}`\n"
            f"- Scan: {showcase.target.scan_index} — {showcase.target.scan_label or ''}\n"
            f"- Tumor-candidate: **{_yn(cand.get('is_tumor_candidate'))}** "
            f"(confidence {cand.get('confidence')})\n"
            f"- Figures: {', '.join(f'`latest_auto_{k}.png`' for k in copied) or 'none'}\n"
        )
        if copied.get("selected"):
            showcase_block += "\n![Selected reconstruction](latest_auto_selected.png)\n"
        if copied.get("beamformers"):
            showcase_block += "\n![Beamformer comparison](latest_auto_beamformers.png)\n"
        if copied.get("roi_refine"):
            showcase_block += "\n![ROI refine](latest_auto_roi_refine.png)\n"

    return f"""# Auto Analysis Report

- Generated: {now}
- Profile: **{batch.profile}**
- Roots: {', '.join(f'`{r}`' for r in batch.roots) or '(none)'}
- Summary: {batch.message}

{_DISCLAIMER}

This file is the **automation output**. Mentor guide and beginner explainer
documents are separate reference notes; they are not generated here.

## 1. Where tumors are in the dataset

{_tumor_index_table(batch)}

## 2. Detection results (this run)

{_counts_table(batch)}

{_detection_table(batch)}

## 3. Showcase case (auto-opened in GUI)

{showcase_block}

## 4. What the pipeline did instead of manual GUI analysis

{_case_walkthrough(batch)}

## 5. How to read this without opening every GUI tab

1. Use **section 1** to find labeled tumor scan indices in UM-BMID files.
2. Use **section 2** for Yes/No candidate decisions, type/severity labels, and localization error.
3. Use **section 3** images as the Reconstruction Image / Compare / ROI refine tabs.
4. A recommended tumor case is also loaded into the GUI after Auto Validation.
"""

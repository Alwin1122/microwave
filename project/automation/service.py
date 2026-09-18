"""Orchestration service for automated validation batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from automation.cache import ValidationCache
from automation.discovery import discover_files
from automation.pipeline import validate_one_target
from automation.policy import get_threshold_profile
from automation.reports import write_automation_reports
from automation.resolver import build_tumor_index, resolve_measurement_targets
from automation.types import (
    DatasetValidationResult,
    DatasetVerdict,
    MetricEvidence,
    ResolvedTarget,
    ValidationBatchResult,
)

ProgressCb = Callable[[int, str], None]


def umbmid_root() -> Path | None:
    """Real UM-BMID simple-clean cubes (not the tiny project/datasets samples)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    path = repo_root / "umbmid" / "simple-clean"
    return path if path.is_dir() else None


def default_validation_roots() -> list[str]:
    """Use only UM-BMID cubes when they are present."""
    umbmid = umbmid_root()
    if umbmid is not None:
        return [str(umbmid)]
    project_root = Path(__file__).resolve().parent.parent
    datasets = project_root / "datasets"
    if datasets.is_dir():
        return [str(datasets)]
    return [str(project_root)]


class AutoValidationService:
    """Reusable automation facade used by CLI, GUI, and MCP."""

    def __init__(self, output_dir: str | None = None, cache_dir: str | None = None) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else project_root / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache = ValidationCache(cache_dir=cache_dir)

    def run(
        self,
        roots: list[str],
        *,
        max_files: int = 1,
        threshold_profile: str = "balanced",
        include_reconstruction_checks: bool = True,
        bmid_strategy: str = "diverse",
        max_bmid_scans: int = 8,
        use_cache: bool = True,
        progress: ProgressCb | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> ValidationBatchResult:
        profile = get_threshold_profile(threshold_profile)
        figure_dir = self.output_dir / "auto_cases"
        if progress:
            progress(2, f"Discovering files under {len(roots)} root(s)…")
        files = discover_files(roots, max_files=max_files)
        if progress:
            progress(8, f"Found {len(files)} candidate file(s); resolving targets…")
        targets = resolve_measurement_targets(
            files, bmid_strategy=bmid_strategy, max_bmid_scans=max_bmid_scans
        )
        tumor_index = build_tumor_index(
            [item.path for item in files if item.kind.value == "measurement"]
            + [t.measurement_path for t in targets]
        )
        if not targets:
            batch = ValidationBatchResult(
                profile=profile.name,
                roots=list(roots),
                results=[],
                counts={"pass": 0, "warning": 0, "fail": 0, "blocked": 0},
                message="No measurement targets resolved from the provided roots.",
                tumor_index=tumor_index,
            )
            paths = write_automation_reports(batch, self.output_dir)
            batch.report_paths = paths
            return batch

        results: list[DatasetValidationResult] = []
        for idx, target in enumerate(targets):
            if should_stop and should_stop():
                batch = _batch_from_results(
                    profile.name, roots, results, tumor_index, in_progress=False
                )
                batch.message = (
                    f"Cancelled after {len(results)} of {len(targets)} target(s)."
                )
                if progress:
                    progress(100, batch.message)
                paths = write_automation_reports(batch, self.output_dir)
                batch.report_paths = paths
                return batch
            pct = 10 + int(80 * idx / max(len(targets), 1))
            label = Path(target.measurement_path).name
            if progress:
                progress(pct, f"[{idx + 1}/{len(targets)}] Validating {label}…")

            cached_payload = None
            if use_cache:
                cached_payload = self.cache.get(
                    target.measurement_path,
                    target.scan_index,
                    profile.name,
                    include_reconstruction_checks,
                )
            cached_ok = False
            if cached_payload and "payload" in cached_payload:
                result = _result_from_dict(cached_payload["payload"])
                figures = (result.details or {}).get("figures") or {}
                cached_ok = bool(figures) and all(Path(p).is_file() for p in figures.values())
                if cached_ok:
                    result.cached = True
                    results.append(result)
                    continue

            result = validate_one_target(
                target,
                profile=profile,
                include_reconstruction_checks=include_reconstruction_checks,
                progress=(lambda msg: progress(pct, msg)) if progress else None,
                figure_dir=str(figure_dir) if include_reconstruction_checks else None,
            )
            if use_cache and result.status != DatasetVerdict.BLOCKED:
                self.cache.set(
                    target.measurement_path,
                    target.scan_index,
                    profile.name,
                    include_reconstruction_checks,
                    result.to_dict(),
                )
            results.append(result)
            if (idx + 1) % 5 == 0 or idx + 1 == len(targets):
                interim = _batch_from_results(
                    profile.name, roots, results, tumor_index, in_progress=idx + 1 < len(targets)
                )
                write_automation_reports(interim, self.output_dir)

        batch = _batch_from_results(profile.name, roots, results, tumor_index)
        if progress:
            progress(92, "Writing auto analysis report…")
        paths = write_automation_reports(batch, self.output_dir)
        batch.report_paths = paths
        if progress:
            progress(100, "Auto validation complete.")
        return batch

    def get_latest_report_paths(self) -> dict[str, str]:
        mapping = {
            "analysis": self.output_dir / "latest_auto_analysis.md",
            "json": self.output_dir / "latest_auto_validation.json",
        }
        return {k: str(v) for k, v in mapping.items() if v.is_file()}

    def list_recommended_uploads(self, batch: ValidationBatchResult | None = None) -> list[dict[str, Any]]:
        if batch is None:
            latest = self.output_dir / "latest_auto_validation.json"
            if not latest.is_file():
                return []
            import json

            payload = json.loads(latest.read_text(encoding="utf-8"))
            results = payload.get("results", [])
            return [
                {
                    "path": r.get("target", {}).get("measurement_path"),
                    "status": r.get("status"),
                    "scan_index": r.get("target", {}).get("scan_index"),
                    "reasons": r.get("reasons", [])[:3],
                }
                for r in results
                if r.get("recommended_upload")
            ]
        return [
            {
                "path": r.target.measurement_path,
                "status": r.status.value,
                "scan_index": r.target.scan_index,
                "reasons": r.reasons[:3],
            }
            for r in batch.results
            if r.recommended_upload
        ]

    def explain_dataset_status(self, measurement_path: str, scan_index: int | None = None) -> dict[str, Any]:
        latest = self.output_dir / "latest_auto_validation.json"
        if not latest.is_file():
            return {"found": False, "message": "No latest auto-validation JSON found. Run validation first."}
        import json

        payload = json.loads(latest.read_text(encoding="utf-8"))
        needle = str(Path(measurement_path).resolve()) if Path(measurement_path).exists() else measurement_path
        for item in payload.get("results", []):
            path = item.get("target", {}).get("measurement_path", "")
            idx = item.get("target", {}).get("scan_index")
            same_path = str(Path(path).resolve()) == needle if Path(path).exists() else path.endswith(Path(measurement_path).name)
            if same_path and (scan_index is None or idx == scan_index):
                return {"found": True, "result": item, "message": "; ".join(item.get("reasons", [])[:3])}
        return {"found": False, "message": "Dataset not present in latest auto-validation results."}


def _batch_from_results(
    profile_name: str,
    roots: list[str],
    results: list[DatasetValidationResult],
    tumor_index: list[dict],
    *,
    in_progress: bool = False,
) -> ValidationBatchResult:
    n_yes = sum(
        1
        for r in results
        if ((r.details or {}).get("tumor_candidate") or {}).get("is_tumor_candidate")
    )
    counts = {
        "pass": sum(1 for r in results if r.status == DatasetVerdict.PASS),
        "warning": sum(1 for r in results if r.status == DatasetVerdict.WARNING),
        "fail": sum(1 for r in results if r.status == DatasetVerdict.FAIL),
        "blocked": sum(1 for r in results if r.status == DatasetVerdict.BLOCKED),
    }
    prefix = "In progress — " if in_progress else ""
    return ValidationBatchResult(
        profile=profile_name,
        roots=list(roots),
        results=results,
        counts=counts,
        message=(
            f"{prefix}Validated {len(results)} target(s): "
            f"{counts['pass']} pass, {counts['warning']} warning, "
            f"{counts['fail']} fail, {counts['blocked']} blocked; "
            f"{n_yes} tumor-candidate Yes."
        ),
        tumor_index=tumor_index,
    )


def run_auto_validation(
    roots: list[str],
    *,
    max_files: int = 1,
    threshold_profile: str = "balanced",
    include_reconstruction_checks: bool = True,
    progress: ProgressCb | None = None,
    output_dir: str | None = None,
    bmid_strategy: str = "diverse",
    max_bmid_scans: int = 8,
) -> ValidationBatchResult:
    service = AutoValidationService(output_dir=output_dir)
    return service.run(
        roots,
        max_files=max_files,
        threshold_profile=threshold_profile,
        include_reconstruction_checks=include_reconstruction_checks,
        bmid_strategy=bmid_strategy,
        max_bmid_scans=max_bmid_scans,
        progress=progress,
    )


def _result_from_dict(payload: dict[str, Any]) -> DatasetValidationResult:
    target_raw = payload.get("target") or {}
    target = ResolvedTarget(
        measurement_path=str(target_raw.get("measurement_path")),
        metadata_path=target_raw.get("metadata_path"),
        scan_index=target_raw.get("scan_index"),
        scan_label=target_raw.get("scan_label"),
        source_note=str(target_raw.get("source_note") or ""),
    )
    metrics = [
        MetricEvidence(
            name=str(m.get("name")),
            value=m.get("value"),
            threshold=m.get("threshold"),
            comparison=m.get("comparison"),
            ok=m.get("ok"),
        )
        for m in payload.get("metrics") or []
    ]
    return DatasetValidationResult(
        target=target,
        status=DatasetVerdict(str(payload.get("status", "fail"))),
        reasons=list(payload.get("reasons") or []),
        remediation=list(payload.get("remediation") or []),
        metrics=metrics,
        recommended_upload=bool(payload.get("recommended_upload")),
        cached=True,
        details=dict(payload.get("details") or {}),
    )

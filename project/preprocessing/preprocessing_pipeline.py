"""
preprocessing/preprocessing_pipeline.py

Purpose:
    Orchestrate the full Module 2 signal-preprocessing pipeline:
        Raw Dataset -> Noise Filtering -> Calibration -> Normalization ->
        Background Subtraction -> Artifact Suppression -> Processed Dataset
    and compute the signal-quality metrics / statistics needed for the
    GUI's processing summary and comparison plots.

Input:
    A MicrowaveDataset (from Module 1) plus a PreprocessingConfig
    describing which methods/parameters to use at each stage.

Output:
    A PreprocessingResult containing the processed MicrowaveDataset, the
    original dataset (for comparison), per-stage intermediate S-parameter
    snapshots, and a SignalQualityReport.

Description:
    This is the single function the GUI's "Run Preprocessing" button
    calls. Each stage is wrapped so failures are reported with the stage
    name attached, and progress can be reported via an optional callback
    (used to drive the GUI progress bar).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from data_loader.dataset_info import MicrowaveDataset
from preprocessing.artifact_suppression import apply_artifact_suppression, background_subtraction
from preprocessing.calibration import apply_calibration
from preprocessing.filtering import apply_noise_filter
from preprocessing.normalization import apply_normalization
from preprocessing.touchstone_s21 import (
    TouchstoneS21ProcessingResult,
    TouchstoneS21ValidationReport,
    process_touchstone_s21_dataset,
)
from utils.exceptions import MicrowaveFrameworkError
from utils.logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[int, str], None]]

_STAGE_WEIGHTS = {
    "Noise Filtering": 20,
    "Calibration": 20,
    "Normalization": 15,
    "Background Subtraction": 15,
    "Artifact Suppression": 25,
    "Quality Evaluation": 5,
}


@dataclass
class PreprocessingConfig:
    """
    Purpose:
        Holds all user-configurable parameters for the Module 2 pipeline,
        typically populated from GUI controls.
    """

    filter_method: str = "savgol"
    filter_kwargs: dict = field(default_factory=dict)

    calibration_method: str = "self"
    calibration_reference: Optional[np.ndarray] = None

    normalization_method: str = "max"

    do_background_subtraction: bool = True
    background_reference: Optional[np.ndarray] = None

    artifact_method: str = "hybrid"
    artifact_kwargs: dict = field(default_factory=dict)

    repeated_measurements: Optional[list[MicrowaveDataset | np.ndarray]] = None
    reference_dataset: Optional[MicrowaveDataset | np.ndarray] = None
    spike_detection_method: str = "hampel"
    spike_detection_kwargs: dict = field(default_factory=dict)
    generate_normalized_copy: bool = True


@dataclass
class SignalQualityReport:
    """
    Purpose:
        Numeric signal-quality metrics computed before/after preprocessing,
        displayed in the GUI's "Processing Summary" panel.
    """

    snr_before_db: float
    snr_after_db: float
    dynamic_range_before_db: float
    dynamic_range_after_db: float
    mean_magnitude_before: float
    mean_magnitude_after: float
    std_magnitude_before: float
    std_magnitude_after: float
    artifact_reduction_ratio: float

    def to_display_dict(self) -> dict:
        return {
            "SNR Before (dB)": f"{self.snr_before_db:.2f}",
            "SNR After (dB)": f"{self.snr_after_db:.2f}",
            "Dynamic Range Before (dB)": f"{self.dynamic_range_before_db:.2f}",
            "Dynamic Range After (dB)": f"{self.dynamic_range_after_db:.2f}",
            "Mean Magnitude Before": f"{self.mean_magnitude_before:.6g}",
            "Mean Magnitude After": f"{self.mean_magnitude_after:.6g}",
            "Std. Dev. Before": f"{self.std_magnitude_before:.6g}",
            "Std. Dev. After": f"{self.std_magnitude_after:.6g}",
            "Artifact Reduction Ratio": f"{self.artifact_reduction_ratio:.3f}",
        }


@dataclass
class PreprocessingResult:
    """
    Purpose:
        Full output of the Module 2 pipeline: the processed dataset, the
        original dataset for comparison, intermediate stage outputs (for
        plotting), and the quality report.
    """

    original_dataset: MicrowaveDataset
    processed_dataset: MicrowaveDataset
    stage_outputs: dict  # stage name -> np.ndarray (S-parameters snapshot)
    quality_report: SignalQualityReport
    config: PreprocessingConfig
    validation_report: Optional[TouchstoneS21ValidationReport] = None
    touchstone_s21_result: Optional[TouchstoneS21ProcessingResult] = None


def estimate_snr_db(s_params: np.ndarray) -> float:
    """
    Purpose:
        Estimate the signal-to-noise ratio of an S-parameter trace set by
        treating the smooth (low-frequency-variation) component as signal
        and the high-frequency residual as noise.
    Input:
        s_params (np.ndarray): complex S-parameters, shape (n_freq, ...).
    Output:
        float: estimated SNR in dB. Returns +inf-safe large value if noise
        power is ~0.
    """
    magnitude = np.abs(s_params).reshape(s_params.shape[0], -1)
    if magnitude.shape[0] < 5:
        signal_power = np.mean(magnitude ** 2)
        return 10 * np.log10(signal_power + 1e-30)

    # Smooth version approximates the "signal"; residual approximates "noise"
    kernel = np.ones(5) / 5
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 0, magnitude)
    noise = magnitude - smoothed

    signal_power = np.mean(smoothed ** 2)
    noise_power = np.mean(noise ** 2) + 1e-30
    return float(10 * np.log10(signal_power / noise_power))


def compute_signal_statistics(s_params: np.ndarray) -> dict:
    """
    Purpose:
        Compute descriptive statistics of an S-parameter array's
        magnitude, used for the GUI's "Signal Statistics" display.
    Input:
        s_params (np.ndarray): complex S-parameters.
    Output:
        dict with mean, std, min, max, dynamic_range_db, snr_db.
    """
    magnitude = np.abs(s_params)
    mag_min = float(np.min(magnitude)) if magnitude.size else 0.0
    mag_max = float(np.max(magnitude)) if magnitude.size else 0.0
    dynamic_range_db = 20 * np.log10((mag_max + 1e-30) / (mag_min + 1e-30))

    return {
        "mean": float(np.mean(magnitude)) if magnitude.size else 0.0,
        "std": float(np.std(magnitude)) if magnitude.size else 0.0,
        "min": mag_min,
        "max": mag_max,
        "dynamic_range_db": float(dynamic_range_db),
        "snr_db": estimate_snr_db(s_params) if s_params.size else 0.0,
    }


def _report_progress(callback: ProgressCallback, percent: int, message: str) -> None:
    if callback is not None:
        callback(percent, message)


def _is_touchstone_s2p_dataset(dataset: MicrowaveDataset) -> bool:
    return dataset.file_type.startswith("Touchstone") and dataset.file_path.lower().endswith(".s2p")


def run_preprocessing_pipeline(
    dataset: MicrowaveDataset,
    config: Optional[PreprocessingConfig] = None,
    progress_callback: ProgressCallback = None,
) -> PreprocessingResult:
    """
    Purpose:
        Execute the complete Module 2 pipeline on a loaded dataset:
        Noise Filtering -> Calibration -> Normalization ->
        Background Subtraction -> Artifact Suppression, recording
        intermediate outputs and a final signal-quality report.
    Input:
        dataset (MicrowaveDataset): dataset produced by Module 1.
        config (PreprocessingConfig | None): pipeline configuration; a
            sensible default configuration is used if None.
        progress_callback (Callable[[int, str], None] | None): optional
            callback invoked as (percent_complete, stage_description),
            used to drive a GUI progress bar.
    Output:
        PreprocessingResult with the processed dataset and diagnostics.
    Raises:
        MicrowaveFrameworkError subclasses if any stage fails; the
        exception message is prefixed with the stage name.
    """
    config = config or PreprocessingConfig()

    if _is_touchstone_s2p_dataset(dataset):
        _report_progress(progress_callback, 5, "Parsing Touchstone S21 header and trace")
        touchstone_result = process_touchstone_s21_dataset(dataset, config)

        stage_outputs: dict = {
            "Raw": touchstone_result.raw_s21.reshape(-1, 1),
            "Corrected": touchstone_result.corrected_s21.reshape(-1, 1),
            "Filtered": touchstone_result.filtered_s21.reshape(-1, 1),
            "Windowed": touchstone_result.windowed_s21.reshape(-1, 1),
        }
        if touchstone_result.averaged_s21 is not None:
            stage_outputs["Averaged"] = touchstone_result.averaged_s21.reshape(-1, 1)
        if touchstone_result.reference_subtracted_s21 is not None:
            stage_outputs["Reference Subtracted"] = touchstone_result.reference_subtracted_s21.reshape(-1, 1)
        if touchstone_result.normalized_s21 is not None:
            stage_outputs["Normalized"] = touchstone_result.normalized_s21.reshape(-1, 1)

        raw_stats = compute_signal_statistics(touchstone_result.raw_s21.reshape(-1, 1))
        filtered_stats = compute_signal_statistics(touchstone_result.filtered_s21.reshape(-1, 1))
        raw_energy = np.mean(np.abs(touchstone_result.raw_s21) ** 2)
        filtered_energy = np.mean(np.abs(touchstone_result.filtered_s21) ** 2)
        artifact_reduction_ratio = float(
            1.0 - (filtered_energy / raw_energy) if raw_energy > 0 else 0.0
        )

        quality_report = SignalQualityReport(
            snr_before_db=raw_stats["snr_db"],
            snr_after_db=filtered_stats["snr_db"],
            dynamic_range_before_db=raw_stats["dynamic_range_db"],
            dynamic_range_after_db=filtered_stats["dynamic_range_db"],
            mean_magnitude_before=raw_stats["mean"],
            mean_magnitude_after=filtered_stats["mean"],
            std_magnitude_before=raw_stats["std"],
            std_magnitude_after=filtered_stats["std"],
            artifact_reduction_ratio=artifact_reduction_ratio,
        )

        processed_dataset = copy.copy(dataset)
        processed_dataset.frequencies = touchstone_result.uniform_frequencies_hz.copy()
        processed_dataset.s_parameters = touchstone_result.filtered_s21.reshape(-1, 1)
        processed_dataset.n_ports = 1
        processed_dataset.available_variables = ["S21"]
        processed_dataset.metadata = dict(dataset.metadata)
        processed_dataset.metadata["preprocessing_stages"] = list(stage_outputs.keys())
        processed_dataset.metadata["module2_s21"] = {
            "header": touchstone_result.header.to_dict(),
            "raw_magnitude_db": touchstone_result.raw_magnitude_db,
            "wrapped_phase_deg": touchstone_result.wrapped_phase_deg,
            "unwrapped_phase_deg": touchstone_result.unwrapped_phase_deg,
            "corrected_s21": touchstone_result.corrected_s21,
            "windowed_s21": touchstone_result.windowed_s21,
            "normalized_s21": touchstone_result.normalized_s21,
            "delta_f_hz": touchstone_result.delta_f_hz,
        }

        _report_progress(progress_callback, 100, "Preprocessing complete")
        logger.info("Touchstone S21 preprocessing completed successfully.")
        return PreprocessingResult(
            original_dataset=dataset,
            processed_dataset=processed_dataset,
            stage_outputs=stage_outputs,
            quality_report=quality_report,
            config=config,
            validation_report=touchstone_result.report,
            touchstone_s21_result=touchstone_result,
        )

    stage_outputs: dict = {}
    percent_done = 0

    original_s_params = dataset.s_parameters.copy()
    stage_outputs["Raw"] = original_s_params.copy()
    current = original_s_params.copy()

    stages = [
        ("Noise Filtering", lambda x: apply_noise_filter(x, method=config.filter_method, **config.filter_kwargs)),
        ("Calibration", lambda x: apply_calibration(x, method=config.calibration_method, reference=config.calibration_reference)),
        ("Normalization", lambda x: apply_normalization(x, method=config.normalization_method)),
        ("Background Subtraction", lambda x: background_subtraction(x, background=config.background_reference) if config.do_background_subtraction else x),
        ("Artifact Suppression", lambda x: apply_artifact_suppression(x, method=config.artifact_method, background=None, **config.artifact_kwargs)),
    ]

    for stage_name, stage_fn in stages:
        try:
            logger.info(f"--- Pipeline stage: {stage_name} ---")
            _report_progress(progress_callback, percent_done, f"Running: {stage_name}")
            current = stage_fn(current)
            stage_outputs[stage_name] = current.copy()
        except MicrowaveFrameworkError:
            raise
        except Exception as exc:  # wrap unexpected errors with stage context
            raise MicrowaveFrameworkError(f"[{stage_name}] {exc}") from exc

        percent_done += _STAGE_WEIGHTS.get(stage_name, 0)
        _report_progress(progress_callback, min(percent_done, 95), f"Completed: {stage_name}")

    _report_progress(progress_callback, 97, "Evaluating signal quality")
    stats_before = compute_signal_statistics(original_s_params)
    stats_after = compute_signal_statistics(current)

    artifact_before = np.mean(np.abs(original_s_params) ** 2)
    artifact_after = np.mean(np.abs(current) ** 2)
    artifact_reduction_ratio = float(
        1.0 - (artifact_after / artifact_before) if artifact_before > 0 else 0.0
    )

    quality_report = SignalQualityReport(
        snr_before_db=stats_before["snr_db"],
        snr_after_db=stats_after["snr_db"],
        dynamic_range_before_db=stats_before["dynamic_range_db"],
        dynamic_range_after_db=stats_after["dynamic_range_db"],
        mean_magnitude_before=stats_before["mean"],
        mean_magnitude_after=stats_after["mean"],
        std_magnitude_before=stats_before["std"],
        std_magnitude_after=stats_after["std"],
        artifact_reduction_ratio=artifact_reduction_ratio,
    )

    processed_dataset = copy.copy(dataset)
    processed_dataset.s_parameters = current
    processed_dataset.metadata = dict(dataset.metadata)
    processed_dataset.metadata["preprocessing_stages"] = list(stage_outputs.keys())

    _report_progress(progress_callback, 100, "Preprocessing complete")
    logger.info("Preprocessing pipeline completed successfully.")

    return PreprocessingResult(
        original_dataset=dataset,
        processed_dataset=processed_dataset,
        stage_outputs=stage_outputs,
        quality_report=quality_report,
        config=config,
        validation_report=None,
        touchstone_s21_result=None,
    )

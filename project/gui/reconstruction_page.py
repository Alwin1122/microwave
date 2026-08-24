"""GUI panel for beamforming reconstruction and image display."""

from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data_loader.dataset_info import MicrowaveDataset
from quality.beamformer_selector import select_best_beamformer
from reconstruction.auto_calibrate import AutoCalibrateResult, auto_calibrate_reconstruction
from reconstruction.reconstruction_manager import (
    ROIRefinement,
    infer_reconstruction_assessment,
    infer_reconstruction_config,
    reconstruct_all,
    reconstruct_high_resolution_roi,
)
from roi.roi_detector import ROIResult, detect_roi, roi_centroid_meters
from utils.logger import StatusLog, get_logger
from utils.session_report import ReconstructionSnapshot

logger = get_logger(__name__)


class AutoCalibrateWorker(QThread):
    """Background worker for Module 3 auto-tweak search."""

    progress_updated = Signal(int, str)
    finished_ok = Signal(object)
    finished_error = Signal(str)

    def __init__(
        self,
        s_parameters,
        frequencies,
        config,
        tumor_xy_m,
        quick: bool,
        include_wave_speeds: bool,
        baseline_prefer_off_center: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.s_parameters = s_parameters
        self.frequencies = frequencies
        self.config = config
        self.tumor_xy_m = tumor_xy_m
        self.quick = quick
        self.include_wave_speeds = include_wave_speeds
        self.baseline_prefer_off_center = baseline_prefer_off_center

    def run(self) -> None:
        try:
            result = auto_calibrate_reconstruction(
                self.s_parameters,
                self.frequencies,
                self.config,
                tumor_xy_m=self.tumor_xy_m,
                quick=self.quick,
                include_wave_speeds=self.include_wave_speeds,
                baseline_prefer_off_center=self.baseline_prefer_off_center,
                progress_callback=lambda pct, msg: self.progress_updated.emit(pct, msg),
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001 - surface to GUI
            self.finished_error.emit(str(exc))


class ReconstructionPanel(QWidget):
    """GUI panel for running beamforming reconstruction and displaying results."""

    reconstruction_completed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset: MicrowaveDataset | None = None
        self.processed_dataset: MicrowaveDataset | None = None
        self.status_log = StatusLog()
        self.last_roi_result: ROIResult | None = None
        self.last_roi_refinement: ROIRefinement | None = None
        self.last_snapshot: ReconstructionSnapshot | None = None
        self.last_auto_result: AutoCalibrateResult | None = None
        self.auto_worker: AutoCalibrateWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("Module 3 — Reconstruction & Beamformer Selection")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        config_group = QGroupBox("Reconstruction Configuration")
        form = QFormLayout()

        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Processed Dataset", "Raw Dataset"])
        form.addRow("Source Signal:", self.data_source_combo)

        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["64x64", "128x128"])
        form.addRow("Grid Resolution:", self.grid_combo)

        self.radius_combo = QComboBox()
        self.radius_combo.addItems(["8 cm", "10 cm", "12 cm", "18 cm"])
        self.radius_combo.setCurrentText("8 cm")
        form.addRow("Antenna Radius:", self.radius_combo)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["3.0e8 m/s (air)", "2.1e8 m/s", "1.8e8 m/s"])
        self.speed_combo.setCurrentText("3.0e8 m/s (air)")
        form.addRow("Wave Speed:", self.speed_combo)

        self.span_combo = QComboBox()
        self.span_combo.addItems(["10 cm x 10 cm", "12 cm x 12 cm", "16 cm x 16 cm"])
        self.span_combo.setCurrentText("10 cm x 10 cm")
        form.addRow("Field of View:", self.span_combo)

        self.angle_offset_combo = QComboBox()
        self.angle_offset_combo.addItems(["0°", "90°", "180°", "270°"])
        self.angle_offset_combo.setCurrentText("0°")
        form.addRow("Antenna Angle Offset:", self.angle_offset_combo)

        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(["CCW", "CW"])
        form.addRow("Antenna Rotation:", self.rotation_combo)

        self.flip_combo = QComboBox()
        self.flip_combo.addItems(["None", "Flip X", "Flip Y", "Flip X+Y"])
        form.addRow("Axis Flip:", self.flip_combo)

        self.arc_combo = QComboBox()
        self.arc_combo.addItems(["360°", "355° (BMID)"])
        form.addRow("Antenna Arc:", self.arc_combo)

        self.phase_delay_combo = QComboBox()
        self.phase_delay_combo.addItems(["Off", "On (BMID)"])
        form.addRow("Phase-delay Radius:", self.phase_delay_combo)

        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItems(["Peak score", "Prefer off-center"])
        self.roi_mode_combo.setCurrentText("Prefer off-center")
        form.addRow("ROI Mode:", self.roi_mode_combo)

        self.beamformer_mode_combo = QComboBox()
        self.beamformer_mode_combo.addItems(
            [
                "Quality score",
                "Prefer DMAS-D4",
                "Closest to tumor GT",
                "Force DAS",
                "Force DMAS",
                "Force DMAS-D4",
            ]
        )
        self.beamformer_mode_combo.setCurrentText("Prefer DMAS-D4")
        form.addRow("Beamformer Pick:", self.beamformer_mode_combo)

        self.auto_quick_check = QCheckBox("Quick search (fewer geometry combos)")
        self.auto_quick_check.setChecked(True)
        form.addRow("Auto Tweak:", self.auto_quick_check)

        self.auto_wave_check = QCheckBox("Also sweep wave speed (3.0 / 2.1 / 1.8 e8)")
        self.auto_wave_check.setChecked(False)
        form.addRow("", self.auto_wave_check)

        config_group.setLayout(form)
        layout.addWidget(config_group)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Reconstruction")
        self.run_button.clicked.connect(self.on_run_clicked)
        self.run_button.setEnabled(False)
        run_row.addWidget(self.run_button)

        self.auto_tweak_button = QPushButton("Auto Tweak Settings")
        self.auto_tweak_button.setToolTip(
            "Search angle / flip / arc / phase-delay / ROI / beamformer automatically, "
            "apply the best settings to the manual controls, then reconstruct."
        )
        self.auto_tweak_button.clicked.connect(self.on_auto_tweak_clicked)
        self.auto_tweak_button.setEnabled(False)
        run_row.addWidget(self.auto_tweak_button)

        self.export_report_button = QPushButton("Download Final Report")
        self.export_report_button.setToolTip("Export Markdown + JSON session report for review")
        self.export_report_button.setEnabled(False)
        run_row.addWidget(self.export_report_button)

        self.status_label = QLabel("No dataset loaded.")
        run_row.addWidget(self.status_label, stretch=1)
        layout.addLayout(run_row)

        self.auto_progress = QProgressBar()
        self.auto_progress.setValue(0)
        self.auto_progress.setVisible(False)
        layout.addWidget(self.auto_progress)

        self.content_tabs = QTabWidget()
        layout.addWidget(self.content_tabs, stretch=1)

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)

        selected_group = QGroupBox("Selected Reconstruction")
        selected_layout = QVBoxLayout()
        self.selected_figure = Figure(figsize=(7, 6))
        self.selected_canvas = FigureCanvasQTAgg(self.selected_figure)
        selected_layout.addWidget(self.selected_canvas)
        selected_group.setLayout(selected_layout)
        overview_layout.addWidget(selected_group, stretch=3)

        overview_bottom = QHBoxLayout()

        summary_group = QGroupBox("Selected Reconstruction Summary")
        summary_layout = QVBoxLayout()
        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        summary_layout.addWidget(self.summary_table)
        summary_group.setLayout(summary_layout)
        overview_bottom.addWidget(summary_group, stretch=1)

        roi_group = QGroupBox("ROI Localization")
        roi_layout = QVBoxLayout()
        self.roi_table = QTableWidget(0, 2)
        self.roi_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.setEditTriggers(QTableWidget.NoEditTriggers)
        roi_layout.addWidget(self.roi_table)
        roi_group.setLayout(roi_layout)
        overview_bottom.addWidget(roi_group, stretch=1)

        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout()
        self.log_view = QLabel()
        self.log_view.setWordWrap(True)
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        overview_bottom.addWidget(log_group, stretch=1)

        overview_layout.addLayout(overview_bottom, stretch=1)

        assumptions_group = QGroupBox("Reconstruction Assumptions")
        assumptions_layout = QVBoxLayout()
        self.assumptions_label = QLabel("No reconstruction assumptions available yet.")
        self.assumptions_label.setWordWrap(True)
        assumptions_layout.addWidget(self.assumptions_label)
        assumptions_group.setLayout(assumptions_layout)
        overview_layout.addWidget(assumptions_group)
        self.content_tabs.addTab(overview_tab, "Overview")

        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)

        plots_group = QGroupBox("Beamformer Comparison")
        plots_layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        plots_layout.addWidget(self.canvas)
        plots_group.setLayout(plots_layout)
        comparison_layout.addWidget(plots_group, stretch=3)

        metrics_group = QGroupBox("Beamformer Quality Metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_table = QTableWidget(0, 6)
        self.metrics_table.setHorizontalHeaderLabels(["Algorithm", "SNR", "SCR", "Contrast", "Time (s)", "Score"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        metrics_layout.addWidget(self.metrics_table)
        metrics_group.setLayout(metrics_layout)
        comparison_layout.addWidget(metrics_group, stretch=1)
        self.content_tabs.addTab(comparison_tab, "Beamformer Comparison")

        refinement_tab = QWidget()
        refinement_layout_root = QHBoxLayout(refinement_tab)

        refinement_plot_group = QGroupBox("High-Resolution ROI Reconstruction")
        refinement_plot_layout = QVBoxLayout()
        self.refined_figure = Figure(figsize=(10, 7.0))
        self.refined_canvas = FigureCanvasQTAgg(self.refined_figure)
        self.refined_canvas.setMinimumHeight(500)
        refinement_plot_layout.addWidget(self.refined_canvas)
        refinement_plot_group.setLayout(refinement_plot_layout)
        refinement_layout_root.addWidget(refinement_plot_group, stretch=4)

        refinement_info_group = QGroupBox("Refined ROI Details")
        refinement_info_layout = QVBoxLayout()
        self.refinement_table = QTableWidget(0, 2)
        self.refinement_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.refinement_table.horizontalHeader().setStretchLastSection(True)
        self.refinement_table.verticalHeader().setVisible(False)
        self.refinement_table.setEditTriggers(QTableWidget.NoEditTriggers)
        refinement_info_layout.addWidget(self.refinement_table)
        refinement_info_group.setLayout(refinement_info_layout)
        refinement_layout_root.addWidget(refinement_info_group, stretch=1)
        self.content_tabs.addTab(refinement_tab, "ROI Refinement")

    def _log(self, message: str, level: str = "INFO") -> None:
        line = self.status_log.add(message, level)
        self.log_view.setText(line)

    def set_dataset(self, dataset: MicrowaveDataset) -> None:
        self.dataset = dataset
        self.status_label.setText(f"Loaded dataset: {dataset.file_name}")
        self.run_button.setEnabled(True)
        self.auto_tweak_button.setEnabled(True)
        self._apply_dataset_defaults(dataset)
        self._log(f"Dataset '{dataset.file_name}' ready for reconstruction.")

    def set_processed_dataset(self, dataset: MicrowaveDataset) -> None:
        self.processed_dataset = dataset
        self._log("Processed dataset available for reconstruction.")

    def on_run_clicked(self) -> None:
        if self.dataset is None:
            self._log("No dataset loaded.", "ERROR")
            return

        source = self.data_source_combo.currentText()
        if source == "Processed Dataset" and self.processed_dataset is not None:
            dataset_to_use = self.processed_dataset
        else:
            dataset_to_use = self.dataset

        try:
            self._run_reconstruction(dataset_to_use)
        except Exception as exc:
            self._log(f"Reconstruction failed: {exc}", "ERROR")

    def on_auto_tweak_clicked(self) -> None:
        if self.auto_worker is not None and self.auto_worker.isRunning():
            self._log("Auto tweak already running.", "WARN")
            return

        dataset = self._dataset_for_reconstruction()
        if dataset is None:
            self._log("No dataset available for auto tweak.", "ERROR")
            return

        config = self._current_reconstruction_config(dataset)
        config.n_x = min(config.n_x, 64)
        config.n_y = min(config.n_y, 64)
        tumor_xy = self._tumor_xy_m(dataset)
        quick = self.auto_quick_check.isChecked()
        include_waves = self.auto_wave_check.isChecked()
        prefer_off = self.roi_mode_combo.currentText() == "Prefer off-center"

        self.run_button.setEnabled(False)
        self.auto_tweak_button.setEnabled(False)
        self.auto_progress.setVisible(True)
        self.auto_progress.setValue(0)
        objective = "tumor GT distance" if tumor_xy is not None else "image quality"
        mode = "quick" if quick else "full"
        self._log(f"Auto tweak started ({mode}, objective={objective})...")

        self.auto_worker = AutoCalibrateWorker(
            dataset.s_parameters,
            dataset.frequencies,
            config,
            tumor_xy,
            quick=quick,
            include_wave_speeds=include_waves,
            baseline_prefer_off_center=prefer_off,
            parent=self,
        )
        self.auto_worker.progress_updated.connect(self._on_auto_progress)
        self.auto_worker.finished_ok.connect(self._on_auto_finished)
        self.auto_worker.finished_error.connect(self._on_auto_failed)
        self.auto_worker.start()

    def _dataset_for_reconstruction(self) -> MicrowaveDataset | None:
        source = self.data_source_combo.currentText()
        if source == "Processed Dataset" and self.processed_dataset is not None:
            return self.processed_dataset
        return self.dataset

    def _on_auto_progress(self, percent: int, message: str) -> None:
        self.auto_progress.setValue(int(percent))
        self.status_label.setText(message)
        self._log(message)

    def _on_auto_failed(self, message: str) -> None:
        self.auto_progress.setVisible(False)
        self.run_button.setEnabled(self.dataset is not None)
        self.auto_tweak_button.setEnabled(self.dataset is not None)
        self._log(f"Auto tweak failed: {message}", "ERROR")

    def _on_auto_finished(self, result: AutoCalibrateResult) -> None:
        self.last_auto_result = result
        self.auto_progress.setValue(100)
        best = result.best
        dist_txt = (
            f"{best.tumor_gt_distance_m * 100:.2f} cm to GT"
            if best.tumor_gt_distance_m is not None
            else f"score={best.score:.4f}"
        )
        baseline_txt = ""
        if result.baseline is not None and result.baseline.tumor_gt_distance_m is not None:
            baseline_txt = f" (baseline {result.baseline.tumor_gt_distance_m * 100:.2f} cm)"

        self.run_button.setEnabled(True)
        self.auto_tweak_button.setEnabled(True)
        self.auto_progress.setVisible(False)

        if not result.improved:
            search = result.search_best
            search_txt = ""
            if search is not None and search.tumor_gt_distance_m is not None:
                search_txt = f" Best search candidate was {search.tumor_gt_distance_m * 100:.2f} cm."
            self._log(
                f"Auto tweak kept current settings{baseline_txt}.{search_txt} "
                f"No improvement over manual baseline ({len(result.trials)} trials).",
                "WARN",
            )
            return

        self._apply_auto_result_to_controls(result)
        self._log(
            f"Auto tweak improved → {best.beamformer} | {best.geometry.label()} | {dist_txt}"
            f"{baseline_txt} (from {len(result.trials)} trials). Reconstructing...",
            "SUCCESS",
        )

        dataset = self._dataset_for_reconstruction()
        if dataset is None:
            return
        try:
            self._run_reconstruction(dataset)
        except Exception as exc:
            self._log(f"Reconstruction after auto tweak failed: {exc}", "ERROR")

    def _apply_auto_result_to_controls(self, result: AutoCalibrateResult) -> None:
        g = result.best.geometry
        self.speed_combo.setCurrentText(self._speed_text(g.wave_speed))
        self.angle_offset_combo.setCurrentText(self._angle_offset_text(g.antenna_angle_offset_deg))
        self.rotation_combo.setCurrentText("CW" if g.antenna_clockwise else "CCW")
        self.flip_combo.setCurrentText(self._flip_text(g.antenna_flip_x, g.antenna_flip_y))
        self.arc_combo.setCurrentText("355° (BMID)" if abs(g.antenna_span_deg - 355.0) < 1e-3 else "360°")
        self.phase_delay_combo.setCurrentText("On (BMID)" if g.use_bmid_phase_delay_radius else "Off")
        self.roi_mode_combo.setCurrentText(
            "Prefer off-center" if g.prefer_off_center_roi else "Peak score"
        )
        force_map = {
            "DAS": "Force DAS",
            "DMAS": "Force DMAS",
            "DMAS-D4": "Force DMAS-D4",
        }
        self.beamformer_mode_combo.setCurrentText(force_map.get(result.best.beamformer, "Prefer DMAS-D4"))

    def _run_reconstruction(self, dataset: MicrowaveDataset) -> None:
        self._log("Starting reconstruction...")

        s_params = dataset.s_parameters
        freqs = dataset.frequencies
        config = self._current_reconstruction_config(dataset)
        tumor_xy = self._tumor_xy_m(dataset)
        prefer_off_center = self.roi_mode_combo.currentText() == "Prefer off-center"
        beamformer_mode = self._selected_beamformer_mode()

        images, timings = reconstruct_all(
            s_params,
            freqs,
            config=config,
            return_timings=True,
        )
        selected_name, quality_metrics = select_best_beamformer(
            images,
            timings,
            mode=beamformer_mode,
            x_span=config.x_span,
            y_span=config.y_span,
            tumor_xy_m=tumor_xy,
            prefer_off_center=prefer_off_center,
        )
        roi_result = detect_roi(
            images[selected_name],
            prefer_off_center=prefer_off_center,
            x_span=config.x_span,
            y_span=config.y_span,
        )
        refinement = reconstruct_high_resolution_roi(
            s_params,
            freqs,
            selected_name,
            roi_result.bounding_box,
            images[selected_name].shape,
            config=config,
            min_grid_size=max(96, config.n_x),
        )
        self.last_roi_result = roi_result
        self.last_roi_refinement = refinement

        centroid_m = roi_centroid_meters(
            roi_result,
            images[selected_name].shape,
            config.x_span,
            config.y_span,
        )
        gt_distance = None
        if tumor_xy is not None:
            gt_distance = float(
                np.hypot(centroid_m[0] - tumor_xy[0], centroid_m[1] - tumor_xy[1])
            )

        self.last_snapshot = ReconstructionSnapshot(
            selected_beamformer=selected_name,
            source_label=self.data_source_combo.currentText(),
            config=config,
            quality_metrics=quality_metrics,
            roi=roi_result,
            refinement=refinement,
            tumor_xy_m=tumor_xy,
            roi_centroid_m=centroid_m,
            tumor_gt_distance_m=gt_distance,
            image_peak=float(np.max(images[selected_name])),
            image_mean=float(np.mean(images[selected_name])),
            selection_mode=beamformer_mode,
            prefer_off_center_roi=prefer_off_center,
        )
        self.export_report_button.setEnabled(True)

        self._plot_selected_image(
            images[selected_name],
            selected_name,
            roi_result,
            config.x_span,
            config.y_span,
            tumor_xy_m=tumor_xy,
        )
        self._plot_images(
            images,
            selected_name,
            roi_result,
            config.x_span,
            config.y_span,
            tumor_xy_m=tumor_xy,
        )
        self._plot_refined_image(refinement)
        self._populate_summary(images, selected_name)
        self._populate_metrics(quality_metrics)
        self._populate_roi(roi_result)
        self._populate_refinement(refinement)
        gt_note = ""
        if tumor_xy is not None:
            gt_note = f" Tumor GT=({tumor_xy[0]*100:.2f}, {tumor_xy[1]*100:.2f}) cm."
            if gt_distance is not None:
                gt_note += f" ROI distance={gt_distance*100:.2f} cm."
        geom_note = (
            f" offset={config.antenna_angle_offset_deg:.0f}°, "
            f"arc={config.antenna_span_deg:.0f}°, "
            f"phase_delay={'on' if config.use_bmid_phase_delay_radius else 'off'}, "
            f"pick={beamformer_mode}."
        )
        self._log(
            f"Reconstruction completed. Selected beamformer: {selected_name}.{geom_note}"
            f" ROI bbox={roi_result.bounding_box}. Refined grid={refinement.grid_shape[1]}x{refinement.grid_shape[0]}.{gt_note}",
            "SUCCESS",
        )
        self.reconstruction_completed.emit(self.last_snapshot)

    def _tumor_xy_m(self, dataset: MicrowaveDataset) -> tuple[float, float] | None:
        meta = dataset.metadata or {}
        if not meta.get("bmid_has_tumor"):
            return None
        x_m = meta.get("tumor_x_m")
        y_m = meta.get("tumor_y_m")
        if x_m is None or y_m is None:
            return None
        return float(x_m), float(y_m)

    def save_figures(self, output_dir: str, basename: str) -> list[str]:
        """Save current reconstruction figures as PNGs; return written paths."""
        os.makedirs(output_dir, exist_ok=True)
        paths: list[str] = []
        selected_path = os.path.join(output_dir, f"{basename}_selected.png")
        compare_path = os.path.join(output_dir, f"{basename}_beamformers.png")
        refined_path = os.path.join(output_dir, f"{basename}_roi_refine.png")
        self.selected_figure.savefig(selected_path, dpi=140, bbox_inches="tight")
        self.figure.savefig(compare_path, dpi=140, bbox_inches="tight")
        self.refined_figure.savefig(refined_path, dpi=140, bbox_inches="tight")
        paths.extend([selected_path, compare_path, refined_path])
        return paths

    def _apply_dataset_defaults(self, dataset: MicrowaveDataset) -> None:
        assessment = infer_reconstruction_assessment(dataset)
        config = assessment.config
        self.radius_combo.setCurrentText(self._radius_text(config.antenna_radius))
        self.speed_combo.setCurrentText(self._speed_text(config.wave_speed))
        self.span_combo.setCurrentText(self._span_text(config.x_span, config.y_span))
        self.angle_offset_combo.setCurrentText(self._angle_offset_text(config.antenna_angle_offset_deg))
        self.rotation_combo.setCurrentText("CW" if config.antenna_clockwise else "CCW")
        self.flip_combo.setCurrentText(self._flip_text(config.antenna_flip_x, config.antenna_flip_y))
        self.arc_combo.setCurrentText("355° (BMID)" if abs(config.antenna_span_deg - 355.0) < 1e-3 else "360°")
        self.phase_delay_combo.setCurrentText("On (BMID)" if config.use_bmid_phase_delay_radius else "Off")
        meta = dataset.metadata or {}
        if str(meta.get("dataset_family", "")).upper() == "UM-BMID":
            self.roi_mode_combo.setCurrentText("Peak score")
            self.beamformer_mode_combo.setCurrentText(
                "Force DMAS-D4" if meta.get("bmid_has_tumor") else "Prefer DMAS-D4"
            )
            self.span_combo.setCurrentText("12 cm x 12 cm")
            self.phase_delay_combo.setCurrentText("Off")
            self.arc_combo.setCurrentText("360°")
            self.auto_quick_check.setChecked(True)
        self._update_assumption_summary(assessment.source_notes, assessment.warnings)

    def _current_reconstruction_config(self, dataset: MicrowaveDataset):
        config = infer_reconstruction_config(dataset)
        grid_size = self._selected_grid_size()
        config.n_x = grid_size
        config.n_y = grid_size
        config.antenna_radius = self._selected_radius_m()
        config.wave_speed = self._selected_wave_speed()
        config.x_span, config.y_span = self._selected_span()
        config.antenna_angle_offset_deg = self._selected_angle_offset_deg()
        config.antenna_clockwise = self.rotation_combo.currentText() == "CW"
        config.antenna_flip_x, config.antenna_flip_y = self._selected_flips()
        config.antenna_span_deg = 355.0 if "355" in self.arc_combo.currentText() else 360.0
        config.use_bmid_phase_delay_radius = self.phase_delay_combo.currentText().startswith("On")
        return config

    def _selected_beamformer_mode(self) -> str:
        text = self.beamformer_mode_combo.currentText()
        return {
            "Quality score": "quality",
            "Prefer DMAS-D4": "prefer_dmas_d4",
            "Closest to tumor GT": "tumor_gt",
            "Force DAS": "force_das",
            "Force DMAS": "force_dmas",
            "Force DMAS-D4": "force_dmas_d4",
        }.get(text, "quality")

    def _selected_angle_offset_deg(self) -> float:
        text = self.angle_offset_combo.currentText().replace("°", "").strip()
        try:
            return float(text)
        except ValueError:
            return 0.0

    def _selected_flips(self) -> tuple[bool, bool]:
        text = self.flip_combo.currentText()
        return {
            "None": (False, False),
            "Flip X": (True, False),
            "Flip Y": (False, True),
            "Flip X+Y": (True, True),
        }.get(text, (False, False))

    def _angle_offset_text(self, degrees: float) -> str:
        mapping = {0.0: "0°", 90.0: "90°", 180.0: "180°", 270.0: "270°"}
        for key, label in mapping.items():
            if abs(float(degrees) - key) < 1e-6:
                return label
        return "0°"

    def _flip_text(self, flip_x: bool, flip_y: bool) -> str:
        if flip_x and flip_y:
            return "Flip X+Y"
        if flip_x:
            return "Flip X"
        if flip_y:
            return "Flip Y"
        return "None"

    def _update_assumption_summary(self, source_notes: dict[str, str], warnings: list[str]) -> None:
        lines = [f"{key}: {value}" for key, value in source_notes.items()]
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in warnings)
        else:
            lines.append("No fallback assumptions detected.")
        self.assumptions_label.setText("\n".join(lines))

    def _selected_grid_size(self) -> int:
        text = self.grid_combo.currentText()
        try:
            return int(text.split("x", maxsplit=1)[0])
        except (TypeError, ValueError, IndexError):
            return 64

    def _selected_radius_m(self) -> float:
        text = self.radius_combo.currentText()
        return {
            "8 cm": 0.08,
            "10 cm": 0.10,
            "12 cm": 0.12,
            "18 cm": 0.18,
        }.get(text, 0.08)

    def _selected_wave_speed(self) -> float:
        text = self.speed_combo.currentText()
        return {
            "3.0e8 m/s (air)": 3.0e8,
            "2.1e8 m/s": 2.1e8,
            "1.8e8 m/s": 1.8e8,
        }.get(text, 3.0e8)

    def _selected_span(self) -> tuple[tuple[float, float], tuple[float, float]]:
        text = self.span_combo.currentText()
        extent_cm = {
            "10 cm x 10 cm": 0.05,
            "12 cm x 12 cm": 0.06,
            "16 cm x 16 cm": 0.08,
        }.get(text, 0.05)
        return (-extent_cm, extent_cm), (-extent_cm, extent_cm)

    def _radius_text(self, radius_m: float) -> str:
        value_cm = int(round(radius_m * 100.0))
        mapping = {8: "8 cm", 10: "10 cm", 12: "12 cm", 18: "18 cm"}
        return mapping.get(value_cm, "8 cm")

    def _speed_text(self, wave_speed: float) -> str:
        if abs(wave_speed - 2.1e8) < 1e6:
            return "2.1e8 m/s"
        if abs(wave_speed - 1.8e8) < 1e6:
            return "1.8e8 m/s"
        return "3.0e8 m/s (air)"

    def _span_text(self, x_span: tuple[float, float], y_span: tuple[float, float]) -> str:
        half_span = max(abs(float(x_span[0])), abs(float(x_span[1])), abs(float(y_span[0])), abs(float(y_span[1])))
        mapping = {
            0.05: "10 cm x 10 cm",
            0.06: "12 cm x 12 cm",
            0.08: "16 cm x 16 cm",
        }
        for key, value in mapping.items():
            if abs(half_span - key) < 1e-6:
                return value
        return "10 cm x 10 cm"

    def _configure_image_axes(
        self,
        ax,
        image: np.ndarray,
        title: str,
        x_span: tuple[float, float] | None = None,
        y_span: tuple[float, float] | None = None,
    ) -> None:
        vmin = float(np.percentile(image, 5))
        vmax = float(np.percentile(image, 99))
        if vmax <= vmin:
            vmin = float(np.min(image))
            vmax = float(np.max(image) + 1e-12)
        extent = None
        if x_span is not None and y_span is not None:
            extent = [x_span[0] * 100.0, x_span[1] * 100.0, y_span[0] * 100.0, y_span[1] * 100.0]
        im = ax.imshow(
            image,
            cmap="inferno",
            origin="lower",
            aspect="equal",
            interpolation="bicubic",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        ax.set_title(title)
        if extent is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
        return im

    def _plot_selected_image(
        self,
        image: np.ndarray,
        selected_name: str,
        roi_result: ROIResult,
        x_span: tuple[float, float],
        y_span: tuple[float, float],
        tumor_xy_m: tuple[float, float] | None = None,
    ) -> None:
        self.selected_figure.clear()
        ax = self.selected_figure.add_subplot(1, 1, 1)
        im = self._configure_image_axes(ax, image, f"Selected Reconstruction: {selected_name}", x_span=x_span, y_span=y_span)
        x0, y0, x1, y1 = roi_result.bounding_box
        x_axis = np.linspace(x_span[0] * 100.0, x_span[1] * 100.0, image.shape[1])
        y_axis = np.linspace(y_span[0] * 100.0, y_span[1] * 100.0, image.shape[0])
        rect = Rectangle(
            (x_axis[max(0, x0)], y_axis[max(0, y0)]),
            x_axis[min(image.shape[1] - 1, x1 - 1)] - x_axis[max(0, x0)],
            y_axis[min(image.shape[0] - 1, y1 - 1)] - y_axis[max(0, y0)],
            linewidth=2.0,
            edgecolor="cyan",
            facecolor="none",
        )
        ax.add_patch(rect)
        centroid_x = np.interp(roi_result.centroid[0], np.arange(image.shape[1]), x_axis)
        centroid_y = np.interp(roi_result.centroid[1], np.arange(image.shape[0]), y_axis)
        ax.plot(centroid_x, centroid_y, "wo", markersize=5, label="ROI centroid")
        if tumor_xy_m is not None:
            ax.plot(
                tumor_xy_m[0] * 100.0,
                tumor_xy_m[1] * 100.0,
                marker="x",
                markersize=10,
                markeredgewidth=2.0,
                color="lime",
                label="Tumor GT",
            )
            ax.legend(loc="upper right", fontsize=8)
        self.selected_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.selected_figure.tight_layout()
        self.selected_canvas.draw()

    def _plot_images(
        self,
        images: dict[str, np.ndarray],
        selected_name: str,
        roi_result: ROIResult,
        x_span: tuple[float, float],
        y_span: tuple[float, float],
        tumor_xy_m: tuple[float, float] | None = None,
    ) -> None:
        self.figure.clear()
        titles = ["DAS", "DMAS", "DMAS-D4", f"Selected: {selected_name}"]
        keys = ["DAS", "DMAS", "DMAS-D4", selected_name]
        x_axis = np.linspace(x_span[0] * 100.0, x_span[1] * 100.0, images[selected_name].shape[1])
        y_axis = np.linspace(y_span[0] * 100.0, y_span[1] * 100.0, images[selected_name].shape[0])

        for idx, key in enumerate(keys, start=1):
            ax = self.figure.add_subplot(2, 2, idx)
            self._configure_image_axes(ax, images[key], titles[idx - 1], x_span=x_span, y_span=y_span)
            x0, y0, x1, y1 = roi_result.bounding_box
            rect = Rectangle(
                (x_axis[max(0, x0)], y_axis[max(0, y0)]),
                x_axis[min(images[key].shape[1] - 1, x1 - 1)] - x_axis[max(0, x0)],
                y_axis[min(images[key].shape[0] - 1, y1 - 1)] - y_axis[max(0, y0)],
                linewidth=1.5,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)
            if tumor_xy_m is not None:
                ax.plot(
                    tumor_xy_m[0] * 100.0,
                    tumor_xy_m[1] * 100.0,
                    marker="x",
                    markersize=8,
                    markeredgewidth=1.8,
                    color="lime",
                )

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_refined_image(self, refinement: ROIRefinement) -> None:
        self.refined_figure.clear()
        ax = self.refined_figure.add_subplot(1, 1, 1)
        im = self._configure_image_axes(
            ax,
            refinement.image,
            f"{refinement.algorithm} ROI Refinement",
            x_span=refinement.x_span,
            y_span=refinement.y_span,
        )
        self.refined_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.refined_figure.tight_layout()
        self.refined_canvas.draw()

    def _populate_summary(self, images: dict[str, np.ndarray], selected_name: str) -> None:
        selected = images[selected_name]
        mean_val = float(np.mean(selected))
        std_val = float(np.std(selected))
        peak_val = float(np.max(selected))
        contrast = float((np.max(selected) - np.min(selected)) / (np.max(selected) + np.min(selected) + 1e-12))

        info = {
            "Selected Beamformer": selected_name,
            "Mean Intensity": f"{mean_val:.5f}",
            "Std Intensity": f"{std_val:.5f}",
            "Peak Intensity": f"{peak_val:.5f}",
            "Contrast": f"{contrast:.5f}",
        }

        self.summary_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.summary_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.summary_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.summary_table.resizeColumnsToContents()

    def _populate_metrics(self, quality_metrics: dict[str, dict[str, float]]) -> None:
        self.metrics_table.setRowCount(len(quality_metrics))
        for row, (name, metrics) in enumerate(quality_metrics.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{metrics['snr']:.4f}"))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{metrics['scr']:.4f}"))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{metrics['contrast']:.4f}"))
            self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{metrics['time']:.4f}"))
            self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{metrics['score']:.4f}"))
        self.metrics_table.resizeColumnsToContents()

    def _populate_roi(self, roi_result: ROIResult) -> None:
        x0, y0, x1, y1 = roi_result.bounding_box
        info = {
            "Bounding Box": f"({x0}, {y0}) - ({x1}, {y1})",
            "Centroid": f"({roi_result.centroid[0]:.2f}, {roi_result.centroid[1]:.2f})",
            "Area": str(roi_result.area),
            "Threshold": f"{roi_result.threshold:.2f}",
            "ROI Score": f"{roi_result.score:.5f}",
        }

        self.roi_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.roi_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.roi_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.roi_table.resizeColumnsToContents()

    def _populate_refinement(self, refinement: ROIRefinement) -> None:
        x0, x1 = refinement.x_span
        y0, y1 = refinement.y_span
        info = {
            "Algorithm": refinement.algorithm,
            "Grid": f"{refinement.grid_shape[1]} x {refinement.grid_shape[0]}",
            "X Span (m)": f"{x0:.4f} to {x1:.4f}",
            "Y Span (m)": f"{y0:.4f} to {y1:.4f}",
            "Mean Intensity": f"{float(np.mean(refinement.image)):.5f}",
            "Peak Intensity": f"{float(np.max(refinement.image)):.5f}",
        }

        self.refinement_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.refinement_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.refinement_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.refinement_table.resizeColumnsToContents()

"""GUI panel for beamforming reconstruction and image display."""

from __future__ import annotations

from pathlib import Path

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from data_loader.dataset_info import MicrowaveDataset
from gui.plotting import finish_figure, make_gui_figure, plot_complex_matrix, plot_reconstruction_trace, show_message
from gui.styles import set_page_title, set_primary_button
from quality.beamformer_selector import select_best_beamformer
from reconstruction.auto_calibrate import (
    AutoCalibrateResult,
    auto_calibrate_reconstruction,
)
from reconstruction.reconstruction_manager import (
    ROIRefinement,
    infer_reconstruction_assessment,
    infer_reconstruction_config,
    reconstruct_all,
    reconstruct_high_resolution_roi,
)
from quality.characterization import characterize_tumor_region
from quality.confidence import assess_reconstruction_confidence
from quality.tumor_taxonomy import taxonomy_from_metadata
from reconstruction.ifft import frequency_to_time, time_axis_seconds
from roi.roi_detector import (
    ROIResult,
    detect_rois,
    evaluate_rois,
)
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
        baseline_tight_peak: bool = False,
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
        self.baseline_tight_peak = baseline_tight_peak

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
                baseline_tight_peak=self.baseline_tight_peak,
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
        self.last_roi_results: list[ROIResult] = []
        self.last_roi_refinement: ROIRefinement | None = None
        self.last_snapshot: ReconstructionSnapshot | None = None
        self.last_auto_result: AutoCalibrateResult | None = None
        self.auto_worker: AutoCalibrateWorker | None = None
        self.last_time_signals = None
        self.last_time_axis = None
        self.last_images: dict = {}
        self.last_s_params = None
        self.last_frequencies = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel()
        set_page_title(title, "Reconstruction")
        title_col.addWidget(title)
        self.status_label = QLabel("Load a dataset to begin.")
        self.status_label.setObjectName("hintLabel")
        title_col.addWidget(self.status_label)
        header.addLayout(title_col, stretch=1)

        self.run_button = QPushButton("Run")
        set_primary_button(self.run_button)
        self.run_button.clicked.connect(self.on_run_clicked)
        self.run_button.setEnabled(False)
        header.addWidget(self.run_button)

        self.auto_tweak_button = QPushButton("Auto Tweak")
        self.auto_tweak_button.setToolTip(
            "Search geometry / ROI / beamformer, apply best settings, then reconstruct."
        )
        self.auto_tweak_button.clicked.connect(self.on_auto_tweak_clicked)
        self.auto_tweak_button.setEnabled(False)
        header.addWidget(self.auto_tweak_button)

        self.export_report_button = QPushButton("Report")
        self.export_report_button.setToolTip("Export Markdown + JSON session report")
        self.export_report_button.setEnabled(False)
        header.addWidget(self.export_report_button)
        layout.addLayout(header)

        self.auto_progress = QProgressBar()
        self.auto_progress.setValue(0)
        self.auto_progress.setVisible(False)
        self.auto_progress.setTextVisible(False)
        layout.addWidget(self.auto_progress)

        self.content_tabs = QTabWidget()
        layout.addWidget(self.content_tabs, stretch=1)

        # --- Image ---
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(4, 12, 4, 4)
        self.selected_figure = make_gui_figure(7.5, 6.0)
        self.selected_canvas = FigureCanvasQTAgg(self.selected_figure)
        self.selected_canvas.setMinimumHeight(360)
        self.selected_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        overview_layout.addWidget(self.selected_canvas)
        self.content_tabs.addTab(overview_tab, "Image")

        trace_tab = QWidget()
        trace_layout = QVBoxLayout(trace_tab)
        trace_layout.setContentsMargins(4, 12, 4, 4)
        self.steps_figure = make_gui_figure(9.0, 6.4)
        self.steps_canvas = FigureCanvasQTAgg(self.steps_figure)
        self.steps_canvas.setMinimumHeight(420)
        self.steps_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        trace_layout.addWidget(self.steps_canvas)
        self.content_tabs.addTab(trace_tab, "Trace")

        # --- Settings ---
        settings_tab = QWidget()
        settings_tab.setAutoFillBackground(True)
        settings_outer = QVBoxLayout(settings_tab)
        settings_outer.setContentsMargins(8, 16, 8, 8)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        settings_scroll.setAutoFillBackground(True)
        settings_host = QWidget()
        settings_host.setAutoFillBackground(True)
        settings_host.setStyleSheet("background-color: #ffffff; color: #0f172a;")
        settings_grid = QHBoxLayout(settings_host)
        settings_grid.setSpacing(32)

        geom = QGroupBox("Geometry")
        geom_form = QFormLayout()
        geom_form.setSpacing(10)
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Processed Dataset", "Raw Dataset"])
        geom_form.addRow("Source", self.data_source_combo)
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["64x64", "128x128"])
        geom_form.addRow("Grid", self.grid_combo)
        self.radius_combo = QComboBox()
        self.radius_combo.addItems(["8 cm", "10 cm", "12 cm", "18 cm"])
        self.radius_combo.setCurrentText("8 cm")
        geom_form.addRow("Antenna radius", self.radius_combo)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["3.0e8 m/s (air)", "2.1e8 m/s", "1.8e8 m/s"])
        self.speed_combo.setCurrentText("3.0e8 m/s (air)")
        geom_form.addRow("Wave speed", self.speed_combo)
        self.span_combo = QComboBox()
        self.span_combo.addItems(["10 cm x 10 cm", "12 cm x 12 cm", "16 cm x 16 cm"])
        self.span_combo.setCurrentText("10 cm x 10 cm")
        geom_form.addRow("Field of view", self.span_combo)
        self.angle_offset_combo = QComboBox()
        self.angle_offset_combo.addItems(["0°", "90°", "180°", "270°"])
        self.angle_offset_combo.setCurrentText("0°")
        geom_form.addRow("Angle offset", self.angle_offset_combo)
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(["CCW", "CW"])
        geom_form.addRow("Rotation", self.rotation_combo)
        self.flip_combo = QComboBox()
        self.flip_combo.addItems(["None", "Flip X", "Flip Y", "Flip X+Y"])
        geom_form.addRow("Axis flip", self.flip_combo)
        self.arc_combo = QComboBox()
        self.arc_combo.addItems(["360°", "355° (BMID)"])
        geom_form.addRow("Antenna arc", self.arc_combo)
        self.phase_delay_combo = QComboBox()
        self.phase_delay_combo.addItems(["Off", "On (BMID)"])
        geom_form.addRow("Phase-delay radius", self.phase_delay_combo)
        geom.setLayout(geom_form)
        settings_grid.addWidget(geom)

        imaging = QGroupBox("Imaging")
        imaging_form = QFormLayout()
        imaging_form.setSpacing(10)
        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItems(["Peak score", "Prefer off-center", "Tight peak"])
        self.roi_mode_combo.setCurrentText("Peak score")
        imaging_form.addRow("ROI mode", self.roi_mode_combo)
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
        imaging_form.addRow("Beamformer pick", self.beamformer_mode_combo)
        self.auto_quick_check = QCheckBox("Quick Auto Tweak search")
        self.auto_quick_check.setChecked(True)
        imaging_form.addRow("", self.auto_quick_check)
        self.auto_wave_check = QCheckBox("Also sweep wave speed")
        self.auto_wave_check.setChecked(False)
        imaging_form.addRow("", self.auto_wave_check)
        tip = QLabel(
            "BMID tip: radius 18 cm, FOV 12×12 cm, c = 3.0e8, Peak ROI, Prefer DMAS-D4."
        )
        tip.setObjectName("hintLabel")
        tip.setWordWrap(True)
        imaging_form.addRow(tip)
        imaging.setLayout(imaging_form)
        settings_grid.addWidget(imaging)
        settings_grid.addStretch(1)

        settings_scroll.setWidget(settings_host)
        settings_outer.addWidget(settings_scroll)
        self.content_tabs.addTab(settings_tab, "Settings")

        # --- Details ---
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        details_layout.setContentsMargins(8, 16, 8, 8)
        details_layout.setSpacing(12)
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        summary_col = QVBoxLayout()
        summary_col.addWidget(QLabel("Summary"))
        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        summary_col.addWidget(self.summary_table)
        top_row.addLayout(summary_col, stretch=1)

        roi_col = QVBoxLayout()
        roi_col.addWidget(QLabel("ROI"))
        self.roi_table = QTableWidget(0, 2)
        self.roi_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.setEditTriggers(QTableWidget.NoEditTriggers)
        roi_col.addWidget(self.roi_table)
        top_row.addLayout(roi_col, stretch=1)
        details_layout.addLayout(top_row, stretch=2)

        details_layout.addWidget(QLabel("All detected spots (multiple ROIs)"))
        self.spots_table = QTableWidget(0, 8)
        self.spots_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "x (cm)",
                "y (cm)",
                "ROI score",
                "Thresh",
                "Suspicion",
                "Candidate",
                "GT (cm)",
            ]
        )
        self.spots_table.horizontalHeader().setStretchLastSection(True)
        self.spots_table.verticalHeader().setVisible(False)
        self.spots_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.spots_table.setMaximumHeight(180)
        details_layout.addWidget(self.spots_table)

        self.assumptions_label = QLabel("No reconstruction assumptions yet.")
        self.assumptions_label.setObjectName("hintLabel")
        self.assumptions_label.setWordWrap(True)
        details_layout.addWidget(self.assumptions_label)

        self.log_view = QLabel()
        self.log_view.setObjectName("hintLabel")
        self.log_view.setWordWrap(True)
        details_layout.addWidget(self.log_view)
        details_layout.addStretch(1)
        self.content_tabs.addTab(details_tab, "Details")

        tweak_tab = QWidget()
        self.auto_tweak_tab = tweak_tab
        tweak_layout = QVBoxLayout(tweak_tab)
        tweak_layout.setContentsMargins(8, 16, 8, 8)
        self.auto_tweak_notes = QLabel(
            "Run Auto Tweak to search geometry / beamformer settings. "
            "This table shows scores, the objective, and why a winner was kept."
        )
        self.auto_tweak_notes.setObjectName("hintLabel")
        self.auto_tweak_notes.setWordWrap(True)
        tweak_layout.addWidget(self.auto_tweak_notes)
        self.auto_tweak_table = QTableWidget(0, 8)
        self.auto_tweak_table.setHorizontalHeaderLabels(
            [
                "Mark",
                "Beamformer",
                "Score",
                "GT dist (cm)",
                "Quality",
                "Peak",
                "Contrast",
                "Geometry",
            ]
        )
        self.auto_tweak_table.horizontalHeader().setStretchLastSection(True)
        self.auto_tweak_table.verticalHeader().setVisible(False)
        self.auto_tweak_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tweak_layout.addWidget(self.auto_tweak_table)
        self.content_tabs.addTab(tweak_tab, "Auto Tweak")

        # --- Comparison ---
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)
        comparison_layout.setContentsMargins(4, 12, 4, 4)
        self.figure = make_gui_figure(8.8, 6.4)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(380)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        comparison_layout.addWidget(self.canvas, stretch=3)
        self.metrics_table = QTableWidget(0, 8)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Algorithm", "Picked", "SNR", "SCR", "Contrast", "Score", "GT (cm)", "Why"]
        )
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setMaximumHeight(160)
        comparison_layout.addWidget(self.metrics_table, stretch=1)
        self.content_tabs.addTab(comparison_tab, "Compare")

        # --- ROI refine ---
        refinement_tab = QWidget()
        refinement_layout_root = QHBoxLayout(refinement_tab)
        refinement_layout_root.setContentsMargins(4, 12, 4, 4)
        refinement_layout_root.setSpacing(16)
        self.refined_figure = make_gui_figure(8.0, 6.0)
        self.refined_canvas = FigureCanvasQTAgg(self.refined_figure)
        self.refined_canvas.setMinimumHeight(360)
        self.refined_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        refinement_layout_root.addWidget(self.refined_canvas, stretch=3)
        refine_side = QVBoxLayout()
        refine_side.addWidget(QLabel("ROI details"))
        self.refinement_table = QTableWidget(0, 2)
        self.refinement_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.refinement_table.horizontalHeader().setStretchLastSection(True)
        self.refinement_table.verticalHeader().setVisible(False)
        self.refinement_table.setEditTriggers(QTableWidget.NoEditTriggers)
        refine_side.addWidget(self.refinement_table)
        refinement_layout_root.addLayout(refine_side, stretch=1)
        self.content_tabs.addTab(refinement_tab, "ROI refine")

        self.content_tabs.setCurrentIndex(0)

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
        tight_peak = self.roi_mode_combo.currentText() == "Tight peak"

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
            baseline_tight_peak=tight_peak,
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
        self._populate_auto_tweak(result)
        best = result.best
        dist_txt = (
            f"{best.tumor_gt_distance_m * 100:.2f} cm to GT"
            if best.tumor_gt_distance_m is not None
            else f"score={best.score:.4f}"
        )
        baseline_txt = ""
        if (
            result.baseline is not None
            and result.baseline.tumor_gt_distance_m is not None
        ):
            baseline_txt = (
                f" (baseline {result.baseline.tumor_gt_distance_m * 100:.2f} cm)"
            )

        self.run_button.setEnabled(True)
        self.auto_tweak_button.setEnabled(True)
        self.auto_progress.setVisible(False)
        self.content_tabs.setCurrentWidget(self.auto_tweak_tab)

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

    def _populate_auto_tweak(self, result: AutoCalibrateResult) -> None:
        best = result.best
        baseline = result.baseline
        search = result.search_best
        if result.objective == "tumor_gt":
            rule = (
                "Objective: minimize ROI ↔ labeled tumor distance. "
                "A candidate replaces the current settings only if it is at least 0.5 mm closer."
            )
        else:
            rule = (
                "Objective: maximize contrast × off-center compact ROI score "
                "(no tumor GT). Winner must beat baseline score by > 1e-4."
            )
        winner_line = (
            f"Kept: {best.beamformer} | {best.geometry.label()} | score={best.score:.4f}"
        )
        if best.tumor_gt_distance_m is not None:
            winner_line += f" | GT={best.tumor_gt_distance_m * 100:.2f} cm"
        if not result.improved:
            winner_line += " (baseline kept — search did not beat the improvement rule)"
        notes = " ".join(result.notes)
        self.auto_tweak_notes.setText(
            f"{rule}\n{winner_line}\nTrials evaluated: {len(result.trials)}. {notes}"
        )

        def _key(trial):
            return (
                trial.beamformer,
                trial.geometry.label(),
                round(trial.score, 8),
            )

        unique = []
        seen = set()
        ordered = sorted(result.trials, key=lambda t: t.score, reverse=True)
        for trial in ordered:
            key = _key(trial)
            if key in seen:
                continue
            seen.add(key)
            unique.append(trial)
            if len(unique) >= 25:
                break

        self.auto_tweak_table.setRowCount(len(unique))
        for row, trial in enumerate(unique):
            mark = []
            if baseline is not None and _key(trial) == _key(baseline):
                mark.append("BASE")
            if _key(trial) == _key(best):
                mark.append("WIN")
            if search is not None and _key(trial) == _key(search) and _key(trial) != _key(best):
                mark.append("SEARCH-BEST")
            gt = trial.tumor_gt_distance_m
            self.auto_tweak_table.setItem(row, 0, QTableWidgetItem("+".join(mark) or ""))
            self.auto_tweak_table.setItem(row, 1, QTableWidgetItem(trial.beamformer))
            self.auto_tweak_table.setItem(row, 2, QTableWidgetItem(f"{trial.score:.5f}"))
            self.auto_tweak_table.setItem(
                row, 3, QTableWidgetItem("n/a" if gt is None else f"{gt * 100:.2f}")
            )
            self.auto_tweak_table.setItem(
                row, 4, QTableWidgetItem(f"{trial.quality_score:.4f}")
            )
            self.auto_tweak_table.setItem(row, 5, QTableWidgetItem(f"{trial.image_peak:.4f}"))
            self.auto_tweak_table.setItem(
                row, 6, QTableWidgetItem(f"{trial.image_contrast:.4f}")
            )
            self.auto_tweak_table.setItem(row, 7, QTableWidgetItem(trial.geometry.label()))
        self.auto_tweak_table.resizeColumnsToContents()

    def _apply_auto_result_to_controls(self, result: AutoCalibrateResult) -> None:
        g = result.best.geometry
        self.speed_combo.setCurrentText(self._speed_text(g.wave_speed))
        self.angle_offset_combo.setCurrentText(
            self._angle_offset_text(g.antenna_angle_offset_deg)
        )
        self.rotation_combo.setCurrentText("CW" if g.antenna_clockwise else "CCW")
        self.flip_combo.setCurrentText(
            self._flip_text(g.antenna_flip_x, g.antenna_flip_y)
        )
        self.arc_combo.setCurrentText(
            "355° (BMID)" if abs(g.antenna_span_deg - 355.0) < 1e-3 else "360°"
        )
        self.phase_delay_combo.setCurrentText(
            "On (BMID)" if g.use_bmid_phase_delay_radius else "Off"
        )
        self.roi_mode_combo.setCurrentText(
            "Tight peak"
            if g.tight_peak_roi
            else ("Prefer off-center" if g.prefer_off_center_roi else "Peak score")
        )
        force_map = {
            "DAS": "Force DAS",
            "DMAS": "Force DMAS",
            "DMAS-D4": "Force DMAS-D4",
        }
        self.beamformer_mode_combo.setCurrentText(
            force_map.get(result.best.beamformer, "Prefer DMAS-D4")
        )

    def _run_reconstruction(self, dataset: MicrowaveDataset) -> None:
        self._log("Starting reconstruction...")

        s_params = dataset.s_parameters
        freqs = dataset.frequencies
        config = self._current_reconstruction_config(dataset)
        tumor_xy = self._tumor_xy_m(dataset)
        prefer_off_center = self.roi_mode_combo.currentText() == "Prefer off-center"
        tight_peak = self.roi_mode_combo.currentText() == "Tight peak"
        beamformer_mode = self._selected_beamformer_mode()

        images, timings = reconstruct_all(
            s_params,
            freqs,
            config=config,
            return_timings=True,
        )
        try:
            time_signals = frequency_to_time(
                s_params, freqs, zero_padding=config.zero_padding
            )
            time_s = time_axis_seconds(freqs, time_signals.shape[0])
        except Exception as exc:
            self._log(f"IFFT intermediate plot skipped: {exc}")
            time_signals = None
            time_s = None
        self.last_time_signals = time_signals
        self.last_time_axis = time_s
        self.last_images = images
        self.last_s_params = np.asarray(s_params)
        self.last_frequencies = np.asarray(freqs)
        selected_name, quality_metrics = select_best_beamformer(
            images,
            timings,
            mode=beamformer_mode,
            x_span=config.x_span,
            y_span=config.y_span,
            tumor_xy_m=tumor_xy,
            prefer_off_center=prefer_off_center,
            tight_peak=tight_peak,
        )
        rois = detect_rois(
            images[selected_name],
            prefer_off_center=prefer_off_center,
            tight_peak=tight_peak,
            x_span=config.x_span,
            y_span=config.y_span,
            prior_xy_m=tumor_xy,
            prior_weight=0.0,
            max_rois=4,
            min_score_ratio=0.28,
        )
        evaluated = evaluate_rois(
            images[selected_name],
            rois,
            config.x_span,
            config.y_span,
            tumor_xy_m=tumor_xy,
        )
        roi_result = evaluated.localization_roi
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
        self.last_roi_results = list(evaluated.rois)
        self.last_roi_refinement = refinement

        centroid_m = evaluated.localization_centroid_m
        gt_distance = None if evaluated.gt_distance_cm is None else evaluated.gt_distance_cm / 100.0
        candidate = evaluated.combined
        characterization = characterize_tumor_region(
            images[selected_name],
            roi_result,
            x_span=config.x_span,
            y_span=config.y_span,
            centroid_m=centroid_m,
        )
        candidate_payload = {
            "is_tumor_candidate": candidate.is_tumor_candidate,
            "confidence": candidate.confidence,
            "suspicion_score": candidate.suspicion_score,
            "threshold": candidate.threshold,
            "reason": candidate.reason,
            "features": dict(candidate.features),
            "n_rois": len(evaluated.rois),
            "spots": evaluated.spot_summaries,
        }
        confidence = assess_reconstruction_confidence(
            selected_quality=quality_metrics.get(selected_name),
            all_quality_metrics=quality_metrics,
            selected_beamformer=selected_name,
            tumor_candidate=candidate_payload,
            characterization=characterization.to_dict(),
            tumor_gt_distance_m=gt_distance,
        )

        self.last_snapshot = ReconstructionSnapshot(
            selected_beamformer=selected_name,
            source_label=self.data_source_combo.currentText(),
            config=config,
            quality_metrics=quality_metrics,
            roi=roi_result,
            rois=list(evaluated.rois),
            refinement=refinement,
            tumor_xy_m=tumor_xy,
            roi_centroid_m=centroid_m,
            tumor_gt_distance_m=gt_distance,
            image_peak=float(np.max(images[selected_name])),
            image_mean=float(np.mean(images[selected_name])),
            selection_mode=beamformer_mode,
            prefer_off_center_roi=prefer_off_center,
            selected_image=np.asarray(images[selected_name]).copy(),
            beamformer_images={
                name: np.asarray(image).copy() for name, image in images.items()
            },
            tumor_candidate_result=candidate_payload,
            tumor_characterization=characterization.to_dict(),
            reconstruction_confidence=confidence.to_dict(),
        )
        self.export_report_button.setEnabled(True)

        self._plot_selected_image(
            images[selected_name],
            selected_name,
            evaluated.rois,
            config.x_span,
            config.y_span,
            tumor_xy_m=tumor_xy,
        )
        self._plot_images(
            images,
            selected_name,
            evaluated.rois,
            config.x_span,
            config.y_span,
            tumor_xy_m=tumor_xy,
        )
        self._plot_refined_image(refinement)
        self._plot_reconstruction_trace(
            np.asarray(s_params),
            np.asarray(freqs),
            time_s,
            time_signals,
            images,
            selected_name,
            config.x_span,
            config.y_span,
        )
        self._populate_summary(images, selected_name, dataset)
        self._populate_metrics(quality_metrics, selected_name)
        self._populate_roi(
            roi_result,
            candidate_result=self.last_snapshot.tumor_candidate_result,
            characterization=self.last_snapshot.tumor_characterization,
            confidence=self.last_snapshot.reconstruction_confidence,
            extra_rois=evaluated.rois,
            spot_summaries=evaluated.spot_summaries,
            localization_roi=evaluated.localization_roi,
        )
        self._populate_refinement(refinement)
        gt_note = ""
        if tumor_xy is not None:
            gt_note = (
                f" Tumor GT=({tumor_xy[0] * 100:.2f}, {tumor_xy[1] * 100:.2f}) cm."
            )
            if gt_distance is not None:
                gt_note += f" ROI distance={gt_distance * 100:.2f} cm."
        candidate_note = (
            f" Candidate={'YES' if candidate.is_tumor_candidate else 'NO'}"
            f" (conf={candidate.confidence:.2f})."
            f" Overall confidence={confidence.overall:.2f} ({confidence.label})."
            f" Diam={characterization.equivalent_diameter_cm:.2f} cm."
        )
        geom_note = (
            f" offset={config.antenna_angle_offset_deg:.0f}°, "
            f"arc={config.antenna_span_deg:.0f}°, "
            f"phase_delay={'on' if config.use_bmid_phase_delay_radius else 'off'}, "
            f"pick={beamformer_mode}."
        )
        self._log(
            f"Reconstruction completed. Selected beamformer: {selected_name}.{geom_note}"
            f" ROIs={len(evaluated.rois)} (primary bbox={evaluated.primary.bounding_box})."
            f" Refined grid={refinement.grid_shape[1]}x{refinement.grid_shape[0]}.{gt_note}{candidate_note}",
            "SUCCESS",
        )
        self.content_tabs.setCurrentIndex(0)
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
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for suffix, figure in (
            ("selected", self.selected_figure),
            ("beamformers", self.figure),
            ("roi_refine", self.refined_figure),
            ("trace", self.steps_figure),
        ):
            target = out / f"{basename}_{suffix}.png"
            figure.savefig(str(target), dpi=140, bbox_inches="tight")
            paths.append(str(target))
        return paths

    def _apply_dataset_defaults(self, dataset: MicrowaveDataset) -> None:
        assessment = infer_reconstruction_assessment(dataset)
        config = assessment.config
        self.radius_combo.setCurrentText(self._radius_text(config.antenna_radius))
        self.speed_combo.setCurrentText(self._speed_text(config.wave_speed))
        self.span_combo.setCurrentText(self._span_text(config.x_span, config.y_span))
        self.angle_offset_combo.setCurrentText(
            self._angle_offset_text(config.antenna_angle_offset_deg)
        )
        self.rotation_combo.setCurrentText("CW" if config.antenna_clockwise else "CCW")
        self.flip_combo.setCurrentText(
            self._flip_text(config.antenna_flip_x, config.antenna_flip_y)
        )
        self.arc_combo.setCurrentText(
            "355° (BMID)" if abs(config.antenna_span_deg - 355.0) < 1e-3 else "360°"
        )
        self.phase_delay_combo.setCurrentText(
            "On (BMID)" if config.use_bmid_phase_delay_radius else "Off"
        )
        meta = dataset.metadata or {}
        if str(meta.get("dataset_family", "")).upper() == "UM-BMID":
            self.roi_mode_combo.setCurrentText("Peak score")
            # Blind default: pick from the image only. Labels are for scoring after.
            # "Closest to tumor GT" stays in the menu as a lab-only check.
            self.beamformer_mode_combo.setCurrentText("Quality score")
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
        config.antenna_span_deg = (
            355.0 if "355" in self.arc_combo.currentText() else 360.0
        )
        config.use_bmid_phase_delay_radius = (
            self.phase_delay_combo.currentText().startswith("On")
        )
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

    def _update_assumption_summary(
        self, source_notes: dict[str, str], warnings: list[str]
    ) -> None:
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

    def _span_text(
        self, x_span: tuple[float, float], y_span: tuple[float, float]
    ) -> str:
        half_span = max(
            abs(float(x_span[0])),
            abs(float(x_span[1])),
            abs(float(y_span[0])),
            abs(float(y_span[1])),
        )
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
            extent = [
                x_span[0] * 100.0,
                x_span[1] * 100.0,
                y_span[0] * 100.0,
                y_span[1] * 100.0,
            ]
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
        ax.set_title(title, fontsize=10, pad=8)
        if extent is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
        return im

    def _overlay_rois(
        self,
        ax,
        image: np.ndarray,
        rois: list[ROIResult] | ROIResult,
        x_span: tuple[float, float],
        y_span: tuple[float, float],
        tumor_xy_m: tuple[float, float] | None = None,
        linewidth: float = 2.0,
        show_labels: bool = True,
    ) -> None:
        if isinstance(rois, ROIResult):
            rois = [rois]
        colors = ("cyan", "yellow", "magenta", "white")
        x_axis = np.linspace(x_span[0] * 100.0, x_span[1] * 100.0, image.shape[1])
        y_axis = np.linspace(y_span[0] * 100.0, y_span[1] * 100.0, image.shape[0])
        for roi in rois:
            color = colors[(max(int(roi.rank), 1) - 1) % len(colors)]
            x0, y0, x1, y1 = roi.bounding_box
            rect = Rectangle(
                (x_axis[max(0, x0)], y_axis[max(0, y0)]),
                x_axis[min(image.shape[1] - 1, x1 - 1)] - x_axis[max(0, x0)],
                y_axis[min(image.shape[0] - 1, y1 - 1)] - y_axis[max(0, y0)],
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            centroid_x = np.interp(roi.centroid[0], np.arange(image.shape[1]), x_axis)
            centroid_y = np.interp(roi.centroid[1], np.arange(image.shape[0]), y_axis)
            ax.plot(centroid_x, centroid_y, marker="o", color=color, markersize=4)
            if show_labels:
                ax.text(
                    centroid_x,
                    centroid_y,
                    f" #{roi.rank}",
                    color=color,
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )
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

    def _plot_selected_image(
        self,
        image: np.ndarray,
        selected_name: str,
        rois: list[ROIResult] | ROIResult,
        x_span: tuple[float, float],
        y_span: tuple[float, float],
        tumor_xy_m: tuple[float, float] | None = None,
    ) -> None:
        self.selected_figure.clear()
        ax = self.selected_figure.add_subplot(1, 1, 1)
        im = self._configure_image_axes(
            ax,
            image,
            f"Selected Reconstruction: {selected_name}",
            x_span=x_span,
            y_span=y_span,
        )
        self._overlay_rois(ax, image, rois, x_span, y_span, tumor_xy_m=tumor_xy_m)
        if tumor_xy_m is not None:
            ax.legend(loc="upper right", fontsize=8)
        self.selected_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        finish_figure(self.selected_figure)
        self.selected_canvas.draw()

    def _plot_images(
        self,
        images: dict[str, np.ndarray],
        selected_name: str,
        rois: list[ROIResult] | ROIResult,
        x_span: tuple[float, float],
        y_span: tuple[float, float],
        tumor_xy_m: tuple[float, float] | None = None,
    ) -> None:
        self.figure.clear()
        titles = [
            f"DAS  score={self._score_text(quality_name='DAS')}",
            f"DMAS  score={self._score_text(quality_name='DMAS')}",
            f"DMAS-D4  score={self._score_text(quality_name='DMAS-D4')}",
            f"Selected: {selected_name}",
        ]
        keys = ["DAS", "DMAS", "DMAS-D4", selected_name]

        for idx, key in enumerate(keys, start=1):
            ax = self.figure.add_subplot(2, 2, idx)
            self._configure_image_axes(
                ax, images[key], titles[idx - 1], x_span=x_span, y_span=y_span
            )
            self._overlay_rois(
                ax,
                images[key],
                rois,
                x_span,
                y_span,
                tumor_xy_m=tumor_xy_m,
                linewidth=1.5,
                show_labels=True,
            )

        finish_figure(self.figure)
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
        finish_figure(self.refined_figure)
        self.refined_canvas.draw()

    def _plot_reconstruction_trace(
        self,
        s_params,
        freqs,
        time_s,
        time_signals,
        images: dict,
        selected_name: str,
        x_span: tuple[float, float],
        y_span: tuple[float, float],
    ) -> None:
        if s_params is None:
            show_message(self.steps_figure, "Run reconstruction to trace S → image.")
            self.steps_canvas.draw()
            return
        plot_reconstruction_trace(
            self.steps_figure,
            freqs,
            s_params,
            time_s,
            time_signals,
            images,
            selected_name,
            x_span,
            y_span,
            self._configure_image_axes,
            overlay_rois=lambda ax, image: self._overlay_rois(
                ax, image, self.last_roi_results or [], x_span, y_span
            ),
            scores={
                name: float((self.last_snapshot.quality_metrics or {}).get(name, {}).get("score") or 0.0)
                if self.last_snapshot is not None
                else None
                for name in images
            },
        )
        self.steps_canvas.draw()

    def _populate_metrics(
        self, quality_metrics: dict[str, dict], selected_name: str | None = None
    ) -> None:
        self.metrics_table.setRowCount(len(quality_metrics))
        for row, (name, metrics) in enumerate(quality_metrics.items()):
            picked = "YES" if metrics.get("selected") or name == selected_name else ""
            gt = metrics.get("tumor_gt_distance_m")
            gt_txt = f"{float(gt) * 100:.2f}" if isinstance(gt, (int, float)) else "n/a"
            why = str(metrics.get("selection_reason") or "")
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(picked))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{metrics['snr']:.4f}"))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{metrics['scr']:.4f}"))
            self.metrics_table.setItem(
                row, 4, QTableWidgetItem(f"{metrics['contrast']:.4f}")
            )
            self.metrics_table.setItem(
                row, 5, QTableWidgetItem(f"{metrics['score']:.4f}")
            )
            self.metrics_table.setItem(row, 6, QTableWidgetItem(gt_txt))
            self.metrics_table.setItem(row, 7, QTableWidgetItem(why))
        self.metrics_table.resizeColumnsToContents()

    def _score_text(self, quality_name: str) -> str:
        snap = self.last_snapshot
        if snap is None:
            return "n/a"
        metrics = (snap.quality_metrics or {}).get(quality_name) or {}
        score = metrics.get("score")
        if not isinstance(score, (int, float)):
            return "n/a"
        picked = " ★" if quality_name == snap.selected_beamformer else ""
        return f"{score:.3f}{picked}"

    def _populate_summary(
        self,
        images: dict[str, np.ndarray],
        selected_name: str,
        dataset: MicrowaveDataset | None = None,
    ) -> None:
        selected = images[selected_name]
        mean_val = float(np.mean(selected))
        std_val = float(np.std(selected))
        peak_val = float(np.max(selected))
        contrast = float(
            (np.max(selected) - np.min(selected))
            / (np.max(selected) + np.min(selected) + 1e-12)
        )

        info = {
            "Selected Beamformer": selected_name,
            "Mean Intensity": f"{mean_val:.5f}",
            "Std Intensity": f"{std_val:.5f}",
            "Peak Intensity": f"{peak_val:.5f}",
            "Contrast": f"{contrast:.5f}",
        }
        if dataset is not None:
            tax = taxonomy_from_metadata(dataset.metadata)
            info.update({f"Taxonomy {k}": v for k, v in tax.to_display_dict().items()})

        self.summary_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.summary_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.summary_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.summary_table.resizeColumnsToContents()

    def _populate_roi(
        self,
        roi_result: ROIResult,
        candidate_result: dict[str, object] | None = None,
        characterization: dict[str, float] | None = None,
        confidence: dict[str, object] | None = None,
        extra_rois: list[ROIResult] | None = None,
        spot_summaries: list[dict] | None = None,
        localization_roi: ROIResult | None = None,
    ) -> None:
        x0, y0, x1, y1 = roi_result.bounding_box
        n_spots = len(extra_rois) if extra_rois else 1
        loc_rank = localization_roi.rank if localization_roi is not None else roi_result.rank
        cand_threshold = None
        if candidate_result is not None:
            cand_threshold = candidate_result.get("threshold")
        info = {
            "Detected spots": str(n_spots),
            "Localization spot": f"#{loc_rank} (closest to GT if labeled, else highest score)",
            "Bounding Box": f"({x0}, {y0}) - ({x1}, {y1})",
            "Centroid (px)": f"({roi_result.centroid[0]:.1f}, {roi_result.centroid[1]:.1f})",
            "Area (px)": str(roi_result.area),
            "ROI intensity threshold": f"{roi_result.threshold:.2f} (pixels ≥ this fraction of peak after smoothing)",
            "ROI Score": f"{roi_result.score:.5f}",
            "Keep-spot score floor": "0.28 × best ROI score (weaker blobs are dropped)",
        }
        if cand_threshold is not None:
            info["Candidate threshold"] = (
                f"{float(cand_threshold):.2f} (suspicion/confidence must reach this for Yes)"
            )
        if extra_rois:
            for roi in extra_rois:
                info[f"Spot #{roi.rank} score"] = f"{roi.score:.4f} @ px ({roi.centroid[0]:.1f}, {roi.centroid[1]:.1f})"
        if candidate_result:
            is_candidate = bool(candidate_result.get("is_tumor_candidate", False))
            cand_conf = float(candidate_result.get("confidence", 0.0) or 0.0)
            suspicion = float(candidate_result.get("suspicion_score", 0.0) or 0.0)
            info["Tumor Candidate"] = "Yes" if is_candidate else "No"
            info["Candidate Confidence"] = f"{cand_conf:.3f}"
            info["Suspicion Score"] = f"{suspicion:.3f}"
            info["Candidate reason"] = str(candidate_result.get("reason") or "")
        if characterization:
            info["Centroid (cm)"] = (
                f"({characterization.get('centroid_x_cm', 0.0):.2f}, "
                f"{characterization.get('centroid_y_cm', 0.0):.2f})"
            )
            info["Eq. Diameter (cm)"] = f"{characterization.get('equivalent_diameter_cm', 0.0):.2f}"
            info["BBox (cm)"] = (
                f"{characterization.get('bbox_width_cm', 0.0):.2f} x "
                f"{characterization.get('bbox_height_cm', 0.0):.2f}"
            )
            info["Area (cm2)"] = f"{characterization.get('area_cm2', 0.0):.3f}"
            info["FWHM (cm)"] = f"{characterization.get('fwhm_cm', 0.0):.2f}"
            info["Compactness"] = f"{characterization.get('compactness', 0.0):.3f}"
            info["Local SCR"] = f"{characterization.get('local_scr', 0.0):.3f}"
        if confidence:
            overall = float(confidence.get("overall", 0.0) or 0.0)
            label = str(confidence.get("label", "n/a"))
            info["Overall Confidence"] = f"{overall:.3f} ({label})"
            components = confidence.get("components") or {}
            if isinstance(components, dict):
                info["Conf: Detection"] = f"{float(components.get('detection', 0.0) or 0.0):.3f}"
                info["Conf: Localization"] = f"{float(components.get('localization', 0.0) or 0.0):.3f}"

        self.roi_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.roi_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.roi_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.roi_table.resizeColumnsToContents()
        self._populate_spots_table(spot_summaries, extra_rois, localization_roi)

    def _populate_spots_table(
        self,
        spot_summaries: list[dict] | None,
        extra_rois: list[ROIResult] | None,
        localization_roi: ROIResult | None,
    ) -> None:
        rows = list(spot_summaries or [])
        if not rows and extra_rois:
            for roi in extra_rois:
                rows.append(
                    {
                        "rank": roi.rank,
                        "centroid_cm": [None, None],
                        "score": roi.score,
                        "threshold": roi.threshold,
                        "is_tumor_candidate": None,
                        "confidence": None,
                        "gt_distance_cm": None,
                    }
                )
        loc_rank = localization_roi.rank if localization_roi is not None else None
        self.spots_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            cx, cy = (item.get("centroid_cm") or [None, None])[:2]
            cand = item.get("is_tumor_candidate")
            if cand is True:
                cand_txt = "Yes"
            elif cand is False:
                cand_txt = "No"
            else:
                cand_txt = "n/a"
            mark = "LOC" if loc_rank is not None and int(item.get("rank") or 0) == int(loc_rank) else ""
            gt = item.get("gt_distance_cm")
            self.spots_table.setItem(row, 0, QTableWidgetItem(f"{item.get('rank')} {mark}".strip()))
            self.spots_table.setItem(
                row, 1, QTableWidgetItem("n/a" if cx is None else f"{float(cx):.2f}")
            )
            self.spots_table.setItem(
                row, 2, QTableWidgetItem("n/a" if cy is None else f"{float(cy):.2f}")
            )
            self.spots_table.setItem(
                row, 3, QTableWidgetItem(f"{float(item.get('score') or 0.0):.4f}")
            )
            thresh = extra_rois[row].threshold if extra_rois and row < len(extra_rois) else None
            self.spots_table.setItem(
                row,
                4,
                QTableWidgetItem("n/a" if thresh is None else f"{float(thresh):.2f}"),
            )
            conf = item.get("confidence")
            self.spots_table.setItem(
                row, 5, QTableWidgetItem("n/a" if conf is None else f"{float(conf):.3f}")
            )
            self.spots_table.setItem(row, 6, QTableWidgetItem(cand_txt))
            self.spots_table.setItem(
                row,
                7,
                QTableWidgetItem("n/a" if gt is None else f"{float(gt):.2f}"),
            )
        self.spots_table.resizeColumnsToContents()

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

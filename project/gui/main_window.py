"""
gui/main_window.py

Purpose:
    Top-level PySide6 application window. Hosts Module 1's UploadPage and
    Module 2's preprocessing controls (Run Preprocessing button, progress
    bar, status log, signal plots, processing summary) in a tabbed layout.

Input:
    None directly; wires together UploadPage's `dataset_loaded` signal
    with the Module 2 preprocessing pipeline.

Output:
    A QMainWindow ready to be shown by main.py.

Description:
    Module 2's controls are intentionally implemented in this file (rather
    than a separate gui/preprocessing_page.py) to match the project's
    specified GUI file layout (gui/main_window.py, gui/upload_page.py),
    while still keeping all preprocessing *logic* in the preprocessing/
    package — this widget only orchestrates calls into
    preprocessing.preprocessing_pipeline.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from data_loader.dataset_info import MicrowaveDataset
from data_loader.loader import load_dataset
from gui.upload_page import UploadPage
from gui.reconstruction_page import ReconstructionPanel
from preprocessing.preprocessing_pipeline import (
    PreprocessingConfig,
    PreprocessingResult,
    run_preprocessing_pipeline,
)
from utils.exceptions import MicrowaveFrameworkError
from utils.logger import StatusLog, get_logger
from utils.session_report import SessionReportContext, write_session_report

logger = get_logger(__name__)


class PreprocessingWorker(QThread):
    """
    Purpose:
        Run the (potentially slow) preprocessing pipeline on a background
        thread so the GUI stays responsive and the progress bar updates
        smoothly.
    """

    progress_updated = Signal(int, str)
    finished_ok = Signal(object)  # PreprocessingResult
    failed = Signal(str)

    def __init__(self, dataset: MicrowaveDataset, config: PreprocessingConfig) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config

    def run(self) -> None:
        try:
            result = run_preprocessing_pipeline(
                self.dataset,
                self.config,
                progress_callback=lambda pct, msg: self.progress_updated.emit(pct, msg),
            )
            self.finished_ok.emit(result)
        except MicrowaveFrameworkError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected error in preprocessing worker")
            self.failed.emit(f"Unexpected error: {exc}")


class PreprocessingPanel(QWidget):
    """
    Purpose:
        Module 2 GUI panel: pipeline configuration controls, Run button,
        progress bar, status log, comparison/frequency-response plots, and
        the processing summary table.
    """

    preprocessing_completed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset: MicrowaveDataset | None = None
        self.repeated_measurement_paths: list[str] = []
        self.reference_dataset_path: str | None = None
        self.status_log = StatusLog()
        self.worker: PreprocessingWorker | None = None
        self.last_result: PreprocessingResult | None = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("Module 2 — Signal Preprocessing")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        # --- Configuration controls ---
        config_group = QGroupBox("Pipeline Configuration")
        form = QFormLayout()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["savgol", "gaussian", "median", "moving_average", "butterworth", "none"])
        form.addRow("Noise Filtering:", self.filter_combo)

        self.calibration_combo = QComboBox()
        self.calibration_combo.addItems(["self", "none"])
        form.addRow("Calibration:", self.calibration_combo)

        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(["max", "minmax", "zscore", "none"])
        form.addRow("Normalization:", self.normalization_combo)

        self.background_combo = QComboBox()
        self.background_combo.addItems(["enabled", "disabled"])
        form.addRow("Background Subtraction:", self.background_combo)

        self.artifact_combo = QComboBox()
        self.artifact_combo.addItems(["hybrid", "svd", "background", "none"])
        form.addRow("Artifact Suppression:", self.artifact_combo)

        auxiliary_row = QVBoxLayout()

        repeated_row = QHBoxLayout()
        self.repeated_button = QPushButton("Select Repeated .s2p Files")
        self.repeated_button.clicked.connect(self._select_repeated_measurements)
        repeated_row.addWidget(self.repeated_button)
        self.repeated_label = QLabel("No repeated files selected.")
        self.repeated_label.setWordWrap(True)
        repeated_row.addWidget(self.repeated_label, stretch=1)
        auxiliary_row.addLayout(repeated_row)

        reference_row = QHBoxLayout()
        self.reference_button = QPushButton("Select Reference .s2p File")
        self.reference_button.clicked.connect(self._select_reference_dataset)
        reference_row.addWidget(self.reference_button)
        self.reference_label = QLabel("No reference file selected.")
        self.reference_label.setWordWrap(True)
        reference_row.addWidget(self.reference_label, stretch=1)
        auxiliary_row.addLayout(reference_row)

        self.touchstone_hint_label = QLabel(
            "Repeated-measurement averaging and reference subtraction are available for Touchstone .s2p datasets only."
        )
        self.touchstone_hint_label.setWordWrap(True)
        auxiliary_row.addWidget(self.touchstone_hint_label)

        form.addRow("S21 Auxiliary Inputs:", auxiliary_row)

        config_group.setLayout(form)
        layout.addWidget(config_group)

        # --- Run button + progress bar ---
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Preprocessing")
        self.run_button.clicked.connect(self.on_run_clicked)
        self.run_button.setEnabled(False)
        run_row.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        run_row.addWidget(self.progress_bar, stretch=1)
        layout.addLayout(run_row)

        # --- Plots ---
        plots_group = QGroupBox("Signal Plots")
        plots_layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        plots_layout.addWidget(self.canvas)
        plots_group.setLayout(plots_layout)
        layout.addWidget(plots_group, stretch=2)

        # --- Summary + status log ---
        bottom_row = QHBoxLayout()

        summary_group = QGroupBox("Processing Summary")
        summary_layout = QVBoxLayout()
        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        summary_layout.addWidget(self.summary_table)
        summary_group.setLayout(summary_layout)
        bottom_row.addWidget(summary_group, stretch=1)

        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        bottom_row.addWidget(log_group, stretch=1)

        layout.addLayout(bottom_row, stretch=1)

    # ------------------------------------------------------------------ #
    def _log(self, message: str, level: str = "INFO") -> None:
        line = self.status_log.add(message, level)
        self.log_view.appendPlainText(line)

    def set_dataset(self, dataset: MicrowaveDataset) -> None:
        """
        Purpose:
            Receive a newly loaded dataset from Module 1 (via the
            dataset_loaded signal) and enable the Run Preprocessing
            button.
        Input:
            dataset (MicrowaveDataset)
        Output:
            None.
        """
        self.dataset = dataset
        self.run_button.setEnabled(True)
        self._reset_auxiliary_inputs()
        self._update_auxiliary_controls()
        self._log(f"Dataset '{dataset.file_name}' ready for preprocessing.")

    def _is_touchstone_s2p_dataset(self) -> bool:
        return self.dataset is not None and self.dataset.file_path.lower().endswith(".s2p")

    def _reset_auxiliary_inputs(self) -> None:
        self.repeated_measurement_paths = []
        self.reference_dataset_path = None
        self.repeated_label.setText("No repeated files selected.")
        self.reference_label.setText("No reference file selected.")

    def _update_auxiliary_controls(self) -> None:
        enabled = self._is_touchstone_s2p_dataset()
        self.repeated_button.setEnabled(enabled)
        self.reference_button.setEnabled(enabled)
        self.touchstone_hint_label.setVisible(True)
        if enabled:
            self.touchstone_hint_label.setText(
                "Optional .s2p auxiliary inputs: select repeated measurements for complex averaging and a matching reference for complex subtraction."
            )
        else:
            self.touchstone_hint_label.setText(
                "Repeated-measurement averaging and reference subtraction are available for Touchstone .s2p datasets only."
            )

    def _select_repeated_measurements(self) -> None:
        if not self._is_touchstone_s2p_dataset():
            return

        start_dir = self.dataset.file_path.rsplit("\\", 1)[0] if self.dataset is not None and "\\" in self.dataset.file_path else ""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Repeated Touchstone Measurements",
            start_dir,
            "Touchstone S2P Files (*.s2p)",
        )
        if not file_paths:
            return

        current_path = self.dataset.file_path if self.dataset is not None else None
        self.repeated_measurement_paths = [path for path in file_paths if path != current_path]
        self.repeated_label.setText(f"{len(self.repeated_measurement_paths)} repeated file(s) selected.")
        self._log(f"Selected {len(self.repeated_measurement_paths)} repeated .s2p file(s) for averaging.")

    def _select_reference_dataset(self) -> None:
        if not self._is_touchstone_s2p_dataset():
            return

        start_dir = self.dataset.file_path.rsplit("\\", 1)[0] if self.dataset is not None and "\\" in self.dataset.file_path else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Touchstone Measurement",
            start_dir,
            "Touchstone S2P Files (*.s2p)",
        )
        if not file_path:
            return

        if self.dataset is not None and file_path == self.dataset.file_path:
            QMessageBox.warning(self, "Invalid Reference", "Select a different .s2p file as the reference dataset.")
            return

        self.reference_dataset_path = file_path
        self.reference_label.setText(file_path.split("\\")[-1])
        self._log(f"Selected reference .s2p file: {file_path}")

    def _load_auxiliary_datasets(self) -> tuple[list[MicrowaveDataset] | None, MicrowaveDataset | None]:
        if not self._is_touchstone_s2p_dataset():
            return None, None

        repeated_datasets: list[MicrowaveDataset] = []
        for path in self.repeated_measurement_paths:
            repeated_datasets.append(load_dataset(path))

        reference_dataset = load_dataset(self.reference_dataset_path) if self.reference_dataset_path else None
        return repeated_datasets or None, reference_dataset

    def on_run_clicked(self) -> None:
        """
        Purpose:
            Build a PreprocessingConfig from the current GUI controls and
            launch the pipeline on a background thread.
        Input:
            None (triggered by Qt signal).
        Output:
            None. Side effects: disables Run button, starts worker thread.
        """
        if self.dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset in Module 1 first.")
            return

        try:
            repeated_measurements, reference_dataset = self._load_auxiliary_datasets()
        except MicrowaveFrameworkError as exc:
            self._log(f"Failed to load auxiliary Touchstone dataset: {exc}", "ERROR")
            QMessageBox.critical(self, "Auxiliary Dataset Error", str(exc))
            return
        except FileNotFoundError as exc:
            self._log(str(exc), "ERROR")
            QMessageBox.critical(self, "Auxiliary Dataset Error", str(exc))
            return

        config = PreprocessingConfig(
            filter_method=self.filter_combo.currentText(),
            calibration_method=self.calibration_combo.currentText(),
            normalization_method=self.normalization_combo.currentText(),
            do_background_subtraction=(self.background_combo.currentText() == "enabled"),
            artifact_method=self.artifact_combo.currentText(),
            repeated_measurements=repeated_measurements,
            reference_dataset=reference_dataset,
        )

        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self._log("Starting preprocessing pipeline...")

        self.worker = PreprocessingWorker(self.dataset, config)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self._log(message)

    def _on_failed(self, message: str) -> None:
        self._log(f"Preprocessing failed: {message}", "ERROR")
        QMessageBox.critical(self, "Preprocessing Error", message)
        self.run_button.setEnabled(True)

    def _on_finished(self, result: PreprocessingResult) -> None:
        self.last_result = result
        self._log("Preprocessing completed successfully.", "SUCCESS")
        if result.validation_report is not None:
            if result.validation_report.averaging_performed:
                self._log("Complex averaging was applied using the selected repeated .s2p measurements.")
            if result.validation_report.reference_subtraction_performed:
                self._log("Complex reference subtraction was applied using the selected reference .s2p measurement.")
        self._populate_summary(result)
        self._plot_result(result)
        self.run_button.setEnabled(True)
        self.preprocessing_completed.emit(result)

    def _populate_summary(self, result: PreprocessingResult) -> None:
        info = result.quality_report.to_display_dict()
        if result.validation_report is not None:
            info.update(result.validation_report.to_display_dict())
        self.summary_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.summary_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.summary_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.summary_table.resizeColumnsToContents()

    def _plot_result(self, result: PreprocessingResult) -> None:
        if result.touchstone_s21_result is not None:
            self._plot_touchstone_result(result)
            return

        self.figure.clear()

        freqs_ghz = result.original_dataset.frequencies / 1e9
        original = result.original_dataset.s_parameters
        processed = result.processed_dataset.s_parameters

        orig_flat = original.reshape(original.shape[0], -1)
        proc_flat = processed.reshape(processed.shape[0], -1)
        trace_idx = 0

        ax1 = self.figure.add_subplot(2, 2, 1)
        ax1.plot(freqs_ghz, 20 * np.log10(np.abs(orig_flat[:, trace_idx]) + 1e-12))
        ax1.set_title("Original Signal")
        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(2, 2, 2)
        ax2.plot(freqs_ghz, 20 * np.log10(np.abs(proc_flat[:, trace_idx]) + 1e-12), color="darkorange")
        ax2.set_title("Processed Signal")
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.grid(True, alpha=0.3)

        ax3 = self.figure.add_subplot(2, 2, 3)
        ax3.plot(freqs_ghz, np.abs(orig_flat[:, trace_idx]), label="Original", alpha=0.7)
        ax3.plot(freqs_ghz, np.abs(proc_flat[:, trace_idx]), label="Processed", alpha=0.7)
        ax3.set_title("Comparison Plot")
        ax3.set_xlabel("Frequency (GHz)")
        ax3.set_ylabel("Magnitude (linear)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        ax4 = self.figure.add_subplot(2, 2, 4)
        mean_mag_before = np.mean(np.abs(orig_flat), axis=1)
        mean_mag_after = np.mean(np.abs(proc_flat), axis=1)
        ax4.plot(freqs_ghz, mean_mag_before, label="Before", alpha=0.7)
        ax4.plot(freqs_ghz, mean_mag_after, label="After", alpha=0.7)
        ax4.set_title("Frequency Response (Mean over Traces)")
        ax4.set_xlabel("Frequency (GHz)")
        ax4.set_ylabel("Magnitude (linear)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_touchstone_result(self, result: PreprocessingResult) -> None:
        self.figure.clear()

        touchstone = result.touchstone_s21_result
        raw_freqs_ghz = touchstone.raw_frequencies_hz / 1e9
        uniform_freqs_ghz = touchstone.uniform_frequencies_hz / 1e9

        ax1 = self.figure.add_subplot(3, 2, 1)
        ax1.plot(raw_freqs_ghz, touchstone.raw_magnitude_db, color="tab:blue")
        ax1.set_title("Raw S21 Magnitude")
        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(3, 2, 2)
        ax2.plot(raw_freqs_ghz, touchstone.wrapped_phase_deg, label="Wrapped", alpha=0.8)
        ax2.plot(raw_freqs_ghz, touchstone.unwrapped_phase_deg, label="Unwrapped", alpha=0.8)
        ax2.set_title("S21 Phase")
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("Phase (deg)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = self.figure.add_subplot(3, 2, 3)
        ax3.plot(raw_freqs_ghz, touchstone.raw_s21.real, label="Raw Real", alpha=0.8)
        ax3.plot(uniform_freqs_ghz, touchstone.filtered_s21.real, label="Filtered Real", alpha=0.8)
        ax3.set_title("Real Part")
        ax3.set_xlabel("Frequency (GHz)")
        ax3.set_ylabel("Real(S21)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        ax4 = self.figure.add_subplot(3, 2, 4)
        ax4.plot(raw_freqs_ghz, touchstone.raw_s21.imag, label="Raw Imag", alpha=0.8)
        ax4.plot(uniform_freqs_ghz, touchstone.filtered_s21.imag, label="Filtered Imag", alpha=0.8)
        ax4.set_title("Imaginary Part")
        ax4.set_xlabel("Frequency (GHz)")
        ax4.set_ylabel("Imag(S21)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        ax5 = self.figure.add_subplot(3, 2, 5)
        ax5.plot(uniform_freqs_ghz, 20 * np.log10(np.abs(touchstone.corrected_s21) + 1e-12), label="Corrected", alpha=0.75)
        ax5.plot(uniform_freqs_ghz, 20 * np.log10(np.abs(touchstone.filtered_s21) + 1e-12), label="Filtered", alpha=0.75)
        ax5.plot(uniform_freqs_ghz, 20 * np.log10(np.abs(touchstone.windowed_s21) + 1e-12), label="Hamming", alpha=0.75)
        ax5.set_title("Processed Variants")
        ax5.set_xlabel("Frequency (GHz)")
        ax5.set_ylabel("Magnitude (dB)")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        ax6 = self.figure.add_subplot(3, 2, 6)
        ax6.plot(raw_freqs_ghz, 20 * np.log10(np.abs(touchstone.raw_s21) + 1e-12), label="Raw", alpha=0.75)
        ax6.plot(uniform_freqs_ghz, 20 * np.log10(np.abs(touchstone.filtered_s21) + 1e-12), label="Processed", alpha=0.75)
        ax6.set_title("Raw vs Processed")
        ax6.set_xlabel("Frequency (GHz)")
        ax6.set_ylabel("Magnitude (dB)")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()


class MainWindow(QMainWindow):
    """
    Purpose:
        Application entry-point window combining Module 1 (UploadPage)
        and Module 2 (PreprocessingPanel) as tabs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Microwave Imaging Framework — Data Acquisition, Preprocessing & Reconstruction")
        self.resize(1150, 950)

        self.upload_page = UploadPage()
        self.preprocessing_panel = PreprocessingPanel()
        self.reconstruction_panel = ReconstructionPanel()
        self._session_dataset: MicrowaveDataset | None = None

        tabs = QTabWidget()
        tabs.addTab(self.upload_page, "1. Data Acquisition")
        tabs.addTab(self.preprocessing_panel, "2. Signal Preprocessing")
        tabs.addTab(self.reconstruction_panel, "3. Reconstruction")
        self.setCentralWidget(tabs)

        toolbar = QToolBar("Session")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.report_action = toolbar.addAction("Download Final Report")
        self.report_action.setToolTip(
            "Copy the latest auto-saved report (and figures) to a folder you choose"
        )
        self.report_action.triggered.connect(self._download_final_report)
        self.auto_report_label = QLabel("Auto-report: waiting for pipeline…")
        self.auto_report_label.setStyleSheet("color: #555; padding-left: 12px;")
        toolbar.addWidget(self.auto_report_label)

        self.upload_page.dataset_loaded.connect(self._on_dataset_loaded)
        self.preprocessing_panel.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.reconstruction_panel.export_report_button.clicked.connect(self._download_final_report)
        self.reconstruction_panel.reconstruction_completed.connect(self._on_reconstruction_completed)
        self._tabs = tabs
        self._results_dir = os.path.join(os.getcwd(), "results")
        self._latest_report_paths: dict[str, str] = {}

    def _on_dataset_loaded(self, dataset: MicrowaveDataset) -> None:
        self._session_dataset = dataset
        self.preprocessing_panel.set_dataset(dataset)
        self.reconstruction_panel.set_dataset(dataset)
        self.reconstruction_panel.last_snapshot = None
        self.reconstruction_panel.export_report_button.setEnabled(False)
        self._tabs.setCurrentWidget(self.preprocessing_panel)
        self.auto_report_label.setText("Auto-report: dataset loaded — run preprocess/reconstruct")

    def _on_preprocessing_completed(self, result: PreprocessingResult) -> None:
        self.reconstruction_panel.set_processed_dataset(result.processed_dataset)
        self._tabs.setCurrentWidget(self.reconstruction_panel)
        self._autosave_session_report(reason="preprocessing")

    def _on_reconstruction_completed(self, _snapshot) -> None:
        self.reconstruction_panel.export_report_button.setEnabled(True)
        self._autosave_session_report(reason="reconstruction")

    def _build_report_context(self) -> SessionReportContext | None:
        dataset = self._session_dataset or self.reconstruction_panel.dataset
        preprocessing = self.preprocessing_panel.last_result
        reconstruction = self.reconstruction_panel.last_snapshot
        if dataset is None and preprocessing is None and reconstruction is None:
            return None
        return SessionReportContext(
            dataset=dataset,
            preprocessing=preprocessing,
            reconstruction=reconstruction,
        )

    def _autosave_session_report(self, reason: str) -> None:
        """Rewrite results/latest_session_report.* plus a timestamped archive copy."""
        ctx = self._build_report_context()
        if ctx is None:
            return

        try:
            os.makedirs(self._results_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_stem = f"session_report_{stamp}"
            figure_paths: list[str] = []
            if ctx.reconstruction is not None:
                figure_paths = self.reconstruction_panel.save_figures(
                    self._results_dir, "latest_session_report"
                )
                # Also keep timestamped figure copies next to the archive report.
                self.reconstruction_panel.save_figures(self._results_dir, archive_stem)

            latest_paths = write_session_report(
                ctx,
                output_dir=self._results_dir,
                basename="latest_session_report",
                figure_paths=figure_paths,
            )
            archive_paths = write_session_report(
                ctx,
                output_dir=self._results_dir,
                basename=archive_stem,
                figure_paths=[
                    os.path.join(self._results_dir, f"{archive_stem}_selected.png"),
                    os.path.join(self._results_dir, f"{archive_stem}_beamformers.png"),
                    os.path.join(self._results_dir, f"{archive_stem}_roi_refine.png"),
                ]
                if ctx.reconstruction is not None
                else None,
            )
            self._latest_report_paths = latest_paths
            self.auto_report_label.setText(
                f"Auto-report updated ({reason}): results/latest_session_report.md"
            )
            logger.info(
                "Auto-saved session report (%s) to %s and %s",
                reason,
                latest_paths["markdown"],
                archive_paths["markdown"],
            )
        except Exception:
            logger.exception("Auto-save of session report failed")
            self.auto_report_label.setText("Auto-report: save failed (see log)")

    def _download_final_report(self) -> None:
        ctx = self._build_report_context()
        if ctx is None:
            QMessageBox.information(
                self,
                "Nothing to Export",
                "Load a dataset (and optionally run preprocessing / reconstruction) before downloading a report.",
            )
            return

        # Ensure latest files exist / are fresh before offering a copy.
        self._autosave_session_report(reason="manual-export")

        default_dir = self._results_dir
        os.makedirs(default_dir, exist_ok=True)
        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose folder to copy the final report",
            default_dir,
        )
        if not target_dir:
            return

        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_stem = f"session_report_{stamp}"
            figure_paths: list[str] = []
            if ctx.reconstruction is not None:
                figure_paths = self.reconstruction_panel.save_figures(target_dir, file_stem)
            paths = write_session_report(
                ctx,
                output_dir=target_dir,
                basename=file_stem,
                figure_paths=figure_paths,
            )
            message = (
                f"Report copied:\n\n"
                f"Markdown: {paths['markdown']}\n"
                f"JSON: {paths['json']}\n\n"
                f"Also always auto-updated at:\n"
                f"{os.path.join(self._results_dir, 'latest_session_report.md')}"
            )
            if figure_paths:
                message += "\nFigures:\n- " + "\n- ".join(figure_paths)
            QMessageBox.information(self, "Final Report Saved", message)
            logger.info("Session report written to %s", paths["markdown"])
        except Exception as exc:
            logger.exception("Failed to write session report")
            QMessageBox.critical(self, "Report Export Failed", str(exc))

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

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
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
    QVBoxLayout,
    QWidget,
)

from data_loader.dataset_info import MicrowaveDataset
from gui.upload_page import UploadPage
from preprocessing.preprocessing_pipeline import (
    PreprocessingConfig,
    PreprocessingResult,
    run_preprocessing_pipeline,
)
from utils.exceptions import MicrowaveFrameworkError
from utils.logger import StatusLog, get_logger

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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset: MicrowaveDataset | None = None
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
        self.filter_combo.addItems(["savgol", "moving_average", "butterworth", "none"])
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
        self._log(f"Dataset '{dataset.file_name}' ready for preprocessing.")

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

        config = PreprocessingConfig(
            filter_method=self.filter_combo.currentText(),
            calibration_method=self.calibration_combo.currentText(),
            normalization_method=self.normalization_combo.currentText(),
            do_background_subtraction=(self.background_combo.currentText() == "enabled"),
            artifact_method=self.artifact_combo.currentText(),
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
        self._populate_summary(result)
        self._plot_result(result)
        self.run_button.setEnabled(True)

    def _populate_summary(self, result: PreprocessingResult) -> None:
        info = result.quality_report.to_display_dict()
        self.summary_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.summary_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.summary_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.summary_table.resizeColumnsToContents()

    def _plot_result(self, result: PreprocessingResult) -> None:
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


class MainWindow(QMainWindow):
    """
    Purpose:
        Application entry-point window combining Module 1 (UploadPage)
        and Module 2 (PreprocessingPanel) as tabs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Microwave Imaging Framework — Data Acquisition & Preprocessing")
        self.resize(1150, 850)

        self.upload_page = UploadPage()
        self.preprocessing_panel = PreprocessingPanel()

        tabs = QTabWidget()
        tabs.addTab(self.upload_page, "1. Data Acquisition")
        tabs.addTab(self.preprocessing_panel, "2. Signal Preprocessing")
        self.setCentralWidget(tabs)

        self.upload_page.dataset_loaded.connect(self._on_dataset_loaded)
        self._tabs = tabs

    def _on_dataset_loaded(self, dataset: MicrowaveDataset) -> None:
        self.preprocessing_panel.set_dataset(dataset)
        self._tabs.setCurrentWidget(self.preprocessing_panel)

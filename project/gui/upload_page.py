"""
gui/upload_page.py

Purpose:
    PySide6 widget implementing Module 1 (Data Acquisition) of the GUI:
    a "Load Dataset" file browser button, a dataset information panel, and
    a status log, with graceful error handling for invalid/corrupted
    files.

Input:
    User interaction (button clicks, file dialog selection).

Output:
    Emits `dataset_loaded(MicrowaveDataset)` Qt signal when a dataset is
    successfully loaded, so gui/main_window.py can forward it to the
    Module 2 preprocessing page.

Description:
    This widget owns no preprocessing logic; it is purely responsible for
    Module 1 concerns (loading, validating, and displaying dataset info).
    UM-BMID fd_data cubes open a scan-selection dialog before loading.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from automation.service import AutoValidationService
from data_loader.bmid_loader import (
    BmidScanInfo,
    ScanSelectionRequiredError,
    is_bmid_fd_filename,
    is_likely_metadata_file,
    list_bmid_scans,
    resolve_measurement_from_metadata_path,
)
from data_loader.dataset_info import MicrowaveDataset
from data_loader.loader import load_dataset_with_summary
from gui.styles import set_page_title, set_primary_button
from utils.exceptions import MicrowaveFrameworkError
from utils.logger import StatusLog, get_logger

logger = get_logger(__name__)

SUPPORTED_FILE_FILTER = (
    "Microwave Datasets (*.mat *.s1p *.s2p *.s4p *.s8p);;"
    "MATLAB Files (*.mat);;"
    "Touchstone Files (*.s1p *.s2p *.s4p *.s8p);;"
    "All Files (*)"
)


class AutoValidationWorker(QThread):
    """Background runner for one-click auto validation (no UI freeze)."""

    progress_updated = Signal(int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        roots: list[str],
        *,
        max_files: int = 8,
        threshold_profile: str = "balanced",
        include_reconstruction_checks: bool = True,
        output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.roots = roots
        self.max_files = max_files
        self.threshold_profile = threshold_profile
        self.include_reconstruction_checks = include_reconstruction_checks
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            service = AutoValidationService(output_dir=self.output_dir)
            batch = service.run(
                self.roots,
                max_files=self.max_files,
                threshold_profile=self.threshold_profile,
                include_reconstruction_checks=self.include_reconstruction_checks,
                progress=lambda pct, msg: self.progress_updated.emit(pct, msg),
            )
            self.finished_ok.emit(batch)
        except Exception as exc:  # pragma: no cover - defensive GUI path
            logger.exception("Auto validation worker failed")
            self.failed.emit(str(exc))


class BmidScanPickerDialog(QDialog):
    """Choose one scan from a UM-BMID multi-scan frequency-domain cube."""

    def __init__(
        self, scans: list[BmidScanInfo], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select UM-BMID Scan")
        self.resize(640, 420)
        self._scans = scans
        self.selected_index: int | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"{len(scans)} scans found. Pick one tumor or healthy case to load "
                "(adipose-reference clean S21)."
            )
        )

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Tumor only", "Healthy only"])
        self.filter_combo.currentTextChanged.connect(self._repopulate)
        filter_row.addWidget(self.filter_combo, stretch=1)
        layout.addLayout(filter_row)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.list_widget, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._repopulate()

    def _repopulate(self) -> None:
        mode = self.filter_combo.currentText()
        self.list_widget.clear()
        for scan in self._scans:
            if mode == "Tumor only" and not scan.has_tumor:
                continue
            if mode == "Healthy only" and scan.has_tumor:
                continue
            self.list_widget.addItem(scan.label)
            item = self.list_widget.item(self.list_widget.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, scan.index)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    def accept(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.warning(self, "No Scan Selected", "Please select a scan.")
            return
        self.selected_index = int(item.data(Qt.ItemDataRole.UserRole))
        super().accept()


class UploadPage(QWidget):
    """
    Purpose:
        Module 1 GUI page: file browser, dataset info panel, status log.
    """

    dataset_loaded = Signal(object)  # emits MicrowaveDataset

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.status_log = StatusLog()
        self.current_dataset: MicrowaveDataset | None = None
        self._auto_worker: AutoValidationWorker | None = None
        self._results_dir = str(Path(__file__).resolve().parent.parent / "results")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel()
        set_page_title(title, "Data Acquisition")
        title_col.addWidget(title)
        subtitle = QLabel(
            "Load MATLAB (.mat) or Touchstone (.sNp). BMID cubes open a scan picker. "
            "Use Run Auto Validation for batch quality checks + dual reports."
        )
        subtitle.setObjectName("hintLabel")
        subtitle.setWordWrap(True)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, stretch=1)

        self.load_button = QPushButton("Load dataset")
        set_primary_button(self.load_button)
        self.load_button.clicked.connect(self.on_load_dataset_clicked)
        header.addWidget(self.load_button)

        self.auto_validate_button = QPushButton("Run Auto Validation")
        self.auto_validate_button.setToolTip(
            "Discover datasets, run load/preprocess/reconstruction checks, "
            "classify pass/warn/fail, and write guide + beginner reports."
        )
        self.auto_validate_button.clicked.connect(self.on_run_auto_validation_clicked)
        header.addWidget(self.auto_validate_button)
        layout.addLayout(header)

        self.file_label = QLabel("No file loaded.")
        self.file_label.setObjectName("hintLabel")
        layout.addWidget(self.file_label)

        body = QHBoxLayout()
        body.setSpacing(20)

        info_col = QVBoxLayout()
        info_col.addWidget(QLabel("Dataset"))
        self.info_table = QTableWidget(0, 2)
        self.info_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.info_table.setAlternatingRowColors(True)
        info_col.addWidget(self.info_table)
        body.addLayout(info_col, stretch=1)

        log_col = QVBoxLayout()
        log_col.addWidget(QLabel("Log"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        log_col.addWidget(self.log_view)
        body.addLayout(log_col, stretch=1)

        layout.addLayout(body, stretch=1)

    def _log(self, message: str, level: str = "INFO") -> None:
        line = self.status_log.add(message, level)
        self.log_view.appendPlainText(line)

    def on_load_dataset_clicked(self) -> None:
        start_dir = os.path.join(os.getcwd(), "datasets")
        if not os.path.isdir(start_dir):
            start_dir = os.getcwd()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Microwave Dataset", start_dir, SUPPORTED_FILE_FILTER
        )
        if not file_path:
            return

        self.load_dataset_from_path(file_path)

    def load_dataset_from_path(
        self, file_path: str, scan_index: int | None = None
    ) -> None:
        if scan_index is None and is_likely_metadata_file(file_path):
            resolved = resolve_measurement_from_metadata_path(file_path)
            if resolved is not None and os.path.abspath(resolved) != os.path.abspath(file_path):
                self._log(
                    "Selected metadata-only file; automatically loading paired measurement: "
                    f"{resolved}"
                )
                file_path = resolved
            else:
                self._log(
                    "Selected file appears metadata-only and no paired fd_data file was found automatically.",
                    "WARN",
                )

        self._log(f"Loading file: {file_path}")
        try:
            if is_bmid_fd_filename(file_path) and scan_index is None:
                scans = list_bmid_scans(file_path)
                dialog = BmidScanPickerDialog(scans, self)
                if dialog.exec() != QDialog.Accepted or dialog.selected_index is None:
                    self._log("BMID scan selection cancelled.")
                    return
                scan_index = dialog.selected_index
                self._log(
                    f"Selected BMID scan index {scan_index}: {scans[scan_index].label}"
                )

            dataset, summary = load_dataset_with_summary(
                file_path, scan_index=scan_index
            )
        except ScanSelectionRequiredError as exc:
            # Some UM-BMID cubes use generic names (e.g. fd_data_gen_*),
            # so prompt here when the loader detects multi-scan content.
            try:
                scans = list_bmid_scans(file_path)
            except Exception:
                self._log(f"Failed to load dataset: {exc}", "ERROR")
                QMessageBox.critical(self, "Dataset Load Error", str(exc))
                return

            dialog = BmidScanPickerDialog(scans, self)
            if dialog.exec() != QDialog.Accepted or dialog.selected_index is None:
                self._log("BMID scan selection cancelled.")
                return

            chosen = dialog.selected_index
            self._log(f"Selected BMID scan index {chosen}: {scans[chosen].label}")
            self.load_dataset_from_path(file_path, scan_index=chosen)
            return
        except MicrowaveFrameworkError as exc:
            self._log(f"Failed to load dataset: {exc}", "ERROR")
            QMessageBox.critical(self, "Dataset Load Error", str(exc))
            return
        except FileNotFoundError as exc:
            self._log(str(exc), "ERROR")
            QMessageBox.critical(self, "File Not Found", str(exc))
            return
        except Exception as exc:  # defensive catch-all, never crash the GUI
            logger.exception("Unexpected error while loading dataset")
            self._log(f"Unexpected error: {exc}", "ERROR")
            QMessageBox.critical(self, "Unexpected Error", str(exc))
            return

        self.current_dataset = dataset
        self.file_label.setText(f"Loaded: {summary.file_name}")
        display = summary.to_display_dict()
        meta = dataset.metadata or {}
        if meta.get("dataset_family") == "UM-BMID":
            display["BMID Scan"] = str(meta.get("bmid_scan_index"))
            display["Phantom"] = str(meta.get("bmid_phant_id", "N/A"))
            display["Tumor"] = (
                f"{meta['tumor_diameter_m'] * 100:.1f} cm @ "
                f"({meta['tumor_x_m'] * 100:.2f}, {meta['tumor_y_m'] * 100:.2f}) cm"
                if meta.get("bmid_has_tumor")
                else "Healthy (no tumor)"
            )
            display["Antenna Radius"] = (
                f"{meta.get('antenna_radius_m', 0) * 100:.0f} cm"
            )
        self._populate_info_table(display)
        self._log(f"Dataset '{summary.file_name}' loaded successfully.", "SUCCESS")
        self.dataset_loaded.emit(dataset)

    def _populate_info_table(self, info: dict) -> None:
        self.info_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.info_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.info_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.info_table.resizeColumnsToContents()

    def on_run_auto_validation_clicked(self) -> None:
        if self._auto_worker is not None and self._auto_worker.isRunning():
            QMessageBox.information(
                self,
                "Auto Validation Running",
                "Validation is already in progress. Watch the log panel for updates.",
            )
            return

        datasets_dir = Path(__file__).resolve().parent.parent / "datasets"
        roots = [str(datasets_dir)] if datasets_dir.is_dir() else [os.getcwd()]
        self._log(
            f"Starting auto validation under: {roots[0]} (profile=balanced)",
            "INFO",
        )
        self.auto_validate_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self._auto_worker = AutoValidationWorker(
            roots,
            max_files=8,
            threshold_profile="balanced",
            include_reconstruction_checks=True,
            output_dir=self._results_dir,
        )
        self._auto_worker.progress_updated.connect(self._on_auto_progress)
        self._auto_worker.finished_ok.connect(self._on_auto_finished)
        self._auto_worker.failed.connect(self._on_auto_failed)
        self._auto_worker.start()

    def _on_auto_progress(self, percent: int, message: str) -> None:
        self._log(f"[Auto {percent}%] {message}")

    def _on_auto_failed(self, message: str) -> None:
        self.auto_validate_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self._log(f"Auto validation failed: {message}", "ERROR")
        QMessageBox.critical(
            self,
            "Auto Validation Failed",
            f"{message}\n\nExisting load/preprocess/reconstruct workflows are unchanged.",
        )

    def _on_auto_finished(self, batch) -> None:
        self.auto_validate_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self._log(batch.message, "SUCCESS")
        guide = batch.report_paths.get("guide")
        beginner = batch.report_paths.get("beginner")
        recommended = [r for r in batch.results if r.recommended_upload]
        summary = (
            f"{batch.message}\n\n"
            f"Recommended uploads: {len(recommended)}\n"
            f"Guide report:\n{guide}\n\n"
            f"Beginner report:\n{beginner}"
        )
        QMessageBox.information(self, "Auto Validation Complete", summary)
        if guide and os.path.isfile(guide):
            QDesktopServices.openUrl(QUrl.fromLocalFile(guide))
        elif beginner and os.path.isfile(beginner):
            QDesktopServices.openUrl(QUrl.fromLocalFile(beginner))


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
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from automation.reports import pick_showcase_result
from automation.service import AutoValidationService, default_validation_roots, umbmid_root
from gui.plotting import make_gui_figure, plot_complex_matrix, show_message
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
from quality.tumor_taxonomy import taxonomy_from_metadata, taxonomy_from_scan
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
        max_files: int = 1,
        threshold_profile: str = "balanced",
        include_reconstruction_checks: bool = True,
        output_dir: str | None = None,
        bmid_strategy: str = "diverse",
        max_bmid_scans: int = 8,
    ) -> None:
        super().__init__()
        self.roots = roots
        self.max_files = max_files
        self.threshold_profile = threshold_profile
        self.include_reconstruction_checks = include_reconstruction_checks
        self.output_dir = output_dir
        self.bmid_strategy = bmid_strategy
        self.max_bmid_scans = max_bmid_scans

    def run(self) -> None:
        try:
            service = AutoValidationService(output_dir=self.output_dir)
            batch = service.run(
                self.roots,
                max_files=self.max_files,
                threshold_profile=self.threshold_profile,
                include_reconstruction_checks=self.include_reconstruction_checks,
                bmid_strategy=self.bmid_strategy,
                max_bmid_scans=self.max_bmid_scans,
                progress=lambda pct, msg: self.progress_updated.emit(pct, msg),
                should_stop=self.isInterruptionRequested,
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
        self.filter_combo.addItems(
            [
                "All",
                "Tumor only",
                "Healthy only",
                "Small tumor",
                "Medium tumor",
                "Large tumor",
            ]
        )
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
            tax = taxonomy_from_scan(scan)
            if mode == "Tumor only" and not scan.has_tumor:
                continue
            if mode == "Healthy only" and scan.has_tumor:
                continue
            if mode == "Small tumor" and tax.size_class != "small":
                continue
            if mode == "Medium tumor" and tax.size_class != "medium":
                continue
            if mode == "Large tumor" and tax.size_class != "large":
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
    auto_validation_finished = Signal(object)  # emits ValidationBatchResult

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
            "Load Dataset opens umbmid/simple-clean so you can pick gen 1 "
            "(matlab-data), gen 2 (matlab-data2), or gen 3 (matlab-data3), "
            "then an fd_data_gen_*.mat cube — not metadata under project/datasets. "
            "Auto Validation defaults to a small diverse subset (healthy + "
            "small/medium/large tumors). Use All scans only when you want the full 2000+ run."
        )
        subtitle.setObjectName("hintLabel")
        subtitle.setWordWrap(True)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, stretch=1)
        layout.addLayout(header)

        actions = QHBoxLayout()
        self.load_button = QPushButton("Load dataset")
        set_primary_button(self.load_button)
        self.load_button.clicked.connect(self.on_load_dataset_clicked)
        actions.addWidget(self.load_button)

        self.auto_validate_button = QPushButton("Run Auto Validation")
        self.auto_validate_button.setToolTip(
            "Reconstruct a small diverse UM-BMID subset (not all 2264 scans), "
            "write an analysis report, and load the best tumor case into the GUI."
        )
        self.auto_validate_button.clicked.connect(self.on_run_auto_validation_clicked)
        actions.addWidget(self.auto_validate_button)

        self.auto_scope_combo = QComboBox()
        self.auto_scope_combo.addItem("Periodic test (8 diverse scans)", "diverse")
        self.auto_scope_combo.addItem("Tumor + healthy pair", "tumor_and_healthy")
        self.auto_scope_combo.addItem("All scans (slow, 2000+)", "all")
        self.auto_scope_combo.setToolTip(
            "Periodic test is the default. All scans reconstructs every labeled case and takes a long time."
        )
        actions.addWidget(self.auto_scope_combo)

        self.cancel_auto_button = QPushButton("Cancel Auto Validation")
        self.cancel_auto_button.setEnabled(False)
        self.cancel_auto_button.clicked.connect(self.on_cancel_auto_validation_clicked)
        actions.addWidget(self.cancel_auto_button)
        actions.addStretch(1)
        layout.addLayout(actions)

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
        self.info_table.setMaximumHeight(160)
        info_col.addWidget(self.info_table)

        info_col.addWidget(QLabel("Raw traces (after load)"))
        self.raw_figure = make_gui_figure(8.6, 6.4)
        self.raw_canvas = FigureCanvasQTAgg(self.raw_figure)
        self.raw_canvas.setMinimumHeight(420)
        self.raw_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        info_col.addWidget(self.raw_canvas, stretch=3)
        show_message(self.raw_figure, "Load a dataset to see Real / Imag / |S| of the S matrix.")
        self.raw_canvas.draw()
        body.addLayout(info_col, stretch=3)

        log_col = QVBoxLayout()
        log_col.addWidget(QLabel("Log"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setMinimumWidth(260)
        log_col.addWidget(self.log_view)
        body.addLayout(log_col, stretch=1)

        layout.addLayout(body, stretch=1)

    def _log(self, message: str, level: str = "INFO") -> None:
        line = self.status_log.add(message, level)
        self.log_view.appendPlainText(line)

    def _default_load_dir(self) -> str:
        """Open at umbmid/simple-clean so gen 1/2/3 folders can be chosen."""
        umbmid = umbmid_root()
        if umbmid is not None:
            return str(umbmid)
        fallback = Path(__file__).resolve().parent.parent / "datasets"
        return str(fallback if fallback.is_dir() else Path.cwd())

    def on_load_dataset_clicked(self) -> None:
        start_dir = self._default_load_dir()
        dialog = QFileDialog(
            self,
            "Load UM-BMID cube — choose gen 1 / 2 / 3, then fd_data_gen_*.mat",
            start_dir,
            SUPPORTED_FILE_FILTER,
        )
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setDirectory(start_dir)
        dialog.setSidebarUrls([QUrl.fromLocalFile(start_dir)])
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.selectedFiles()
        if not chosen:
            return
        self.load_dataset_from_path(chosen[0])

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
                umbmid = umbmid_root()
                hint = (
                    "Selected file is metadata-only (no S-parameter measurements). "
                    "Open an fd_data_*.mat cube under umbmid/simple-clean: "
                    "matlab-data (gen 1), matlab-data2 (gen 2), or matlab-data3 (gen 3)."
                )
                if umbmid is not None:
                    hint += f"\nUM-BMID folder: {umbmid}"
                self._log(hint, "ERROR")
                QMessageBox.warning(self, "Metadata File — Not a Measurement", hint)
                return

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
            tax = taxonomy_from_metadata(meta)
            display.update(tax.to_display_dict())
            display["Antenna Radius"] = (
                f"{meta.get('antenna_radius_m', 0) * 100:.0f} cm"
            )
        self._populate_info_table(display)
        self._plot_raw_traces(dataset)
        self._log(f"Dataset '{summary.file_name}' loaded successfully.", "SUCCESS")
        self.dataset_loaded.emit(dataset)

    def _populate_info_table(self, info: dict) -> None:
        self.info_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.info_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.info_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.info_table.resizeColumnsToContents()

    def _plot_raw_traces(self, dataset: MicrowaveDataset) -> None:
        plot_complex_matrix(
            self.raw_figure,
            dataset.frequencies,
            dataset.s_parameters,
            "Loaded S-parameters",
        )
        self.raw_canvas.draw()

    def _auto_validation_scope(self) -> tuple[str, int, int]:
        strategy = str(self.auto_scope_combo.currentData() or "diverse")
        if strategy == "all":
            return "all", 20, 0
        if strategy == "tumor_and_healthy":
            return "tumor_and_healthy", 1, 2
        return "diverse", 1, 8

    def on_run_auto_validation_clicked(self) -> None:
        if self._auto_worker is not None and self._auto_worker.isRunning():
            QMessageBox.information(
                self,
                "Auto Validation Running",
                "Validation is already in progress. Use Cancel Auto Validation to stop it after the current scan.",
            )
            return

        strategy, max_files, max_scans = self._auto_validation_scope()
        roots = default_validation_roots()
        self._log(
            f"Starting auto validation on UM-BMID cubes: {', '.join(roots)} "
            f"(strategy={strategy}, max_files={max_files}, max_scans={max_scans}, profile=balanced). "
            "project/datasets sample files are not used.",
            "INFO",
        )
        self.auto_validate_button.setEnabled(False)
        self.auto_scope_combo.setEnabled(False)
        self.cancel_auto_button.setEnabled(True)
        self.load_button.setEnabled(False)
        self._auto_worker = AutoValidationWorker(
            roots,
            max_files=max_files,
            threshold_profile="balanced",
            include_reconstruction_checks=True,
            output_dir=self._results_dir,
            bmid_strategy=strategy,
            max_bmid_scans=max_scans,
        )
        self._auto_worker.progress_updated.connect(self._on_auto_progress)
        self._auto_worker.finished_ok.connect(self._on_auto_finished)
        self._auto_worker.failed.connect(self._on_auto_failed)
        self._auto_worker.start()

    def on_cancel_auto_validation_clicked(self) -> None:
        if self._auto_worker is None or not self._auto_worker.isRunning():
            return
        self._auto_worker.requestInterruption()
        self.cancel_auto_button.setEnabled(False)
        self._log("Stopping auto validation after the current scan…")

    def _on_auto_progress(self, percent: int, message: str) -> None:
        self._log(f"[Auto {percent}%] {message}")

    def _reset_auto_buttons(self) -> None:
        self.auto_validate_button.setEnabled(True)
        self.auto_scope_combo.setEnabled(True)
        self.cancel_auto_button.setEnabled(False)
        self.load_button.setEnabled(True)

    def _on_auto_failed(self, message: str) -> None:
        self._reset_auto_buttons()
        self._log(f"Auto validation failed: {message}", "ERROR")
        QMessageBox.critical(
            self,
            "Auto Validation Failed",
            f"{message}\n\nExisting load/preprocess/reconstruct workflows are unchanged.",
        )

    def _on_auto_finished(self, batch) -> None:
        self._reset_auto_buttons()
        self._log(batch.message, "SUCCESS")
        analysis = batch.report_paths.get("analysis")
        n_yes = sum(
            1
            for r in batch.results
            if ((r.details or {}).get("tumor_candidate") or {}).get("is_tumor_candidate")
        )
        showcase = pick_showcase_result(batch)
        showcase_line = "No showcase case selected."
        if showcase is not None:
            showcase_line = (
                f"Showcase: {Path(showcase.target.measurement_path).name} "
                f"scan {showcase.target.scan_index}"
            )
            self._log(showcase_line, "INFO")
        cancelled = str(batch.message).lower().startswith("cancelled")
        title = "Auto Validation Cancelled" if cancelled else "Auto Validation Complete"
        summary = (
            f"{batch.message}\n\n"
            f"Tumor-candidate Yes: {n_yes}\n"
            f"{showcase_line}\n"
            f"Analysis report:\n{analysis}"
        )
        QMessageBox.information(self, title, summary)
        if analysis and os.path.isfile(analysis):
            QDesktopServices.openUrl(QUrl.fromLocalFile(analysis))
        self.auto_validation_finished.emit(batch)


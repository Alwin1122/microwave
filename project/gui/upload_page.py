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
"""

from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data_loader.dataset_info import MicrowaveDataset
from data_loader.loader import load_dataset_with_summary
from utils.exceptions import MicrowaveFrameworkError
from utils.logger import StatusLog, get_logger

logger = get_logger(__name__)

SUPPORTED_FILE_FILTER = (
    "Microwave Datasets (*.mat *.s1p *.s2p *.s4p *.s8p);;"
    "MATLAB Files (*.mat);;"
    "Touchstone Files (*.s1p *.s2p *.s4p *.s8p);;"
    "All Files (*)"
)


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
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("Module 1 — Data Acquisition")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        # --- Load button row ---
        button_row = QHBoxLayout()
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.on_load_dataset_clicked)
        button_row.addWidget(self.load_button)

        self.file_label = QLabel("No file loaded.")
        self.file_label.setStyleSheet("color: #555;")
        button_row.addWidget(self.file_label, stretch=1)
        layout.addLayout(button_row)

        # --- Dataset info panel ---
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout()
        self.info_table = QTableWidget(0, 2)
        self.info_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        info_layout.addWidget(self.info_table)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group, stretch=1)

        # --- Status log ---
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group, stretch=1)

    # ------------------------------------------------------------------ #
    # Logic
    # ------------------------------------------------------------------ #
    def _log(self, message: str, level: str = "INFO") -> None:
        line = self.status_log.add(message, level)
        self.log_view.appendPlainText(line)

    def on_load_dataset_clicked(self) -> None:
        """
        Purpose:
            Handle the "Load Dataset" button: opens a file browser dialog,
            loads and validates the selected file, updates the info panel,
            and emits `dataset_loaded` on success.
        Input:
            None (triggered by Qt signal).
        Output:
            None. Side effect: populates info table / status log, emits
            dataset_loaded signal, or shows an error dialog.
        """
        start_dir = os.path.join(os.getcwd(), "datasets")
        if not os.path.isdir(start_dir):
            start_dir = os.getcwd()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Microwave Dataset", start_dir, SUPPORTED_FILE_FILTER
        )
        if not file_path:
            return  # user cancelled

        self.load_dataset_from_path(file_path)

    def load_dataset_from_path(self, file_path: str) -> None:
        """
        Purpose:
            Programmatic entry point (also used by tests) to load a
            dataset file by path, bypassing the file dialog.
        Input:
            file_path (str): path to the dataset file.
        Output:
            None. Populates UI state; emits dataset_loaded on success.
        """
        self._log(f"Loading file: {file_path}")
        try:
            dataset, summary = load_dataset_with_summary(file_path)
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
        self._populate_info_table(summary.to_display_dict())
        self._log(f"Dataset '{summary.file_name}' loaded successfully.", "SUCCESS")
        self.dataset_loaded.emit(dataset)

    def _populate_info_table(self, info: dict) -> None:
        self.info_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.info_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.info_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.info_table.resizeColumnsToContents()

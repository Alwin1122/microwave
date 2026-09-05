"""Shared Qt styling helpers for the Microwave Imaging Framework GUI."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

# Explicit light palette — avoids Windows dark-mode making labels unreadable.
_LIGHT_BG = QColor("#f4f5f7")
_LIGHT_BASE = QColor("#ffffff")
_LIGHT_TEXT = QColor("#0f172a")
_LIGHT_MUTED = QColor("#475569")
_LIGHT_DISABLED = QColor("#94a3b8")
_LIGHT_HIGHLIGHT = QColor("#dbeafe")
_LIGHT_BUTTON = QColor("#ffffff")

APP_STYLESHEET = """
QMainWindow, QDialog, QWidget {
    background-color: #f4f5f7;
    color: #0f172a;
    font-size: 13px;
}
QTabWidget {
    background-color: #f4f5f7;
}
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background-color: #ffffff;
    top: -1px;
    padding: 12px;
}
QTabBar::tab {
    background-color: transparent;
    color: #64748b;
    padding: 10px 18px;
    margin-right: 4px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: #0f172a;
    font-weight: 600;
    border-bottom: 2px solid #2563eb;
}
QTabBar::tab:hover {
    color: #1e293b;
}
QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: #ffffff;
    border: none;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    color: #0f172a;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #0f172a;
    background-color: #ffffff;
}
QLabel {
    background-color: transparent;
    color: #0f172a;
}
QFormLayout QLabel, QLabel {
    color: #0f172a;
}
QPushButton {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px 16px;
    min-height: 20px;
    color: #0f172a;
}
QPushButton:hover {
    background-color: #f8fafc;
    border-color: #94a3b8;
}
QPushButton:pressed {
    background-color: #e2e8f0;
}
QPushButton:disabled {
    color: #94a3b8;
    background-color: #f1f5f9;
    border-color: #e2e8f0;
}
QPushButton#primaryButton {
    background-color: #2563eb;
    border: 1px solid #1d4ed8;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background-color: #1d4ed8;
}
QPushButton#primaryButton:disabled {
    background-color: #93c5fd;
    border-color: #93c5fd;
    color: #eff6ff;
}
QComboBox, QLineEdit, QPlainTextEdit, QListWidget, QTableWidget, QSpinBox {
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}
QComboBox {
    min-height: 22px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #0f172a;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
    border: 1px solid #cbd5e1;
}
QComboBox:disabled, QLineEdit:disabled {
    color: #94a3b8;
    background-color: #f1f5f9;
}
QProgressBar {
    border: none;
    border-radius: 4px;
    text-align: center;
    background-color: #e2e8f0;
    height: 8px;
    max-height: 8px;
}
QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 4px;
}
QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    spacing: 10px;
    padding: 8px 12px;
    color: #0f172a;
}
QLabel#pageTitle {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
    background-color: transparent;
    padding: 0;
}
QLabel#modeBanner {
    background-color: transparent;
    border: none;
    padding: 0;
    color: #1d4ed8;
    font-size: 12px;
}
QLabel#modeBanner[mode="bmid"] {
    color: #047857;
}
QLabel#modeBanner[mode="s2p"] {
    color: #1d4ed8;
}
QLabel#modeBanner[mode="info"] {
    color: #475569;
}
QLabel#hintLabel {
    color: #475569;
    font-size: 12px;
    background-color: transparent;
}
QHeaderView::section {
    background-color: #f8fafc;
    border: none;
    border-bottom: 1px solid #e2e8f0;
    padding: 8px;
    font-weight: 600;
    color: #334155;
}
QTableWidget {
    gridline-color: #e2e8f0;
    alternate-background-color: #f8fafc;
}
QCheckBox {
    spacing: 8px;
    color: #0f172a;
    background-color: transparent;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    background-color: #ffffff;
}
QCheckBox::indicator:checked {
    background-color: #2563eb;
    border-color: #1d4ed8;
}
"""


def _build_light_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, _LIGHT_BG)
    palette.setColor(QPalette.ColorRole.WindowText, _LIGHT_TEXT)
    palette.setColor(QPalette.ColorRole.Base, _LIGHT_BASE)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Text, _LIGHT_TEXT)
    palette.setColor(QPalette.ColorRole.Button, _LIGHT_BUTTON)
    palette.setColor(QPalette.ColorRole.ButtonText, _LIGHT_TEXT)
    palette.setColor(QPalette.ColorRole.ToolTipBase, _LIGHT_BASE)
    palette.setColor(QPalette.ColorRole.ToolTipText, _LIGHT_TEXT)
    palette.setColor(QPalette.ColorRole.PlaceholderText, _LIGHT_MUTED)
    palette.setColor(QPalette.ColorRole.Highlight, _LIGHT_HIGHLIGHT)
    palette.setColor(QPalette.ColorRole.HighlightedText, _LIGHT_TEXT)
    palette.setColor(QPalette.ColorRole.BrightText, _LIGHT_TEXT)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, _LIGHT_DISABLED)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, _LIGHT_DISABLED)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, _LIGHT_DISABLED)
    return palette


def set_page_title(label: QLabel, text: str) -> None:
    label.setObjectName("pageTitle")
    label.setText(text)


def set_primary_button(button: QPushButton) -> None:
    button.setObjectName("primaryButton")


def set_mode_banner(label: QLabel, mode: str, text: str) -> None:
    label.setObjectName("modeBanner")
    label.setProperty("mode", mode)
    label.setText(text)
    label.setWordWrap(True)
    style = label.style()
    if style is not None:
        style.unpolish(label)
        style.polish(label)


def apply_app_style(widget: QWidget) -> None:
    """Force light theme + stylesheet so Settings labels stay readable."""
    app = widget if isinstance(widget, QApplication) else QApplication.instance()
    if app is not None:
        app.setStyle("Fusion")
        app.setPalette(_build_light_palette())
        app.setStyleSheet(APP_STYLESHEET)
    else:
        widget.setPalette(_build_light_palette())
        widget.setStyleSheet(APP_STYLESHEET)

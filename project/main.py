"""
main.py

Purpose:
    Application entry point for the Microwave Imaging Framework desktop
    app (Module 1: Data Acquisition, Module 2: Signal Preprocessing).

Input:
    None (command-line launch).

Output:
    Launches the PySide6 GUI event loop.

Description:
    Run with:  python main.py
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from gui.main_window import MainWindow
from gui.styles import apply_app_style
from utils.logger import get_logger

logger = get_logger(__name__)


def _bring_window_to_front(window: QWidget) -> None:
    """Restore and raise the GUI so it is not hidden behind other windows."""
    window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    window.showNormal()
    window.raise_()
    window.activateWindow()
    if sys.platform == "win32":
        try:
            import ctypes

            hwnd = int(window.winId())
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _drop_always_on_top() -> None:
        window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        window.show()
        window.raise_()
        window.activateWindow()

    QTimer.singleShot(2500, _drop_always_on_top)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Microwave Imaging Framework")
    apply_app_style(app)

    window = MainWindow()
    _bring_window_to_front(window)

    logger.info("Application started.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

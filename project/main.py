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

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Microwave Imaging Framework")

    window = MainWindow()
    window.show()

    logger.info("Application started.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

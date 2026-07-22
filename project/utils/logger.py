"""
utils/logger.py

Purpose:
    Provide a single, consistently configured logger for the whole
    application, plus a lightweight in-memory "status log" collector that
    the GUI can poll/display to the user (Module 1 & Module 2 both require
    a "Status Log" panel).

Input:
    None (module-level configuration).

Output:
    get_logger(name) -> logging.Logger
    StatusLog class -> in-memory list of timestamped status messages.

Description:
    Keeping logging setup here avoids duplicating logging.basicConfig calls
    across modules and prevents duplicate log handlers when modules are
    imported multiple times (e.g. from tests and from the GUI).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

_CONFIGURED = False


def get_logger(name: str = "mwi_framework") -> logging.Logger:
    """
    Purpose:
        Return a configured logger instance shared across the application.
    Input:
        name (str): logger name, typically __name__ of the calling module.
    Output:
        logging.Logger instance writing to stdout with timestamps.
    """
    global _CONFIGURED
    root = logging.getLogger("mwi_framework")
    if not _CONFIGURED:
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    if name == "mwi_framework":
        return root
    return logging.getLogger(f"mwi_framework.{name}")


class StatusLog:
    """
    Purpose:
        In-memory, timestamped log of status/progress messages intended for
        display in a GUI status panel (separate from the Python logging
        stream, so the GUI can render it directly without hooking into
        logging handlers).
    Input:
        None.
    Output:
        A growing list of formatted strings, accessible via `entries` or
        `as_text()`.
    """

    def __init__(self) -> None:
        self.entries: list[str] = []

    def add(self, message: str, level: str = "INFO") -> str:
        """Append a timestamped message and return the formatted line."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {level}: {message}"
        self.entries.append(line)
        return line

    def info(self, message: str) -> str:
        return self.add(message, "INFO")

    def success(self, message: str) -> str:
        return self.add(message, "SUCCESS")

    def warning(self, message: str) -> str:
        return self.add(message, "WARNING")

    def error(self, message: str) -> str:
        return self.add(message, "ERROR")

    def as_text(self) -> str:
        return "\n".join(self.entries)

    def clear(self) -> None:
        self.entries.clear()

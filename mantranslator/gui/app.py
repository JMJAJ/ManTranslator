"""Qt application bootstrap: create the app, apply the dark theme, show window."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .. import __app_name__
from ..config import ensure_dirs
from .main_window import MainWindow
from .theme import apply_dark_theme


def run() -> int:
    """Create and run the GUI, returning the process exit code."""
    ensure_dirs()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    apply_dark_theme(app)

    window = MainWindow()
    window.show()
    return app.exec()

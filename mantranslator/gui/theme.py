"""Dark theme styling for the Qt GUI.

Provides a single dark palette plus a matching Qt Style Sheet (QSS) so every
widget shares consistent colors, spacing and focus states.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


# Central color tokens reused by both the palette and the QSS below.
BG = "#1e1f22"
BG_ELEVATED = "#2b2d31"
BG_INPUT = "#26282c"
BORDER = "#3a3d43"
TEXT = "#e6e6e6"
TEXT_MUTED = "#9aa0a6"
ACCENT = "#4c8dff"
ACCENT_HOVER = "#6aa1ff"
DANGER = "#ff5c5c"
SUCCESS = "#4cd07d"


def _palette() -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(BG_INPUT))
    pal.setColor(QPalette.AlternateBase, QColor(BG_ELEVATED))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(BG_ELEVATED))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ToolTipBase, QColor(BG_ELEVATED))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_MUTED))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_MUTED))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_MUTED))
    return pal


QSS = f"""
* {{
    outline: none;
}}
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background-color: {BG};
}}
QLabel {{
    background: transparent;
}}
QLabel[muted="true"] {{
    color: {TEXT_MUTED};
}}
QLabel[heading="true"] {{
    font-size: 16px;
    font-weight: 600;
}}
QFrame#Card, QGroupBox {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QGroupBox {{
    margin-top: 14px;
    padding: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_MUTED};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}
QPushButton {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    border: 1px solid {ACCENT};
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
}}
QPushButton[accent="true"] {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background-color: {ACCENT_HOVER};
    border: 1px solid {ACCENT_HOVER};
}}
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QListWidget::item, QTreeWidget::item {{
    padding: 5px 6px;
    border-radius: 4px;
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {ACCENT};
    color: #ffffff;
}}
QHeaderView::section {{
    background-color: {BG_ELEVATED};
    color: {TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px;
}}
QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    height: 18px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 6px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 6px;
    min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0; width: 0;
}}
QToolTip {{
    background-color: {BG_ELEVATED};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px 6px;
    border-radius: 4px;
}}
QSplitter::handle {{
    background: {BORDER};
}}
"""


def apply_dark_theme(app: QApplication) -> None:
    """Apply the Fusion style, dark palette and QSS to the application."""
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setStyleSheet(QSS)

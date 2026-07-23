"""Main application window: navigation, editor, settings, glossary and runs."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __app_name__
from ..config import Settings, load_settings, save_settings
from ..project.manager import ProjectManager
from .editor_view import EditorView
from .project_panel import ProjectPanel
from .settings_panel import SettingsPanel
from .worker import TranslationWorker, WorkerThread


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(__app_name__)
        self.resize(1360, 860)

        self.settings: Settings = load_settings()
        self.manager = ProjectManager()
        self._worker: WorkerThread | None = None

        self._build_ui()
        self._wire()
        self._restore_last_project()

    # --------------------------------------------------------------- build UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        self.project_panel = ProjectPanel(self.manager, self.settings)
        splitter.addWidget(self.project_panel)

        self.tabs = QTabWidget()
        self.editor = EditorView(self.manager, self.settings)
        self.settings_panel = SettingsPanel(self.settings)
        self.glossary_tab = self._build_glossary_tab()
        self.tabs.addTab(self.editor, "Editor")
        self.tabs.addTab(self.glossary_tab, "Glossary")
        self.tabs.addTab(self.settings_panel, "Settings")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        outer.addWidget(splitter, 1)

        # Run controls + progress.
        controls = QHBoxLayout()
        self.stage_label = QLabel("Ready")
        self.stage_label.setProperty("muted", True)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        controls.addWidget(self.stage_label, 1)
        controls.addWidget(self.progress, 2)
        controls.addWidget(self.cancel_btn)
        outer.addLayout(controls)

        self.setStatusBar(QStatusBar())
        self._set_status("Configure a provider in Settings, then create a project.")

    def _build_glossary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        info = QLabel(
            "Names and terms are reused on every translation to keep chapters "
            "consistent. Edit the markdown tables below and save."
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
        layout.addWidget(info)

        self.glossary_edit = QPlainTextEdit()
        self.glossary_edit.setPlaceholderText("Open a project to edit its glossary.")
        layout.addWidget(self.glossary_edit, 1)

        row = QHBoxLayout()
        self.reload_gloss_btn = QPushButton("Reload")
        self.save_gloss_btn = QPushButton("Save Glossary")
        self.save_gloss_btn.setProperty("accent", True)
        self.reload_gloss_btn.clicked.connect(self._reload_glossary)
        self.save_gloss_btn.clicked.connect(self._save_glossary)
        row.addStretch(1)
        row.addWidget(self.reload_gloss_btn)
        row.addWidget(self.save_gloss_btn)
        layout.addLayout(row)
        return tab

    # ----------------------------------------------------------------- wiring
    def _wire(self) -> None:
        self.project_panel.page_selected.connect(self._on_page_selected)
        self.project_panel.translate_requested.connect(self._on_translate)
        self.project_panel.project_changed.connect(self._on_project_changed)
        self.settings_panel.settings_changed.connect(self._on_settings_changed)
        self.editor.page_edited.connect(lambda _p: self.project_panel.refresh())

    # --------------------------------------------------------------- projects
    def _restore_last_project(self) -> None:
        last = self.settings.last_project
        if last and (Path(last) / "project.json").exists():
            if self.project_panel.load_project(last):
                self._on_project_changed()

    def _on_project_changed(self) -> None:
        if self.manager.project:
            self.settings.last_project = self.manager.project.root
            save_settings(self.settings)
            self._reload_glossary()
            self._set_status(f"Project '{self.manager.project.name}' ready.")

    def _on_page_selected(self, chapter, page) -> None:
        self.editor.load_page(chapter, page)
        self.tabs.setCurrentWidget(self.editor)

    def _on_settings_changed(self) -> None:
        # Editor holds references to the same Settings object; refresh fonts too.
        self.editor.settings = self.settings
        self.editor.refresh_fonts()

    # ------------------------------------------------------------ translation
    def _on_translate(self, chapter, pages) -> None:
        if self._worker is not None:
            QMessageBox.information(self, "Busy", "A translation run is in progress.")
            return
        if not self.settings.translation_provider:
            QMessageBox.warning(
                self, "No provider",
                "Select a translation provider in Settings first.",
            )
            self.tabs.setCurrentWidget(self.settings_panel)
            return

        worker = TranslationWorker(self.manager, self.settings, chapter, pages)
        worker.progress.connect(self._on_progress)
        worker.page_progress.connect(self._on_page_progress)
        worker.page_done.connect(self._on_page_done)
        worker.page_failed.connect(self._on_page_failed)
        worker.glossary_updated.connect(self._reload_glossary)
        worker.finished.connect(self._on_finished)

        self._worker = WorkerThread(worker)
        self._set_running(True)
        self._set_status(f"Translating {len(pages)} page(s)...")
        self._worker.start()

    def _on_progress(self, stage: str, current: int, total: int) -> None:
        self.stage_label.setText(stage)
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
        else:
            self.progress.setRange(0, 0)

    def _on_page_progress(self, index: int, count: int) -> None:
        self._set_status(f"Page {index + 1} of {count}")

    def _on_page_done(self, page) -> None:
        self.project_panel.refresh()
        if self.editor._page is page:  # currently open page finished
            self.editor.load_page(self.editor._chapter, page)

    def _on_page_failed(self, page, message: str) -> None:
        self.project_panel.refresh()
        self._set_status(f"Page failed: {message}")

    def _on_finished(self, succeeded: int, failed: int) -> None:
        # Wait for the worker thread to fully terminate before dropping the
        # holder, so the (parent-less, Python-owned) QThread is never destroyed
        # while still running.
        holder = self._worker
        self._worker = None
        if holder is not None:
            try:
                holder.thread.wait(5000)
            except RuntimeError:
                pass
        self._set_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if succeeded else 0)
        self.stage_label.setText("Done")
        self._set_status(f"Finished. {succeeded} succeeded, {failed} failed.")
        self.project_panel.refresh()

    def _cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.stage_label.setText("Cancelling...")

    def _set_running(self, running: bool) -> None:
        self.cancel_btn.setEnabled(running)
        if running:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)

    # --------------------------------------------------------------- glossary
    def _glossary_path(self) -> Path | None:
        if not self.manager.project:
            return None
        return Path(self.manager.project.root) / "glossary.md"

    def _reload_glossary(self) -> None:
        path = self._glossary_path()
        if not path:
            self.glossary_edit.setPlainText("")
            return
        # Reload the glossary object from disk so translation uses fresh data.
        if self.manager.glossary is not None:
            from ..project.glossary import Glossary

            self.manager.glossary = Glossary.load(path)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        self.glossary_edit.setPlainText(text)

    def _save_glossary(self) -> None:
        path = self._glossary_path()
        if not path:
            QMessageBox.information(self, "No project", "Open a project first.")
            return
        try:
            path.write_text(self.glossary_edit.toPlainText(), encoding="utf-8")
            from ..project.glossary import Glossary

            self.manager.glossary = Glossary.load(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._set_status("Glossary saved.")

    # ----------------------------------------------------------------- events
    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._worker is not None:
            # Stop the running translation thread cleanly before teardown.
            # Keep the Python reference (do not null it) so the parent-less
            # QThread is not garbage-collected while it may still be running;
            # the interpreter exit reclaims it after the process closes.
            self._worker.stop_and_wait()
        save_settings(self.settings)
        super().closeEvent(event)

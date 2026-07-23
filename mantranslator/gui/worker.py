"""QThread worker that runs the translation pipeline off the UI thread.

The worker emits Qt signals for progress, per-page completion, errors and
overall completion so the GUI stays responsive and can show a live progress
bar with a working Cancel button. Heavy pipeline objects are constructed inside
``run`` so model loading happens on the worker thread.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from ..config import Settings
from ..core.models import Chapter, Page
from ..core.pipeline import Cancelled, PipelineHooks, TranslationPipeline
from ..project.glossary import Glossary
from ..project.manager import ProjectManager


class TranslationWorker(QObject):
    """Runs the pipeline over a list of pages in a background thread."""

    progress = Signal(str, int, int)         # stage label, current, total
    page_progress = Signal(int, int)         # page index, page count
    page_done = Signal(object)               # the completed Page
    page_failed = Signal(object, str)        # Page, error message
    finished = Signal(int, int)              # succeeded, failed
    glossary_updated = Signal()

    def __init__(self, manager: ProjectManager, settings: Settings,
                 chapter: Chapter, pages: list[Page]) -> None:
        super().__init__()
        self._manager = manager
        self._settings = settings
        self._chapter = chapter
        self._pages = pages
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        assert self._manager.project is not None
        project = self._manager.project
        glossary: Glossary = self._manager.glossary or Glossary.load(
            project.root + "/glossary.md"
        )
        pipeline = TranslationPipeline(
            settings=self._settings,
            glossary=glossary,
            source_lang=project.source_lang,
            target_lang=project.target_lang,
        )

        total = len(self._pages)
        succeeded = failed = 0
        for idx, page in enumerate(self._pages):
            if self._cancel:
                break
            self.page_progress.emit(idx, total)
            hooks = PipelineHooks(
                on_progress=lambda stage, cur, tot: self.progress.emit(stage, cur, tot),
                is_cancelled=lambda: self._cancel,
            )
            out_path = self._manager.output_path_for(self._chapter, page)
            try:
                pipeline.process_page(page, out_path, hooks)
                succeeded += 1
                self.page_done.emit(page)
            except Cancelled:
                break
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                self.page_failed.emit(page, str(exc))

        # Persist glossary and project state after the batch.
        try:
            self._manager.save_glossary()
            self._manager.save()
        except OSError:
            pass
        self.glossary_updated.emit()
        self.finished.emit(succeeded, failed)


class WorkerThread:
    """Convenience holder that owns a QThread + worker and wires teardown."""

    def __init__(self, worker: TranslationWorker) -> None:
        self.thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        # NOTE: no deleteLater here. The QThread has no parent, so it is owned by
        # this Python holder; MainWindow keeps the holder alive and calls
        # wait() before dropping it, preventing 'destroyed while running'.

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        self.worker.cancel()

    def stop_and_wait(self, msecs: int = 8000) -> None:
        """Request cancellation and block until the thread stops (bounded).

        Used on application exit so the QThread is not destroyed while still
        running. The pipeline checks the cancel flag between stages, so this may
        wait for the current stage to finish before returning.
        """
        self.worker.cancel()
        try:
            if self.thread.isRunning():
                self.thread.quit()
                self.thread.wait(msecs)
        except RuntimeError:
            pass

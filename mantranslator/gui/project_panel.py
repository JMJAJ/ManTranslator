"""Project panel: create/open projects, manage chapters and import images."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..core.models import Chapter, Page, PageStatus
from ..project.manager import ProjectManager
from .settings_panel import LANGS


class NewProjectDialog(QDialog):
    """Collects the details needed to create a new project."""

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(460)
        self._parent_dir = ""

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("My Series")

        loc_row = QHBoxLayout()
        self.location = QLineEdit()
        self.location.setPlaceholderText("Folder to create the project in")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        loc_row.addWidget(self.location, 1)
        loc_row.addWidget(browse)

        self.source = QComboBox()
        self.target = QComboBox()
        for code, label in LANGS:
            self.source.addItem(label, code)
            self.target.addItem(label, code)
        self.source.setCurrentIndex(self.source.findData(settings.source_lang))
        self.target.setCurrentIndex(self.target.findData(settings.target_lang))

        form.addRow("Name", self.name)
        form.addRow("Location", loc_row)
        form.addRow("Source language", self.source)
        form.addRow("Target language", self.target)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose parent folder")
        if path:
            self.location.setText(path)

    def _accept(self) -> None:
        if not self.name.text().strip() or not self.location.text().strip():
            QMessageBox.warning(self, "Missing info", "Enter a name and location.")
            return
        self.accept()

    def values(self):
        return (
            self.location.text().strip(),
            self.name.text().strip(),
            self.source.currentData(),
            self.target.currentData(),
        )


class ProjectPanel(QWidget):
    """Left-hand navigation for the current project."""

    page_selected = Signal(object, object)     # Chapter, Page
    translate_requested = Signal(object, list)  # Chapter, list[Page]
    project_changed = Signal()

    def __init__(self, manager: ProjectManager, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.settings = settings

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        new_btn = QPushButton("New")
        open_btn = QPushButton("Open")
        new_btn.clicked.connect(self._new_project)
        open_btn.clicked.connect(self._open_project)
        top.addWidget(new_btn)
        top.addWidget(open_btn)
        layout.addLayout(top)

        self.title = QLabel("No project open")
        self.title.setProperty("heading", True)
        layout.addWidget(self.title)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Chapters & Pages", "Status"])
        self.tree.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.tree, 1)

        actions = QVBoxLayout()
        self.add_chapter_btn = QPushButton("Add Chapter")
        self.import_btn = QPushButton("Import Images...")
        self.translate_chapter_btn = QPushButton("Translate Chapter")
        self.translate_sel_btn = QPushButton("Translate Selected Page")
        self.add_chapter_btn.clicked.connect(self._add_chapter)
        self.import_btn.clicked.connect(self._import_images)
        self.translate_chapter_btn.clicked.connect(self._translate_chapter)
        self.translate_sel_btn.clicked.connect(self._translate_selected)
        for btn in (self.add_chapter_btn, self.import_btn,
                    self.translate_chapter_btn, self.translate_sel_btn):
            btn.setEnabled(False)
            actions.addWidget(btn)
        layout.addLayout(actions)

    # --------------------------------------------------------------- projects
    def _new_project(self) -> None:
        dialog = NewProjectDialog(self.settings, self)
        if not dialog.exec():
            return
        parent_dir, name, source, target = dialog.values()
        try:
            self.manager.create(parent_dir, name, source, target)
        except OSError as exc:
            QMessageBox.critical(self, "Error", f"Could not create project:\n{exc}")
            return
        self._refresh()
        self.project_changed.emit()

    def _open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open project folder")
        if not path:
            return
        if not (Path(path) / "project.json").exists():
            QMessageBox.warning(self, "Not a project",
                                "That folder has no project.json.")
            return
        try:
            self.manager.open(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Error", f"Could not open project:\n{exc}")
            return
        self._refresh()
        self.project_changed.emit()

    def load_project(self, root: str) -> bool:
        """Open a project by path (used to restore the last session)."""
        try:
            self.manager.open(root)
        except (OSError, ValueError):
            return False
        self._refresh()
        return True

    # --------------------------------------------------------------- chapters
    def _add_chapter(self) -> None:
        if not self.manager.project:
            return
        name, ok = QInputDialog.getText(self, "Add Chapter", "Chapter name:")
        if not ok or not name.strip():
            return
        self.manager.add_chapter(name.strip())
        self._refresh()

    def _import_images(self) -> None:
        chapter = self._selected_chapter()
        if chapter is None:
            QMessageBox.information(self, "Select a chapter",
                                    "Select a chapter to import images into.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Import images", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif)",
        )
        if not files:
            return
        self.manager.import_images(chapter, files)
        self._refresh()

    # ------------------------------------------------------------ translation
    def _translate_chapter(self) -> None:
        chapter = self._selected_chapter()
        if chapter and chapter.pages:
            self.translate_requested.emit(chapter, list(chapter.pages))

    def _translate_selected(self) -> None:
        chapter, page = self._selected_chapter_and_page()
        if chapter and page:
            self.translate_requested.emit(chapter, [page])

    # ----------------------------------------------------------------- helpers
    def _refresh(self) -> None:
        project = self.manager.project
        has_project = project is not None
        for btn in (self.add_chapter_btn, self.import_btn,
                    self.translate_chapter_btn, self.translate_sel_btn):
            btn.setEnabled(has_project)
        if not has_project:
            self.title.setText("No project open")
            self.tree.clear()
            return
        self.title.setText(f"{project.name}  ({project.source_lang} -> {project.target_lang})")
        self.tree.clear()
        for chapter in project.chapters:
            citem = QTreeWidgetItem([chapter.name, f"{len(chapter.pages)} pages"])
            citem.setData(0, 0x0100, ("chapter", chapter))  # Qt.UserRole
            for page in chapter.pages:
                label = Path(page.image_path).name
                pitem = QTreeWidgetItem([label, _status_label(page.status)])
                pitem.setData(0, 0x0100, ("page", chapter, page))
                citem.addChild(pitem)
            self.tree.addTopLevelItem(citem)
        self.tree.expandAll()

    def refresh(self) -> None:
        self._refresh()

    def _current_payload(self):
        item = self.tree.currentItem()
        return item.data(0, 0x0100) if item else None

    def _selected_chapter(self) -> Chapter | None:
        payload = self._current_payload()
        if not payload:
            return None
        if payload[0] == "chapter":
            return payload[1]
        if payload[0] == "page":
            return payload[1]
        return None

    def _selected_chapter_and_page(self) -> tuple[Chapter | None, Page | None]:
        payload = self._current_payload()
        if payload and payload[0] == "page":
            return payload[1], payload[2]
        return self._selected_chapter(), None

    def _on_selection(self) -> None:
        chapter, page = self._selected_chapter_and_page()
        if chapter and page:
            self.page_selected.emit(chapter, page)


def _status_label(status: str) -> str:
    return {
        PageStatus.PENDING.value: "Pending",
        PageStatus.DETECTED.value: "Detected",
        PageStatus.TRANSLATED.value: "Translated",
        PageStatus.RENDERED.value: "Done",
        PageStatus.ERROR.value: "Error",
    }.get(status, status)

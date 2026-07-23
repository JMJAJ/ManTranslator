"""Editor view: page canvas with region overlays and a region inspector.

The left side shows the page (source with detected-region overlays, or the
rendered translation) on a zoomable canvas. The right side lets the user edit a
selected region's translation, font, size, color, alignment and orientation,
then re-render the page locally (no AI calls). Pages can be exported here too.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..core.models import Alignment, Orientation, Page, PageStatus, TextRegion
from ..core.pipeline import TranslationPipeline
from ..core.render import FontLibrary
from ..project.manager import ProjectManager


class RegionItem(QGraphicsRectItem):
    """A selectable overlay rectangle bound to a region index."""

    def __init__(self, index: int, rect: QRectF) -> None:
        super().__init__(rect)
        self.index = index
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self._set_style(selected=False)

    def _set_style(self, selected: bool) -> None:
        color = QColor("#4c8dff") if selected else QColor("#ff5c5c")
        pen = QPen(color)
        pen.setWidth(3 if selected else 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(60 if selected else 25)
        self.setBrush(QBrush(fill))


class ImageCanvas(QGraphicsView):
    """Zoomable image canvas that emits the clicked region index."""

    region_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._pixmap_item = None
        self._region_items: list[RegionItem] = []

    def show_image(self, path: str, regions: list[TextRegion] | None = None) -> None:
        self._scene.clear()
        self._region_items = []
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._pixmap_item = None
            return
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        if regions:
            for i, region in enumerate(regions):
                x, y, w, h = region.bbox
                item = RegionItem(i, QRectF(x, y, w, h))
                self._scene.addItem(item)
                self._region_items.append(item)
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def highlight(self, index: int) -> None:
        for item in self._region_items:
            item._set_style(selected=(item.index == index))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        item = self.itemAt(event.pos())
        if isinstance(item, RegionItem):
            self.region_clicked.emit(item.index)
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)


class EditorView(QWidget):
    """Canvas + inspector for reviewing and correcting a page."""

    page_edited = Signal(object)  # the edited Page

    def __init__(self, manager: ProjectManager, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.settings = settings
        self.fonts = FontLibrary()
        self._page: Page | None = None
        self._chapter = None
        self._region: TextRegion | None = None
        self._showing_output = False

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self._build_canvas_side())
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.addWidget(splitter)

    # --------------------------------------------------------------- widgets
    def _build_canvas_side(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        toolbar = QHBoxLayout()
        self.view_toggle = QPushButton("Show Translated")
        self.view_toggle.setCheckable(True)
        self.view_toggle.toggled.connect(self._toggle_view)
        self.export_btn = QPushButton("Export Page...")
        self.export_btn.clicked.connect(self._export_page)
        self.status_label = QLabel("")
        self.status_label.setProperty("muted", True)
        toolbar.addWidget(self.view_toggle)
        toolbar.addWidget(self.export_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(self.status_label)
        layout.addLayout(toolbar)

        self.canvas = ImageCanvas()
        self.canvas.region_clicked.connect(self._select_region)
        layout.addWidget(self.canvas, 1)
        return container

    def _build_inspector(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        heading = QLabel("Region")
        heading.setProperty("heading", True)
        layout.addWidget(heading)

        layout.addWidget(QLabel("Original text"))
        self.source_edit = QPlainTextEdit()
        self.source_edit.setReadOnly(True)
        self.source_edit.setMaximumHeight(90)
        layout.addWidget(self.source_edit)

        layout.addWidget(QLabel("Translation"))
        self.translation_edit = QPlainTextEdit()
        self.translation_edit.setMaximumHeight(120)
        layout.addWidget(self.translation_edit)

        form = QFormLayout()
        self.font_combo = QComboBox()
        self.font_combo.addItem("Auto (best match)", "")
        for name in self.fonts.names:
            self.font_combo.addItem(name, name)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(6, 200)

        self.align_combo = QComboBox()
        self.align_combo.addItem("Left", Alignment.LEFT.value)
        self.align_combo.addItem("Center", Alignment.CENTER.value)
        self.align_combo.addItem("Right", Alignment.RIGHT.value)

        self.orient_combo = QComboBox()
        self.orient_combo.addItem("Horizontal", Orientation.HORIZONTAL.value)
        self.orient_combo.addItem("Vertical", Orientation.VERTICAL.value)

        self.color_btn = QPushButton("Text color")
        self.color_btn.clicked.connect(self._pick_color)
        self._color = QColor(0, 0, 0)

        form.addRow("Font", self.font_combo)
        form.addRow("Size", self.size_spin)
        form.addRow("Align", self.align_combo)
        form.addRow("Orientation", self.orient_combo)
        form.addRow("Color", self.color_btn)
        layout.addLayout(form)

        self.apply_btn = QPushButton("Apply && Re-render Page")
        self.apply_btn.setProperty("accent", True)
        self.apply_btn.clicked.connect(self._apply_and_render)
        layout.addWidget(self.apply_btn)
        layout.addStretch(1)

        self._set_inspector_enabled(False)
        return container

    # --------------------------------------------------------------- loading
    def load_page(self, chapter, page: Page) -> None:
        self._chapter = chapter
        self._page = page
        self._region = None
        self._showing_output = False
        self.view_toggle.blockSignals(True)
        self.view_toggle.setChecked(False)
        self.view_toggle.setText("Show Translated")
        self.view_toggle.blockSignals(False)
        self.view_toggle.setEnabled(bool(page.output_path and Path(page.output_path).exists()))
        self._render_canvas()
        self._set_inspector_enabled(False)
        self._update_status()

    def _render_canvas(self) -> None:
        if not self._page:
            return
        if self._showing_output and self._page.output_path:
            self.canvas.show_image(self._page.output_path, regions=None)
        else:
            self.canvas.show_image(self._page.image_path, regions=self._page.regions)

    def _toggle_view(self, checked: bool) -> None:
        self._showing_output = checked
        self.view_toggle.setText("Show Original" if checked else "Show Translated")
        self._render_canvas()

    # ------------------------------------------------------------- selection
    def _select_region(self, index: int) -> None:
        if not self._page or index >= len(self._page.regions):
            return
        self.canvas.highlight(index)
        region = self._page.regions[index]
        self._region = region
        self.source_edit.setPlainText(region.source_text)
        self.translation_edit.setPlainText(region.translated_text)
        i = self.font_combo.findData(region.font_name)
        self.font_combo.setCurrentIndex(i if i >= 0 else 0)
        self.size_spin.setValue(region.font_size or 24)
        self.align_combo.setCurrentIndex(max(0, self.align_combo.findData(region.alignment)))
        self.orient_combo.setCurrentIndex(max(0, self.orient_combo.findData(region.orientation)))
        self._color = QColor(*region.text_color)
        self._update_color_button()
        self._set_inspector_enabled(True)

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(self._color, self, "Text color")
        if color.isValid():
            self._color = color
            self._update_color_button()

    def _update_color_button(self) -> None:
        c = self._color
        text_col = "#000000" if c.lightness() > 128 else "#ffffff"
        self.color_btn.setStyleSheet(
            f"background-color: {c.name()}; color: {text_col};"
        )

    # ---------------------------------------------------------------- actions
    def _apply_and_render(self) -> None:
        if not self._page or not self._region:
            return
        region = self._region
        region.translated_text = self.translation_edit.toPlainText().strip()
        region.font_name = self.font_combo.currentData() or ""
        region.font_size = self.size_spin.value()
        region.alignment = self.align_combo.currentData()
        region.orientation = self.orient_combo.currentData()
        region.text_color = [self._color.red(), self._color.green(), self._color.blue()]
        region.manual_override = True

        try:
            pipeline = TranslationPipeline(
                self.settings,
                self.manager.glossary,
                self.manager.project.source_lang,
                self.manager.project.target_lang,
            )
            out_path = self.manager.output_path_for(self._chapter, self._page)
            pipeline.rerender_page(self._page, out_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Render failed", str(exc))
            return
        self.manager.save()
        self.view_toggle.setEnabled(True)
        self.view_toggle.setChecked(True)  # jump to the updated result
        self._showing_output = True
        self._render_canvas()
        self._update_status()
        self.page_edited.emit(self._page)

    def _export_page(self) -> None:
        if not self._page or not self._page.output_path:
            QMessageBox.information(self, "Nothing to export",
                                    "Translate or render this page first.")
            return
        suggested = Path(self._page.image_path).stem + "_translated.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export page", suggested, "PNG image (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        try:
            QPixmap(self._page.output_path).save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ----------------------------------------------------------------- state
    def _set_inspector_enabled(self, enabled: bool) -> None:
        for w in (self.translation_edit, self.font_combo, self.size_spin,
                  self.align_combo, self.orient_combo, self.color_btn,
                  self.apply_btn):
            w.setEnabled(enabled)

    def _update_status(self) -> None:
        if not self._page:
            self.status_label.setText("")
            return
        n = len(self._page.regions)
        status = self._page.status
        extra = f" - {self._page.error}" if status == PageStatus.ERROR.value else ""
        self.status_label.setText(f"{n} regions - {status}{extra}")

    def refresh_fonts(self) -> None:
        self.fonts.refresh()
        current = self.font_combo.currentData()
        self.font_combo.clear()
        self.font_combo.addItem("Auto (best match)", "")
        for name in self.fonts.names:
            self.font_combo.addItem(name, name)
        i = self.font_combo.findData(current)
        if i >= 0:
            self.font_combo.setCurrentIndex(i)

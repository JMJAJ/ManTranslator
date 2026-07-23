"""Settings panel: providers, languages, OCR/inpaint options and fonts."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings, save_settings
from .provider_config import ProviderDialog

# (code, label) pairs offered in the language selectors.
LANGS = [
    ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"), ("en", "English"),
    ("es", "Spanish"), ("fr", "French"), ("de", "German"), ("pt", "Portuguese"),
    ("ru", "Russian"), ("it", "Italian"), ("id", "Indonesian"), ("vi", "Vietnamese"),
]

OCR_ENGINES = [
    ("auto", "Auto (by source language)"),
    ("manga-ocr", "manga-ocr (Japanese)"),
    ("paddleocr", "PaddleOCR (KR/CN/EN)"),
    ("tesseract", "Tesseract"),
]

DEVICES = [("auto", "Auto"), ("cpu", "CPU"), ("cuda", "GPU (CUDA)")]


class SettingsPanel(QWidget):
    """Edits the application-wide :class:`Settings` and persists on change."""

    settings_changed = Signal()

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_providers_group())
        layout.addWidget(self._build_roles_group())
        layout.addWidget(self._build_pipeline_group())
        layout.addStretch(1)

        self._reload_providers()
        self._load_values()

    # ------------------------------------------------------------- providers
    def _build_providers_group(self) -> QGroupBox:
        group = QGroupBox("AI Providers")
        outer = QHBoxLayout(group)

        self.provider_list = QListWidget()
        outer.addWidget(self.provider_list, 1)

        buttons = QVBoxLayout()
        add = QPushButton("Add...")
        edit = QPushButton("Edit...")
        remove = QPushButton("Remove")
        add.clicked.connect(self._add_provider)
        edit.clicked.connect(self._edit_provider)
        remove.clicked.connect(self._remove_provider)
        buttons.addWidget(add)
        buttons.addWidget(edit)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        outer.addLayout(buttons)
        return group

    def _build_roles_group(self) -> QGroupBox:
        group = QGroupBox("Model Roles & Languages")
        form = QFormLayout(group)

        self.translation_combo = QComboBox()
        self.vision_combo = QComboBox()
        self.translation_combo.currentIndexChanged.connect(self._save)
        self.vision_combo.currentIndexChanged.connect(self._save)

        self.source_combo = QComboBox()
        self.target_combo = QComboBox()
        for code, label in LANGS:
            self.source_combo.addItem(label, code)
            self.target_combo.addItem(label, code)
        self.source_combo.currentIndexChanged.connect(self._save)
        self.target_combo.currentIndexChanged.connect(self._save)

        form.addRow("Translation model", self.translation_combo)
        form.addRow("Vision-OCR fallback", self.vision_combo)
        form.addRow("Default source language", self.source_combo)
        form.addRow("Default target language", self.target_combo)
        return group

    def _build_pipeline_group(self) -> QGroupBox:
        group = QGroupBox("Pipeline")
        form = QFormLayout(group)

        self.ocr_combo = QComboBox()
        for code, label in OCR_ENGINES:
            self.ocr_combo.addItem(label, code)
        self.ocr_combo.currentIndexChanged.connect(self._save)

        self.device_combo = QComboBox()
        for code, label in DEVICES:
            self.device_combo.addItem(label, code)
        self.device_combo.currentIndexChanged.connect(self._save)

        self.inpaint_check = QCheckBox("Use AI inpainting (LaMa) to erase text")
        self.inpaint_check.stateChanged.connect(self._save)

        form.addRow("OCR engine", self.ocr_combo)
        form.addRow("Compute device", self.device_combo)
        form.addRow("", self.inpaint_check)
        return group

    # --------------------------------------------------------------- actions
    def _add_provider(self) -> None:
        dialog = ProviderDialog(self)
        if dialog.exec():
            self.settings.providers.append(dialog.current_config())
            self._reload_providers()
            self._save()

    def _edit_provider(self) -> None:
        idx = self.provider_list.currentRow()
        if idx < 0 or idx >= len(self.settings.providers):
            return
        dialog = ProviderDialog(self, config=self.settings.providers[idx])
        if dialog.exec():
            self.settings.providers[idx] = dialog.current_config()
            self._reload_providers()
            self._save()

    def _remove_provider(self) -> None:
        idx = self.provider_list.currentRow()
        if idx < 0 or idx >= len(self.settings.providers):
            return
        name = self.settings.providers[idx].name
        if QMessageBox.question(self, "Remove provider",
                                f"Remove '{name}'?") != QMessageBox.Yes:
            return
        del self.settings.providers[idx]
        self._reload_providers()
        self._save()

    # ----------------------------------------------------------------- state
    def _reload_providers(self) -> None:
        self.provider_list.clear()
        names = []
        for p in self.settings.providers:
            self.provider_list.addItem(f"{p.name}  -  {p.model or p.kind}")
            names.append(p.name)

        self._refill_role_combo(self.translation_combo, names,
                                self.settings.translation_provider, allow_none=False)
        self._refill_role_combo(self.vision_combo, names,
                                self.settings.vision_provider, allow_none=True)

    @staticmethod
    def _refill_role_combo(combo: QComboBox, names: list[str],
                           current: str, allow_none: bool) -> None:
        combo.blockSignals(True)
        combo.clear()
        if allow_none:
            combo.addItem("(none)", "")
        for name in names:
            combo.addItem(name, name)
        idx = combo.findData(current)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _load_values(self) -> None:
        for combo, value in (
            (self.source_combo, self.settings.source_lang),
            (self.target_combo, self.settings.target_lang),
            (self.ocr_combo, self.settings.ocr_engine),
            (self.device_combo, self.settings.device),
        ):
            combo.blockSignals(True)
            i = combo.findData(value)
            if i >= 0:
                combo.setCurrentIndex(i)
            combo.blockSignals(False)
        self.inpaint_check.blockSignals(True)
        self.inpaint_check.setChecked(self.settings.use_inpainting)
        self.inpaint_check.blockSignals(False)

    def _save(self) -> None:
        self.settings.translation_provider = self.translation_combo.currentData() or ""
        self.settings.vision_provider = self.vision_combo.currentData() or ""
        self.settings.source_lang = self.source_combo.currentData()
        self.settings.target_lang = self.target_combo.currentData()
        self.settings.ocr_engine = self.ocr_combo.currentData()
        self.settings.device = self.device_combo.currentData()
        self.settings.use_inpainting = self.inpaint_check.isChecked()
        save_settings(self.settings)
        self.settings_changed.emit()

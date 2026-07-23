"""Dialog for adding/editing a single AI provider, with Test Connection."""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..config import ProviderConfig
from ..ai.registry import build_provider

# Human labels for the provider kinds, with sensible default endpoints/models.
KIND_LABELS = {
    "openai_compat": "OpenAI-compatible (OpenAI / LM Studio / Ollama / DeepSeek)",
    "gemini": "Google Gemini",
    "claude": "Anthropic Claude",
}

PRESETS = {
    "LM Studio (local)": ("openai_compat", "http://localhost:1234/v1", "", False),
    "Ollama (local)": ("openai_compat", "http://localhost:11434/v1", "llama3.1", False),
    "OpenAI / ChatGPT": ("openai_compat", "https://api.openai.com/v1", "gpt-4o", True),
    "DeepSeek": ("openai_compat", "https://api.deepseek.com/v1", "deepseek-chat", False),
    "Google Gemini": ("gemini", "", "gemini-1.5-flash", True),
    "Anthropic Claude": ("claude", "", "claude-3-5-sonnet-latest", True),
}


class _TestWorker(QObject):
    done = Signal(bool, str)

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__()
        self._cfg = cfg

    def run(self) -> None:
        try:
            provider = build_provider(self._cfg)
            msg = provider.test()
            self.done.emit(True, msg)
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


class ProviderDialog(QDialog):
    """Create or edit a :class:`ProviderConfig`."""

    def __init__(self, parent=None, config: ProviderConfig | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Provider")
        self.setMinimumWidth(480)
        self._config = config or ProviderConfig()
        self._thread: QThread | None = None
        self._worker: _TestWorker | None = None

        layout = QVBoxLayout(self)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset = QComboBox()
        self.preset.addItem("Custom")
        self.preset.addItems(PRESETS.keys())
        self.preset.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self.preset, 1)
        layout.addLayout(preset_row)

        form = QFormLayout()
        self.name = QLineEdit(self._config.name)
        self.kind = QComboBox()
        for key, label in KIND_LABELS.items():
            self.kind.addItem(label, key)
        self._select_kind(self._config.kind)
        self.kind.currentIndexChanged.connect(self._update_field_state)

        self.base_url = QLineEdit(self._config.base_url)
        self.base_url.setPlaceholderText("http://localhost:1234/v1 (OpenAI-compatible only)")
        self.api_key = QLineEdit(self._config.api_key)
        self.api_key.setEchoMode(QLineEdit.Password)
        self.model = QLineEdit(self._config.model)
        self.model.setPlaceholderText("e.g. gpt-4o, llama3.1, gemini-1.5-flash")
        self.supports_vision = QCheckBox("Model can read images (enables vision OCR)")
        self.supports_vision.setChecked(self._config.supports_vision)

        form.addRow("Name", self.name)
        form.addRow("Type", self.kind)
        form.addRow("Base URL", self.base_url)
        form.addRow("API key", self.api_key)
        form.addRow("Model", self.model)
        form.addRow("", self.supports_vision)
        layout.addLayout(form)

        test_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test)
        self.test_label = QLabel("")
        self.test_label.setWordWrap(True)
        self.test_label.setProperty("muted", True)
        test_row.addWidget(self.test_btn)
        test_row.addWidget(self.test_label, 1)
        layout.addLayout(test_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_field_state()

    # ------------------------------------------------------------- behavior
    def _select_kind(self, kind: str) -> None:
        idx = self.kind.findData(kind)
        if idx >= 0:
            self.kind.setCurrentIndex(idx)

    def _apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            return
        kind, base_url, model, vision = PRESETS[name]
        self._select_kind(kind)
        self.base_url.setText(base_url)
        self.model.setText(model)
        self.supports_vision.setChecked(vision)
        if not self.name.text() or self.name.text() == "New Provider":
            self.name.setText(name)
        self._update_field_state()

    def _update_field_state(self) -> None:
        kind = self.kind.currentData()
        self.base_url.setEnabled(kind == "openai_compat")

    def current_config(self) -> ProviderConfig:
        return ProviderConfig(
            name=self.name.text().strip() or "Provider",
            kind=self.kind.currentData(),
            base_url=self.base_url.text().strip(),
            api_key=self.api_key.text().strip(),
            model=self.model.text().strip(),
            supports_vision=self.supports_vision.isChecked(),
        )

    def _test(self) -> None:
        # Guard against overlapping tests; one running thread at a time.
        if self._thread is not None and self._thread.isRunning():
            return
        self.test_btn.setEnabled(False)
        self.test_label.setStyleSheet("")
        self.test_label.setText("Testing...")

        worker = _TestWorker(self.current_config())
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_test_done)
        worker.done.connect(thread.quit)
        # Keep strong references so neither object is garbage-collected while
        # the thread is running; clear them only after the thread has finished.
        thread.finished.connect(self._clear_test_thread)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_test_done(self, ok: bool, message: str) -> None:
        self.test_btn.setEnabled(True)
        color = "#4cd07d" if ok else "#ff5c5c"
        prefix = "OK - " if ok else "Failed - "
        self.test_label.setStyleSheet(f"color: {color};")
        self.test_label.setText(prefix + message)

    def _clear_test_thread(self) -> None:
        """Drop references once the worker thread's event loop has stopped."""
        self._worker = None
        self._thread = None

    def _shutdown_test(self) -> None:
        """Stop the test thread before the dialog is destroyed.

        The blocking network call runs directly in the worker, so ``quit`` (which
        only affects an event loop) may not return promptly; we wait briefly and
        terminate as a last resort to avoid 'QThread destroyed while running'.
        """
        thread = self._thread
        if thread is None:
            return
        try:
            if thread.isRunning():
                thread.quit()
                if not thread.wait(3000):
                    thread.terminate()
                    thread.wait(2000)
        except RuntimeError:
            # The underlying C++ object may already be gone; nothing to do.
            pass
        self._worker = None
        self._thread = None

    def done(self, result: int) -> None:  # noqa: N802 - Qt override
        # Ensure any in-flight Test Connection thread is stopped before close.
        self._shutdown_test()
        super().done(result)

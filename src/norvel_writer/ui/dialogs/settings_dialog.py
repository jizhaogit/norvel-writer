"""Application settings dialog."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)
from norvel_writer.config.defaults import KNOWN_VISION_MODELS

from norvel_writer.config.defaults import LANGUAGES, language_display


def _make_language_combo(current_code: str) -> QComboBox:
    """Build a QComboBox populated from the LANGUAGES registry."""
    combo = QComboBox()
    selected_index = 0
    for i, (code, (eng, native)) in enumerate(LANGUAGES.items()):
        label = f"{eng}  ({native})" if eng != native else eng
        combo.addItem(label, code)
        if code == current_code:
            selected_index = i
    combo.setCurrentIndex(selected_index)
    return combo


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Ollama ────────────────────────────────────────────────────────
        ollama_group = QGroupBox("Ollama")
        ollama_form = QFormLayout(ollama_group)
        self._ollama_url = QLineEdit()
        ollama_form.addRow("Base URL:", self._ollama_url)
        self._chat_model = QLineEdit()
        ollama_form.addRow("Chat model:", self._chat_model)
        self._embed_model = QLineEdit()
        ollama_form.addRow("Embedding model:", self._embed_model)

        self._vision_model = QComboBox()
        self._vision_model.setEditable(True)
        self._vision_model.addItem("(none — skip image analysis)", "")
        for m in KNOWN_VISION_MODELS:
            self._vision_model.addItem(m, m)
        vision_note = QLabel(
            "Vision model is used to describe imported images (maps, character art). "
            "Leave blank if you have not downloaded a vision model. "
            "Recommended: llava:7b or llama3.2-vision"
        )
        vision_note.setObjectName("subtitle")
        vision_note.setWordWrap(True)
        ollama_form.addRow("Vision model:", self._vision_model)
        ollama_form.addRow("", vision_note)
        layout.addWidget(ollama_group)

        # ── Language ──────────────────────────────────────────────────────
        lang_group = QGroupBox("Language")
        lang_form = QFormLayout(lang_group)

        # UI language
        self._ui_language = QComboBox()
        for code, (eng, native) in LANGUAGES.items():
            label = f"{eng}  ({native})" if eng != native else eng
            self._ui_language.addItem(label, code)
        ui_lang_note = QLabel("Changing the UI language requires restarting the app.")
        ui_lang_note.setObjectName("subtitle")
        ui_lang_note.setWordWrap(True)
        lang_form.addRow("UI language:", self._ui_language)
        lang_form.addRow("", ui_lang_note)

        # Default content language
        self._content_language = QComboBox()
        for code, (eng, native) in LANGUAGES.items():
            label = f"{eng}  ({native})" if eng != native else eng
            self._content_language.addItem(label, code)
        content_lang_note = QLabel(
            'Default language for AI-generated text. The "Write in:" dropdown in the '
            "Draft panel uses this unless a chapter's project overrides it."
        )
        content_lang_note.setObjectName("subtitle")
        content_lang_note.setWordWrap(True)
        lang_form.addRow("Default writing language:", self._content_language)
        lang_form.addRow("", content_lang_note)

        # Default project language
        self._project_language = QComboBox()
        for code, (eng, native) in LANGUAGES.items():
            label = f"{eng}  ({native})" if eng != native else eng
            self._project_language.addItem(label, code)
        proj_lang_note = QLabel(
            "Pre-filled language when you create a new project. "
            "You can still change it per-project."
        )
        proj_lang_note.setObjectName("subtitle")
        proj_lang_note.setWordWrap(True)
        lang_form.addRow("Default new-project language:", self._project_language)
        lang_form.addRow("", proj_lang_note)

        layout.addWidget(lang_group)

        # ── Interface ─────────────────────────────────────────────────────
        ui_group = QGroupBox("Interface")
        ui_form = QFormLayout(ui_group)
        self._theme = QComboBox()
        self._theme.addItems(["dark", "light"])
        ui_form.addRow("Theme:", self._theme)
        layout.addWidget(ui_group)

        # ── Buttons ───────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load(self) -> None:
        from norvel_writer.config.settings import get_config
        cfg = get_config()

        self._ollama_url.setText(cfg.ollama_base_url)
        self._chat_model.setText(cfg.default_chat_model)
        self._embed_model.setText(cfg.default_embed_model)
        # Vision model
        vision = cfg.vision_model or ""
        found = False
        for i in range(self._vision_model.count()):
            if self._vision_model.itemData(i) == vision:
                self._vision_model.setCurrentIndex(i)
                found = True
                break
        if not found and vision:
            self._vision_model.setEditText(vision)
        self._theme.setCurrentText(cfg.theme)

        self._set_combo_by_code(self._ui_language, cfg.ui_language)
        self._set_combo_by_code(self._content_language, cfg.default_content_language)
        self._set_combo_by_code(self._project_language, cfg.default_project_language)

    def _save(self) -> None:
        from norvel_writer.config.settings import get_config
        from norvel_writer.ui.theme import apply_theme
        from PySide6.QtWidgets import QApplication

        cfg = get_config()
        cfg.ollama_base_url = self._ollama_url.text().strip()
        cfg.default_chat_model = self._chat_model.text().strip()
        cfg.default_embed_model = self._embed_model.text().strip()
        vision = self._vision_model.currentData() or self._vision_model.currentText()
        cfg.vision_model = "" if vision == "(none — skip image analysis)" else vision.strip()
        cfg.theme = self._theme.currentText()
        cfg.ui_language = self._ui_language.currentData()
        cfg.default_content_language = self._content_language.currentData()
        cfg.default_project_language = self._project_language.currentData()
        cfg.save()

        # Apply theme immediately — no restart needed
        apply_theme(QApplication.instance(), cfg.theme)

        self.accept()

    @staticmethod
    def _set_combo_by_code(combo: QComboBox, code: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == code:
                combo.setCurrentIndex(i)
                return
        # Fallback: try matching "en" if unknown code
        for i in range(combo.count()):
            if combo.itemData(i) == "en":
                combo.setCurrentIndex(i)
                return

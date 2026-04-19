"""Application settings dialog."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from norvel_writer.config.defaults import KNOWN_VISION_MODELS
from norvel_writer.config.defaults import LANGUAGES, language_display


# ── Ollama advanced settings recommended defaults ─────────────────────────

_DEFAULTS = {
    "temperature":    0.85,
    "top_p":          0.90,
    "min_p":          0.03,
    "repeat_penalty": 1.08,
}

_PRESETS = {
    "Stable":   {"temperature": 0.60, "top_p": 0.85, "min_p": 0.05, "repeat_penalty": 1.15},
    "Balanced": _DEFAULTS,
    "Creative": {"temperature": 1.10, "top_p": 0.95, "min_p": 0.02, "repeat_penalty": 1.05},
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_language_combo(current_code: str) -> QComboBox:
    combo = QComboBox()
    selected_index = 0
    for i, (code, (eng, native)) in enumerate(LANGUAGES.items()):
        label = f"{eng}  ({native})" if eng != native else eng
        combo.addItem(label, code)
        if code == current_code:
            selected_index = i
    combo.setCurrentIndex(selected_index)
    return combo


def _make_slider_widget(
    min_val: int,
    max_val: int,
    initial: int,
    scale: int = 100,
    decimals: int = 2,
) -> tuple[QWidget, QSlider, QLabel]:
    """Return (container, slider, value_label). Stored value = slider.value() / scale."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(min_val, max_val)
    slider.setValue(initial)

    fmt = f"{{:.{decimals}f}}"
    value_label = QLabel(fmt.format(initial / scale))
    value_label.setMinimumWidth(44)
    value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    slider.valueChanged.connect(lambda v: value_label.setText(fmt.format(v / scale)))

    row.addWidget(slider, stretch=1)
    row.addWidget(value_label)
    return container, slider, value_label


def _subtitle(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("subtitle")
    lbl.setWordWrap(True)
    return lbl


# ── Main dialog ────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
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
        vision_note = _subtitle(
            "Vision model is used to describe imported images (maps, character art). "
            "Leave blank if you have not downloaded a vision model. "
            "Recommended: llava:7b or llama3.2-vision"
        )
        ollama_form.addRow("Vision model:", self._vision_model)
        ollama_form.addRow("", vision_note)
        layout.addWidget(ollama_group)

        # ── Ollama Advanced Generation Settings ───────────────────────────
        self._adv_group = QGroupBox("Ollama Advanced Generation Settings")
        adv_form = QFormLayout(self._adv_group)

        # Temperature
        _temp_w, self._slider_temp, self._lbl_temp = _make_slider_widget(0, 200, 85)
        adv_form.addRow("Temperature:", _temp_w)
        adv_form.addRow("", _subtitle("Higher = more creative, lower = more stable"))

        # Top-p
        _top_p_w, self._slider_top_p, self._lbl_top_p = _make_slider_widget(0, 100, 90)
        adv_form.addRow("Top-p:", _top_p_w)
        adv_form.addRow("", _subtitle("Higher = more variety, lower = more conservative"))

        # Min-p
        _min_p_w, self._slider_min_p, self._lbl_min_p = _make_slider_widget(0, 100, 3)
        adv_form.addRow("Min-p:", _min_p_w)
        adv_form.addRow("", _subtitle("Filters out very unlikely token candidates"))

        # Repeat penalty
        _rep_w, self._slider_repeat, self._lbl_repeat = _make_slider_widget(80, 200, 108)
        adv_form.addRow("Repeat penalty:", _rep_w)
        adv_form.addRow("", _subtitle("Reduces repetitive wording and looping"))

        # Seed
        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("Empty = random each run")
        self._seed_edit.setMaximumWidth(180)
        adv_form.addRow("Seed:", self._seed_edit)
        adv_form.addRow("", _subtitle("Same seed can help reproduce similar outputs"))

        # Num context (num_ctx)
        self._num_ctx_edit = QLineEdit()
        self._num_ctx_edit.setPlaceholderText("Empty = use .env default")
        self._num_ctx_edit.setMaximumWidth(180)
        adv_form.addRow("Context window (num_ctx):", self._num_ctx_edit)
        adv_form.addRow("", _subtitle("Token context window; must not exceed your model's maximum"))

        # Num predict (num_predict)
        self._num_predict_edit = QLineEdit()
        self._num_predict_edit.setPlaceholderText("Empty = use .env default")
        self._num_predict_edit.setMaximumWidth(180)
        adv_form.addRow("Max output tokens (num_predict):", self._num_predict_edit)
        adv_form.addRow("", _subtitle("Maximum tokens generated per call (~750 words per 1000 tokens)"))

        # Preset + reset row
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 4, 0, 0)
        preset_layout.setSpacing(6)
        for name, vals in _PRESETS.items():
            btn = QPushButton(name)
            btn.setToolTip(
                f"temperature={vals['temperature']:.2f}  "
                f"top_p={vals['top_p']:.2f}  "
                f"min_p={vals['min_p']:.2f}  "
                f"repeat_penalty={vals['repeat_penalty']:.2f}"
            )
            _vals = dict(vals)
            btn.clicked.connect(lambda _checked=False, v=_vals: self._apply_preset(v))
            preset_layout.addWidget(btn)
        preset_layout.addStretch()

        reset_btn = QPushButton("Reset to recommended defaults")
        reset_btn.clicked.connect(lambda: self._apply_preset(_DEFAULTS))
        preset_layout.addWidget(reset_btn)
        adv_form.addRow("Presets:", preset_row)

        layout.addWidget(self._adv_group)

        # ── Language ──────────────────────────────────────────────────────
        lang_group = QGroupBox("Language")
        lang_form = QFormLayout(lang_group)

        self._ui_language = QComboBox()
        for code, (eng, native) in LANGUAGES.items():
            label = f"{eng}  ({native})" if eng != native else eng
            self._ui_language.addItem(label, code)
        lang_form.addRow("UI language:", self._ui_language)
        lang_form.addRow("", _subtitle("Changing the UI language requires restarting the app."))

        self._content_language = QComboBox()
        for code, (eng, native) in LANGUAGES.items():
            label = f"{eng}  ({native})" if eng != native else eng
            self._content_language.addItem(label, code)
        lang_form.addRow("Default writing language:", self._content_language)
        lang_form.addRow("", _subtitle(
            'Default language for AI-generated text. The "Write in:" dropdown in the '
            "Draft panel uses this unless a chapter's project overrides it."
        ))

        self._project_language = QComboBox()
        for code, (eng, native) in LANGUAGES.items():
            label = f"{eng}  ({native})" if eng != native else eng
            self._project_language.addItem(label, code)
        lang_form.addRow("Default new-project language:", self._project_language)
        lang_form.addRow("", _subtitle(
            "Pre-filled language when you create a new project. "
            "You can still change it per-project."
        ))

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

    def _is_ollama_provider(self) -> bool:
        import os
        return os.environ.get("LLM_PROVIDER", "ollama").lower() == "ollama"

    def _load(self) -> None:
        from norvel_writer.config.settings import get_config
        cfg = get_config()

        self._ollama_url.setText(cfg.ollama_base_url)
        self._chat_model.setText(cfg.default_chat_model)
        self._embed_model.setText(cfg.default_embed_model)

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

        # Advanced Ollama generation settings
        temp = cfg.ollama_gen_temperature if cfg.ollama_gen_temperature is not None else _DEFAULTS["temperature"]
        top_p = cfg.ollama_gen_top_p if cfg.ollama_gen_top_p is not None else _DEFAULTS["top_p"]
        min_p = cfg.ollama_gen_min_p if cfg.ollama_gen_min_p is not None else _DEFAULTS["min_p"]
        rep = cfg.ollama_gen_repeat_penalty if cfg.ollama_gen_repeat_penalty is not None else _DEFAULTS["repeat_penalty"]

        self._slider_temp.setValue(round(temp * 100))
        self._slider_top_p.setValue(round(top_p * 100))
        self._slider_min_p.setValue(round(min_p * 100))
        self._slider_repeat.setValue(round(rep * 100))

        self._seed_edit.setText("" if cfg.ollama_gen_seed is None else str(cfg.ollama_gen_seed))
        self._num_ctx_edit.setText("" if cfg.ollama_gen_num_ctx is None else str(cfg.ollama_gen_num_ctx))
        self._num_predict_edit.setText("" if cfg.ollama_gen_num_predict is None else str(cfg.ollama_gen_num_predict))

        # Show advanced panel only when Ollama is the active provider
        self._adv_group.setVisible(self._is_ollama_provider())

    def _apply_preset(self, vals: dict) -> None:
        self._slider_temp.setValue(round(vals["temperature"] * 100))
        self._slider_top_p.setValue(round(vals["top_p"] * 100))
        self._slider_min_p.setValue(round(vals["min_p"] * 100))
        self._slider_repeat.setValue(round(vals["repeat_penalty"] * 100))

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

        # Advanced Ollama settings — validate and clamp
        cfg.ollama_gen_temperature = max(0.0, min(2.0, self._slider_temp.value() / 100))
        cfg.ollama_gen_top_p = max(0.0, min(1.0, self._slider_top_p.value() / 100))
        cfg.ollama_gen_min_p = max(0.0, min(1.0, self._slider_min_p.value() / 100))
        cfg.ollama_gen_repeat_penalty = max(0.8, min(2.0, self._slider_repeat.value() / 100))

        seed_text = self._seed_edit.text().strip()
        if seed_text:
            try:
                cfg.ollama_gen_seed = int(seed_text)
            except ValueError:
                cfg.ollama_gen_seed = None
        else:
            cfg.ollama_gen_seed = None

        ctx_text = self._num_ctx_edit.text().strip()
        if ctx_text:
            try:
                cfg.ollama_gen_num_ctx = max(512, int(ctx_text))
            except ValueError:
                cfg.ollama_gen_num_ctx = None
        else:
            cfg.ollama_gen_num_ctx = None

        predict_text = self._num_predict_edit.text().strip()
        if predict_text:
            try:
                cfg.ollama_gen_num_predict = max(64, int(predict_text))
            except ValueError:
                cfg.ollama_gen_num_predict = None
        else:
            cfg.ollama_gen_num_predict = None

        cfg.save()

        # Re-initialize the LLM so new sampling settings take effect immediately
        if self._is_ollama_provider():
            try:
                from norvel_writer.llm.langchain_bridge import reset_singletons
                reset_singletons()
            except Exception:
                pass

        apply_theme(QApplication.instance(), cfg.theme)
        self.accept()

    @staticmethod
    def _set_combo_by_code(combo: QComboBox, code: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == code:
                combo.setCurrentIndex(i)
                return
        for i in range(combo.count()):
            if combo.itemData(i) == "en":
                combo.setCurrentIndex(i)
                return

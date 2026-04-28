"""Application settings dialog."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from norvel_writer.config.defaults import KNOWN_VISION_MODELS
from norvel_writer.config.defaults import LANGUAGES, language_display


# ── Step ↔ float conversions (1–10 integer steps → real parameter values) ────
#
#   Temperature   step × 0.20        →  0.20 … 2.00
#   Top-p         step × 0.10        →  0.10 … 1.00
#   Min-p         step × 0.01        →  0.01 … 0.10
#   Repeat        0.90 + step × 0.10 →  1.00 … 1.90

def _temp_to_step(v: float) -> int:
    return max(1, min(10, round(v / 0.2)))

def _step_to_temp(s: int) -> float:
    return round(s * 0.2, 2)

def _top_p_to_step(v: float) -> int:
    return max(1, min(10, round(v / 0.1)))

def _step_to_top_p(s: int) -> float:
    return round(s * 0.1, 2)

def _min_p_to_step(v: float) -> int:
    return max(1, min(10, round(v / 0.01)))

def _step_to_min_p(s: int) -> float:
    return round(s * 0.01, 3)

def _repeat_to_step(v: float) -> int:
    return max(1, min(10, round((v - 0.9) / 0.1)))

def _step_to_repeat(s: int) -> float:
    return round(0.9 + s * 0.1, 2)


# ── Defaults & presets (in 1–10 integer steps) ────────────────────────────────
#
#   step   temp   top_p   min_p   repeat
#     1    0.20   0.10    0.01    1.00
#     2    0.40   0.20    0.02    1.10
#     3    0.60   0.30    0.03    1.20
#     4    0.80   0.40    0.04    1.30
#     5    1.00   0.50    0.05    1.40
#     6    1.20   0.60    0.06    1.50
#     7    1.40   0.70    0.07    1.60
#     8    1.60   0.80    0.08    1.70
#     9    1.80   0.90    0.09    1.80
#    10    2.00   1.00    0.10    1.90

_DEFAULTS = {
    "temperature":    4,   # → 0.80
    "top_p":          9,   # → 0.90
    "min_p":          3,   # → 0.03
    "repeat_penalty": 2,   # → 1.10
}

_PRESETS = {
    "Stable":   {"temperature": 3,  "top_p": 9,  "min_p": 5, "repeat_penalty": 3},
    "Balanced": _DEFAULTS,
    "Creative": {"temperature": 6,  "top_p": 10, "min_p": 2, "repeat_penalty": 1},
}

_PRESET_TIPS = {
    "Stable":   "temp=0.60  top_p=0.90  min_p=0.05  repeat=1.20 — clean continuity",
    "Balanced": "temp=0.80  top_p=0.90  min_p=0.03  repeat=1.10 — normal use",
    "Creative": "temp=1.20  top_p=1.00  min_p=0.02  repeat=1.00 — more surprising output",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_language_combo(current_code: str) -> "QComboBox":
    """Return a QComboBox pre-populated with all supported languages, with
    the item matching *current_code* pre-selected."""
    combo = QComboBox()
    selected_index = 0
    for i, (code, (eng, native)) in enumerate(LANGUAGES.items()):
        label = f"{eng}  ({native})" if eng != native else eng
        combo.addItem(label, code)
        if code == current_code:
            selected_index = i
    combo.setCurrentIndex(selected_index)
    return combo


def _make_step_slider(initial: int = 5) -> tuple[QWidget, QSlider, QLabel]:
    """Return (container, slider, value_label) for a 1–10 integer step slider."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(1, 10)
    slider.setValue(initial)
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    slider.setTickInterval(1)

    value_label = QLabel(str(initial))
    value_label.setMinimumWidth(24)
    value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    slider.valueChanged.connect(lambda v: value_label.setText(str(v)))

    row.addWidget(slider, stretch=1)
    row.addWidget(value_label)
    return container, slider, value_label


def _subtitle(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("subtitle")
    lbl.setWordWrap(True)
    return lbl


# ── Main dialog ────────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        self.setMinimumHeight(420)
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Scrollable content area ────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 4)

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
        self._adv_group = QGroupBox("Ollama Advanced Generation Settings  (1 = low · 10 = high)")
        adv_form = QFormLayout(self._adv_group)

        _temp_w, self._slider_temp, _ = _make_step_slider(_DEFAULTS["temperature"])
        adv_form.addRow("Temperature:", _temp_w)
        adv_form.addRow("", _subtitle(
            "1 = very stable (0.2)  ·  5 = neutral (1.0)  ·  10 = very creative (2.0)"
        ))

        _top_p_w, self._slider_top_p, _ = _make_step_slider(_DEFAULTS["top_p"])
        adv_form.addRow("Top-p:", _top_p_w)
        adv_form.addRow("", _subtitle(
            "1 = conservative (0.1)  ·  5 = moderate (0.5)  ·  10 = full variety (1.0)"
        ))

        _min_p_w, self._slider_min_p, _ = _make_step_slider(_DEFAULTS["min_p"])
        adv_form.addRow("Min-p:", _min_p_w)
        adv_form.addRow("", _subtitle(
            "1 = permissive (0.01)  ·  5 = moderate (0.05)  ·  10 = strict (0.10) — filters unlikely tokens"
        ))

        _rep_w, self._slider_repeat, _ = _make_step_slider(_DEFAULTS["repeat_penalty"])
        adv_form.addRow("Repeat penalty:", _rep_w)
        adv_form.addRow("", _subtitle(
            "1 = minimal (1.0)  ·  5 = moderate (1.4)  ·  10 = strong (1.9) — reduces looping"
        ))

        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("Empty = random each run")
        self._seed_edit.setMaximumWidth(180)
        adv_form.addRow("Seed:", self._seed_edit)
        adv_form.addRow("", _subtitle("Same seed can help reproduce similar outputs"))

        self._num_ctx_edit = QLineEdit()
        self._num_ctx_edit.setPlaceholderText("Empty = use .env default")
        self._num_ctx_edit.setMaximumWidth(180)
        adv_form.addRow("Context window (num_ctx):", self._num_ctx_edit)
        adv_form.addRow("", _subtitle("Token context window; must not exceed your model's maximum"))

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
            btn.setToolTip(_PRESET_TIPS.get(name, ""))
            _vals = dict(vals)
            btn.clicked.connect(lambda _checked=False, v=_vals: self._apply_preset(v))
            preset_layout.addWidget(btn)
        preset_layout.addStretch()
        reset_btn = QPushButton("Reset to defaults")
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

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

        # ── Buttons — always visible, outside the scroll area ─────────────
        btn_wrapper = QWidget()
        bw = QVBoxLayout(btn_wrapper)
        bw.setContentsMargins(12, 4, 12, 8)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        bw.addWidget(buttons)
        outer.addWidget(btn_wrapper)

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

        # Convert stored floats → nearest 1–10 step (or use default step if not yet saved)
        temp_f = cfg.ollama_gen_temperature    if cfg.ollama_gen_temperature    is not None \
                 else _step_to_temp(_DEFAULTS["temperature"])
        top_f  = cfg.ollama_gen_top_p          if cfg.ollama_gen_top_p          is not None \
                 else _step_to_top_p(_DEFAULTS["top_p"])
        min_f  = cfg.ollama_gen_min_p          if cfg.ollama_gen_min_p          is not None \
                 else _step_to_min_p(_DEFAULTS["min_p"])
        rep_f  = cfg.ollama_gen_repeat_penalty if cfg.ollama_gen_repeat_penalty is not None \
                 else _step_to_repeat(_DEFAULTS["repeat_penalty"])

        self._slider_temp.setValue(_temp_to_step(temp_f))
        self._slider_top_p.setValue(_top_p_to_step(top_f))
        self._slider_min_p.setValue(_min_p_to_step(min_f))
        self._slider_repeat.setValue(_repeat_to_step(rep_f))

        self._seed_edit.setText("" if cfg.ollama_gen_seed is None else str(cfg.ollama_gen_seed))
        self._num_ctx_edit.setText("" if cfg.ollama_gen_num_ctx is None else str(cfg.ollama_gen_num_ctx))
        self._num_predict_edit.setText(
            "" if cfg.ollama_gen_num_predict is None else str(cfg.ollama_gen_num_predict)
        )

        # Show advanced panel only when Ollama is the active provider
        self._adv_group.setVisible(self._is_ollama_provider())

    def _apply_preset(self, vals: dict) -> None:
        self._slider_temp.setValue(vals["temperature"])
        self._slider_top_p.setValue(vals["top_p"])
        self._slider_min_p.setValue(vals["min_p"])
        self._slider_repeat.setValue(vals["repeat_penalty"])

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

        # Convert 1–10 integer steps → real float values for Ollama
        cfg.ollama_gen_temperature    = _step_to_temp(self._slider_temp.value())
        cfg.ollama_gen_top_p          = _step_to_top_p(self._slider_top_p.value())
        cfg.ollama_gen_min_p          = _step_to_min_p(self._slider_min_p.value())
        cfg.ollama_gen_repeat_penalty = _step_to_repeat(self._slider_repeat.value())

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

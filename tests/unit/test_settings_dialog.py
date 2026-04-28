"""
Unit tests for settings_dialog.py.

Covers:
  - All 8 step↔float conversion functions (pure, no Qt required)
  - _DEFAULTS and _PRESETS constant validation
  - _make_step_slider widget factory
  - _subtitle label factory
  - _make_language_combo helper
  - SettingsDialog._is_ollama_provider
  - SettingsDialog._apply_preset
  - SettingsDialog._set_combo_by_code
  - SettingsDialog._load  (slider init, field init, panel visibility)
  - SettingsDialog._save  (float conversion, validation, side-effects)
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Pure functions and constants (no Qt instance needed) ──────────────────────
from norvel_writer.ui.dialogs.settings_dialog import (
    _temp_to_step, _step_to_temp,
    _top_p_to_step, _step_to_top_p,
    _min_p_to_step, _step_to_min_p,
    _repeat_to_step, _step_to_repeat,
    _DEFAULTS, _PRESETS, _PRESET_TIPS,
)

# ── Qt availability guard ─────────────────────────────────────────────────────
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QSlider, QWidget
    from norvel_writer.ui.dialogs.settings_dialog import (
        _make_step_slider, _subtitle, _make_language_combo, SettingsDialog,
    )
    _HAS_QT = True
except Exception:
    _HAS_QT = False

requires_qt = pytest.mark.skipif(not _HAS_QT, reason="PySide6 not available")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Qt fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def qapp():
    """Single QApplication for the whole test session (offscreen rendering)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _mock_cfg(**overrides):
    """Return a MagicMock shaped like AppConfig for dialog injection."""
    cfg = MagicMock()
    cfg.ollama_base_url            = "http://127.0.0.1:11434"
    cfg.default_chat_model         = "gemma3:4b"
    cfg.default_embed_model        = "bge-m3"
    cfg.vision_model               = ""
    cfg.theme                      = "dark"
    cfg.ui_language                = "en"
    cfg.default_content_language   = "en"
    cfg.default_project_language   = "en"
    cfg.ollama_gen_temperature     = None
    cfg.ollama_gen_top_p           = None
    cfg.ollama_gen_min_p           = None
    cfg.ollama_gen_repeat_penalty  = None
    cfg.ollama_gen_seed            = None
    cfg.ollama_gen_num_ctx         = None
    cfg.ollama_gen_num_predict     = None
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def dialog(qapp):
    """SettingsDialog loaded with a blank mock config, Ollama provider active."""
    cfg = _mock_cfg()
    with patch("norvel_writer.config.settings.get_config", return_value=cfg), \
         patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=False):
        dlg = SettingsDialog()
        yield dlg
        dlg.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1  Temperature: step × 0.20  →  0.20 … 2.00
# ═══════════════════════════════════════════════════════════════════════════════

class TestStepToTemp:
    @pytest.mark.parametrize("step,expected", [
        (1,  0.20), (2,  0.40), (3,  0.60), (4,  0.80), (5,  1.00),
        (6,  1.20), (7,  1.40), (8,  1.60), (9,  1.80), (10, 2.00),
    ])
    def test_full_table(self, step, expected):
        assert _step_to_temp(step) == pytest.approx(expected, abs=1e-9)

    def test_returns_float(self):
        assert isinstance(_step_to_temp(5), float)

    def test_two_decimal_precision(self):
        for s in range(1, 11):
            v = _step_to_temp(s)
            assert round(v, 2) == v, f"step {s} → {v} has more than 2 decimal places"


class TestTempToStep:
    @pytest.mark.parametrize("step", range(1, 11))
    def test_roundtrip(self, step):
        """step → float → step must be lossless for every valid step."""
        assert _temp_to_step(_step_to_temp(step)) == step

    @pytest.mark.parametrize("v,expected", [
        (0.20, 1), (0.40, 2), (0.60, 3), (0.80, 4),
        (1.00, 5), (1.20, 6), (1.40, 7), (1.60, 8),
        (1.80, 9), (2.00, 10),
    ])
    def test_exact_grid_values(self, v, expected):
        assert _temp_to_step(v) == expected

    def test_old_default_085_maps_to_4(self):
        # Previous float default 0.85 → 0.85/0.2 = 4.25 → rounds to 4
        assert _temp_to_step(0.85) == 4

    def test_clamp_below_minimum(self):
        assert _temp_to_step(0.0)  == 1
        assert _temp_to_step(-1.0) == 1
        # 0.09/0.2=0.45 → round=0 → clamp to 1
        assert _temp_to_step(0.09) == 1

    def test_clamp_above_maximum(self):
        assert _temp_to_step(2.01) == 10
        assert _temp_to_step(5.0)  == 10

    def test_returns_int(self):
        assert isinstance(_temp_to_step(1.0), int)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2  Top-p: step × 0.10  →  0.10 … 1.00
# ═══════════════════════════════════════════════════════════════════════════════

class TestStepToTopP:
    @pytest.mark.parametrize("step,expected", [
        (1,  0.10), (2,  0.20), (3,  0.30), (4,  0.40), (5,  0.50),
        (6,  0.60), (7,  0.70), (8,  0.80), (9,  0.90), (10, 1.00),
    ])
    def test_full_table(self, step, expected):
        assert _step_to_top_p(step) == pytest.approx(expected, abs=1e-9)

    def test_returns_float(self):
        assert isinstance(_step_to_top_p(5), float)

    def test_two_decimal_precision(self):
        for s in range(1, 11):
            v = _step_to_top_p(s)
            assert round(v, 2) == v


class TestTopPToStep:
    @pytest.mark.parametrize("step", range(1, 11))
    def test_roundtrip(self, step):
        assert _top_p_to_step(_step_to_top_p(step)) == step

    @pytest.mark.parametrize("v,expected", [
        (0.10, 1), (0.50, 5), (0.90, 9), (1.00, 10),
    ])
    def test_exact_grid_values(self, v, expected):
        assert _top_p_to_step(v) == expected

    def test_old_default_090_maps_to_9(self):
        assert _top_p_to_step(0.90) == 9

    def test_clamp_below_minimum(self):
        # round(0.0/0.1) = 0 → clamp to 1
        assert _top_p_to_step(0.0)  == 1
        assert _top_p_to_step(-0.5) == 1

    def test_clamp_above_maximum(self):
        assert _top_p_to_step(1.05) == 10
        assert _top_p_to_step(2.0)  == 10

    def test_returns_int(self):
        assert isinstance(_top_p_to_step(0.5), int)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3  Min-p: step × 0.01  →  0.010 … 0.100
# ═══════════════════════════════════════════════════════════════════════════════

class TestStepToMinP:
    @pytest.mark.parametrize("step,expected", [
        (1,  0.010), (2,  0.020), (3,  0.030), (4,  0.040), (5,  0.050),
        (6,  0.060), (7,  0.070), (8,  0.080), (9,  0.090), (10, 0.100),
    ])
    def test_full_table(self, step, expected):
        assert _step_to_min_p(step) == pytest.approx(expected, abs=1e-9)

    def test_returns_float(self):
        assert isinstance(_step_to_min_p(3), float)

    def test_three_decimal_precision(self):
        for s in range(1, 11):
            v = _step_to_min_p(s)
            assert round(v, 3) == v


class TestMinPToStep:
    @pytest.mark.parametrize("step", range(1, 11))
    def test_roundtrip(self, step):
        assert _min_p_to_step(_step_to_min_p(step)) == step

    @pytest.mark.parametrize("v,expected", [
        (0.01, 1), (0.03, 3), (0.05, 5), (0.07, 7), (0.10, 10),
    ])
    def test_exact_grid_values(self, v, expected):
        assert _min_p_to_step(v) == expected

    def test_old_default_003_maps_to_3(self):
        assert _min_p_to_step(0.03) == 3

    def test_old_stable_preset_005_maps_to_5(self):
        assert _min_p_to_step(0.05) == 5

    def test_old_creative_preset_002_maps_to_2(self):
        assert _min_p_to_step(0.02) == 2

    def test_clamp_below_minimum(self):
        # round(0.0/0.01) = 0 → clamp to 1
        assert _min_p_to_step(0.0)   == 1
        assert _min_p_to_step(-0.01) == 1

    def test_clamp_above_maximum(self):
        assert _min_p_to_step(0.11) == 10
        assert _min_p_to_step(1.0)  == 10

    def test_returns_int(self):
        assert isinstance(_min_p_to_step(0.03), int)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4  Repeat penalty: 0.90 + step × 0.10  →  1.00 … 1.90
# ═══════════════════════════════════════════════════════════════════════════════

class TestStepToRepeat:
    @pytest.mark.parametrize("step,expected", [
        (1,  1.00), (2,  1.10), (3,  1.20), (4,  1.30), (5,  1.40),
        (6,  1.50), (7,  1.60), (8,  1.70), (9,  1.80), (10, 1.90),
    ])
    def test_full_table(self, step, expected):
        assert _step_to_repeat(step) == pytest.approx(expected, abs=1e-9)

    def test_returns_float(self):
        assert isinstance(_step_to_repeat(5), float)

    def test_step1_is_minimum_penalty(self):
        assert _step_to_repeat(1) == pytest.approx(1.0)

    def test_step10_is_maximum_penalty(self):
        assert _step_to_repeat(10) == pytest.approx(1.9)

    def test_two_decimal_precision(self):
        for s in range(1, 11):
            v = _step_to_repeat(s)
            assert round(v, 2) == v


class TestRepeatToStep:
    @pytest.mark.parametrize("step", range(1, 11))
    def test_roundtrip(self, step):
        assert _repeat_to_step(_step_to_repeat(step)) == step

    @pytest.mark.parametrize("v,expected", [
        (1.00, 1), (1.10, 2), (1.20, 3), (1.40, 5),
        (1.60, 7), (1.80, 9), (1.90, 10),
    ])
    def test_exact_grid_values(self, v, expected):
        assert _repeat_to_step(v) == expected

    def test_old_default_108_maps_to_2(self):
        # 1.08 was previous float default: (1.08-0.9)/0.1=1.8 → rounds to 2
        assert _repeat_to_step(1.08) == 2

    def test_clamp_below_minimum(self):
        # (0.9-0.9)/0.1 = 0 → clamp to 1
        assert _repeat_to_step(0.9) == 1
        assert _repeat_to_step(0.5) == 1

    def test_clamp_above_maximum(self):
        # (2.0-0.9)/0.1 = 11 → clamp to 10
        assert _repeat_to_step(2.0) == 10
        assert _repeat_to_step(5.0) == 10

    def test_returns_int(self):
        assert isinstance(_repeat_to_step(1.4), int)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5  _DEFAULTS and _PRESETS constant validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDefaults:
    KEYS = {"temperature", "top_p", "min_p", "repeat_penalty"}

    def test_has_all_keys(self):
        assert set(_DEFAULTS.keys()) == self.KEYS

    @pytest.mark.parametrize("key", ["temperature", "top_p", "min_p", "repeat_penalty"])
    def test_values_are_integers(self, key):
        assert isinstance(_DEFAULTS[key], int)

    @pytest.mark.parametrize("key", ["temperature", "top_p", "min_p", "repeat_penalty"])
    def test_values_in_1_to_10(self, key):
        assert 1 <= _DEFAULTS[key] <= 10

    def test_balanced_preset_is_same_object_as_defaults(self):
        assert _PRESETS["Balanced"] is _DEFAULTS


class TestPresets:
    PRESET_NAMES = {"Stable", "Balanced", "Creative"}
    PARAM_KEYS   = {"temperature", "top_p", "min_p", "repeat_penalty"}

    def test_has_exactly_three_presets(self):
        assert set(_PRESETS.keys()) == self.PRESET_NAMES

    @pytest.mark.parametrize("preset", ["Stable", "Balanced", "Creative"])
    def test_preset_has_all_param_keys(self, preset):
        assert set(_PRESETS[preset].keys()) == self.PARAM_KEYS

    @pytest.mark.parametrize("preset,key", [
        (p, k)
        for p in ["Stable", "Balanced", "Creative"]
        for k in ["temperature", "top_p", "min_p", "repeat_penalty"]
    ])
    def test_all_values_are_integers(self, preset, key):
        assert isinstance(_PRESETS[preset][key], int)

    @pytest.mark.parametrize("preset,key", [
        (p, k)
        for p in ["Stable", "Balanced", "Creative"]
        for k in ["temperature", "top_p", "min_p", "repeat_penalty"]
    ])
    def test_all_values_in_1_to_10(self, preset, key):
        v = _PRESETS[preset][key]
        assert 1 <= v <= 10, f"{preset}[{key}]={v} out of range"

    def test_stable_cooler_than_creative(self):
        assert _PRESETS["Stable"]["temperature"] < _PRESETS["Creative"]["temperature"]

    def test_stable_stricter_repeat_than_creative(self):
        assert _PRESETS["Stable"]["repeat_penalty"] > _PRESETS["Creative"]["repeat_penalty"]

    def test_tips_has_all_preset_names(self):
        assert set(_PRESET_TIPS.keys()) == self.PRESET_NAMES

    @pytest.mark.parametrize("preset", ["Stable", "Balanced", "Creative"])
    def test_tips_are_non_empty_strings(self, preset):
        assert isinstance(_PRESET_TIPS[preset], str)
        assert len(_PRESET_TIPS[preset]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6  _make_step_slider
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qt
class TestMakeStepSlider:
    # NOTE: always unpack *all three* return values so the container QWidget
    # stays referenced (and alive) for the duration of each test.  Discarding
    # the container with _ causes immediate GC which deletes child widgets.

    def test_returns_three_tuple(self, qapp):
        result = _make_step_slider(5)
        assert len(result) == 3

    def test_container_is_qwidget(self, qapp):
        container, slider, label = _make_step_slider(5)
        assert isinstance(container, QWidget)

    def test_slider_is_qslider(self, qapp):
        container, slider, label = _make_step_slider(5)
        assert isinstance(slider, QSlider)

    def test_label_is_qlabel(self, qapp):
        container, slider, label = _make_step_slider(5)
        assert isinstance(label, QLabel)

    def test_slider_minimum_is_1(self, qapp):
        container, slider, label = _make_step_slider(5)
        assert slider.minimum() == 1

    def test_slider_maximum_is_10(self, qapp):
        container, slider, label = _make_step_slider(5)
        assert slider.maximum() == 10

    @pytest.mark.parametrize("initial", [1, 3, 5, 7, 10])
    def test_initial_value_set_on_slider(self, qapp, initial):
        container, slider, label = _make_step_slider(initial)
        assert slider.value() == initial

    @pytest.mark.parametrize("initial", [1, 3, 5, 7, 10])
    def test_label_shows_initial_value(self, qapp, initial):
        container, slider, label = _make_step_slider(initial)
        assert label.text() == str(initial)

    def test_label_updates_when_slider_moves(self, qapp):
        container, slider, label = _make_step_slider(3)
        slider.setValue(8)
        assert label.text() == "8"

    def test_label_updates_across_full_range(self, qapp):
        container, slider, label = _make_step_slider(5)
        for v in range(1, 11):
            slider.setValue(v)
            assert label.text() == str(v)

    def test_tick_position_is_below(self, qapp):
        container, slider, label = _make_step_slider(5)
        assert slider.tickPosition() == QSlider.TickPosition.TicksBelow

    def test_tick_interval_is_1(self, qapp):
        container, slider, label = _make_step_slider(5)
        assert slider.tickInterval() == 1

    def test_orientation_is_horizontal(self, qapp):
        container, slider, label = _make_step_slider(5)
        assert slider.orientation() == Qt.Orientation.Horizontal


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7  _subtitle
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qt
class TestSubtitle:
    def test_returns_qlabel(self, qapp):
        assert isinstance(_subtitle("text"), QLabel)

    def test_text_is_set(self, qapp):
        lbl = _subtitle("Higher = more creative")
        assert lbl.text() == "Higher = more creative"

    def test_object_name_is_subtitle(self, qapp):
        lbl = _subtitle("anything")
        assert lbl.objectName() == "subtitle"

    def test_word_wrap_is_enabled(self, qapp):
        lbl = _subtitle("any text")
        assert lbl.wordWrap() is True

    def test_empty_string(self, qapp):
        lbl = _subtitle("")
        assert lbl.text() == ""

    def test_long_text_preserved(self, qapp):
        text = "1 = permissive (0.01)  ·  5 = moderate (0.05)  ·  10 = strict (0.10)"
        lbl = _subtitle(text)
        assert lbl.text() == text


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8  _make_language_combo
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qt
class TestMakeLanguageCombo:
    def test_returns_qcombobox(self, qapp):
        combo = _make_language_combo("en")
        assert isinstance(combo, QComboBox)

    def test_selects_matching_language(self, qapp):
        combo = _make_language_combo("fr")
        assert combo.currentData() == "fr"

    def test_defaults_to_first_when_code_unknown(self, qapp):
        combo = _make_language_combo("xx")
        # Unknown code → selected_index stays 0 (first language in registry)
        assert combo.currentIndex() == 0

    def test_contains_all_languages(self, qapp):
        from norvel_writer.config.defaults import LANGUAGES
        combo = _make_language_combo("en")
        assert combo.count() == len(LANGUAGES)

    def test_itemdata_matches_language_code(self, qapp):
        from norvel_writer.config.defaults import LANGUAGES
        combo = _make_language_combo("en")
        codes = [combo.itemData(i) for i in range(combo.count())]
        assert set(codes) == set(LANGUAGES.keys())

    def test_english_selected_by_code(self, qapp):
        combo = _make_language_combo("en")
        assert combo.currentData() == "en"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9  SettingsDialog._is_ollama_provider
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qt
class TestIsOllamaProvider:
    def test_returns_true_when_env_is_ollama(self, dialog):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            assert dialog._is_ollama_provider() is True

    def test_returns_true_when_env_unset(self, dialog):
        env = {k: v for k, v in os.environ.items() if k != "LLM_PROVIDER"}
        with patch.dict(os.environ, env, clear=True):
            assert dialog._is_ollama_provider() is True

    def test_returns_false_for_openai(self, dialog):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            assert dialog._is_ollama_provider() is False

    def test_returns_false_for_anthropic(self, dialog):
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}):
            assert dialog._is_ollama_provider() is False

    def test_returns_false_for_gemini(self, dialog):
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
            assert dialog._is_ollama_provider() is False

    def test_case_insensitive_ollama(self, dialog):
        with patch.dict(os.environ, {"LLM_PROVIDER": "OLLAMA"}):
            assert dialog._is_ollama_provider() is True

    def test_case_insensitive_openai(self, dialog):
        with patch.dict(os.environ, {"LLM_PROVIDER": "OpenAI"}):
            assert dialog._is_ollama_provider() is False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10  SettingsDialog._apply_preset
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qt
class TestApplyPreset:
    def test_stable_preset_sets_all_sliders(self, dialog):
        p = _PRESETS["Stable"]
        dialog._apply_preset(p)
        assert dialog._slider_temp.value()   == p["temperature"]
        assert dialog._slider_top_p.value()  == p["top_p"]
        assert dialog._slider_min_p.value()  == p["min_p"]
        assert dialog._slider_repeat.value() == p["repeat_penalty"]

    def test_balanced_preset_sets_all_sliders(self, dialog):
        p = _PRESETS["Balanced"]
        dialog._apply_preset(p)
        assert dialog._slider_temp.value()   == p["temperature"]
        assert dialog._slider_top_p.value()  == p["top_p"]
        assert dialog._slider_min_p.value()  == p["min_p"]
        assert dialog._slider_repeat.value() == p["repeat_penalty"]

    def test_creative_preset_sets_all_sliders(self, dialog):
        p = _PRESETS["Creative"]
        dialog._apply_preset(p)
        assert dialog._slider_temp.value()   == p["temperature"]
        assert dialog._slider_top_p.value()  == p["top_p"]
        assert dialog._slider_min_p.value()  == p["min_p"]
        assert dialog._slider_repeat.value() == p["repeat_penalty"]

    def test_custom_arbitrary_values(self, dialog):
        vals = {"temperature": 8, "top_p": 2, "min_p": 9, "repeat_penalty": 4}
        dialog._apply_preset(vals)
        assert dialog._slider_temp.value()   == 8
        assert dialog._slider_top_p.value()  == 2
        assert dialog._slider_min_p.value()  == 9
        assert dialog._slider_repeat.value() == 4

    def test_boundary_min_values(self, dialog):
        vals = {"temperature": 1, "top_p": 1, "min_p": 1, "repeat_penalty": 1}
        dialog._apply_preset(vals)
        assert dialog._slider_temp.value()   == 1
        assert dialog._slider_top_p.value()  == 1
        assert dialog._slider_min_p.value()  == 1
        assert dialog._slider_repeat.value() == 1

    def test_boundary_max_values(self, dialog):
        vals = {"temperature": 10, "top_p": 10, "min_p": 10, "repeat_penalty": 10}
        dialog._apply_preset(vals)
        assert dialog._slider_temp.value()   == 10
        assert dialog._slider_top_p.value()  == 10
        assert dialog._slider_min_p.value()  == 10
        assert dialog._slider_repeat.value() == 10


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11  SettingsDialog._set_combo_by_code
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qt
class TestSetComboByCode:
    def test_selects_item_with_matching_code(self, qapp):
        combo = QComboBox()
        combo.addItem("English", "en")
        combo.addItem("French",  "fr")
        combo.addItem("German",  "de")
        SettingsDialog._set_combo_by_code(combo, "fr")
        assert combo.currentData() == "fr"

    def test_selects_last_item_by_code(self, qapp):
        combo = QComboBox()
        combo.addItem("A", "en")
        combo.addItem("B", "fr")
        combo.addItem("C", "de")
        SettingsDialog._set_combo_by_code(combo, "de")
        assert combo.currentData() == "de"

    def test_falls_back_to_en_for_unknown_code(self, qapp):
        combo = QComboBox()
        combo.addItem("English", "en")
        combo.addItem("French",  "fr")
        SettingsDialog._set_combo_by_code(combo, "xx")
        assert combo.currentData() == "en"

    def test_no_change_when_code_and_en_both_absent(self, qapp):
        combo = QComboBox()
        combo.addItem("Chinese",  "zh")
        combo.addItem("Japanese", "ja")
        combo.setCurrentIndex(1)  # start on Japanese
        SettingsDialog._set_combo_by_code(combo, "xx")
        # Neither "xx" nor "en" exists → index stays at 1
        assert combo.currentIndex() == 1

    def test_selects_en_directly(self, qapp):
        combo = QComboBox()
        combo.addItem("French",  "fr")
        combo.addItem("English", "en")
        SettingsDialog._set_combo_by_code(combo, "en")
        assert combo.currentData() == "en"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12  SettingsDialog._load
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qt
class TestLoad:
    """_load reads AppConfig into widget state on dialog construction."""

    def _open(self, qapp, provider="ollama", **cfg_kwargs):
        cfg = _mock_cfg(**cfg_kwargs)
        with patch("norvel_writer.config.settings.get_config", return_value=cfg), \
             patch.dict(os.environ, {"LLM_PROVIDER": provider}, clear=False):
            dlg = SettingsDialog()
        return dlg, cfg

    # Sliders fall back to _DEFAULTS when config values are None
    def test_slider_defaults_when_all_none(self, qapp):
        dlg, _ = self._open(qapp)
        assert dlg._slider_temp.value()   == _DEFAULTS["temperature"]
        assert dlg._slider_top_p.value()  == _DEFAULTS["top_p"]
        assert dlg._slider_min_p.value()  == _DEFAULTS["min_p"]
        assert dlg._slider_repeat.value() == _DEFAULTS["repeat_penalty"]
        dlg.close()

    # Stored floats are converted to the nearest 1–10 step
    def test_sliders_load_stored_floats_as_steps(self, qapp):
        dlg, _ = self._open(
            qapp,
            ollama_gen_temperature=0.60,    # → step 3
            ollama_gen_top_p=0.50,          # → step 5
            ollama_gen_min_p=0.07,          # → step 7
            ollama_gen_repeat_penalty=1.40, # → step 5
        )
        assert dlg._slider_temp.value()   == 3
        assert dlg._slider_top_p.value()  == 5
        assert dlg._slider_min_p.value()  == 7
        assert dlg._slider_repeat.value() == 5
        dlg.close()

    def test_all_grid_boundaries_load_correctly(self, qapp):
        dlg, _ = self._open(
            qapp,
            ollama_gen_temperature=2.00,    # step 10
            ollama_gen_top_p=0.10,          # step 1
            ollama_gen_min_p=0.10,          # step 10
            ollama_gen_repeat_penalty=1.00, # step 1
        )
        assert dlg._slider_temp.value()   == 10
        assert dlg._slider_top_p.value()  == 1
        assert dlg._slider_min_p.value()  == 10
        assert dlg._slider_repeat.value() == 1
        dlg.close()

    # Seed field
    def test_seed_empty_when_none(self, qapp):
        dlg, _ = self._open(qapp, ollama_gen_seed=None)
        assert dlg._seed_edit.text() == ""
        dlg.close()

    def test_seed_displayed_when_set(self, qapp):
        dlg, _ = self._open(qapp, ollama_gen_seed=42)
        assert dlg._seed_edit.text() == "42"
        dlg.close()

    # num_ctx field
    def test_num_ctx_empty_when_none(self, qapp):
        dlg, _ = self._open(qapp, ollama_gen_num_ctx=None)
        assert dlg._num_ctx_edit.text() == ""
        dlg.close()

    def test_num_ctx_displayed_when_set(self, qapp):
        dlg, _ = self._open(qapp, ollama_gen_num_ctx=8192)
        assert dlg._num_ctx_edit.text() == "8192"
        dlg.close()

    # num_predict field
    def test_num_predict_empty_when_none(self, qapp):
        dlg, _ = self._open(qapp, ollama_gen_num_predict=None)
        assert dlg._num_predict_edit.text() == ""
        dlg.close()

    def test_num_predict_displayed_when_set(self, qapp):
        dlg, _ = self._open(qapp, ollama_gen_num_predict=4096)
        assert dlg._num_predict_edit.text() == "4096"
        dlg.close()

    # Advanced panel visibility
    # isVisible() requires the full parent chain to be shown on screen.
    # isHidden() checks only the widget's own flag — correct for an unshown dialog.
    def test_adv_group_not_hidden_when_ollama(self, qapp):
        dlg, _ = self._open(qapp, provider="ollama")
        assert dlg._adv_group.isHidden() is False
        dlg.close()

    def test_adv_group_hidden_when_openai(self, qapp):
        dlg, _ = self._open(qapp, provider="openai")
        assert dlg._adv_group.isHidden() is True
        dlg.close()

    def test_adv_group_hidden_when_anthropic(self, qapp):
        dlg, _ = self._open(qapp, provider="anthropic")
        assert dlg._adv_group.isHidden() is True
        dlg.close()

    # Basic text fields
    def test_ollama_url_loaded(self, qapp):
        dlg, _ = self._open(qapp, ollama_base_url="http://192.168.1.5:11434")
        assert dlg._ollama_url.text() == "http://192.168.1.5:11434"
        dlg.close()

    def test_chat_model_loaded(self, qapp):
        dlg, _ = self._open(qapp, default_chat_model="llama3.1:8b")
        assert dlg._chat_model.text() == "llama3.1:8b"
        dlg.close()

    def test_embed_model_loaded(self, qapp):
        dlg, _ = self._open(qapp, default_embed_model="nomic-embed-text")
        assert dlg._embed_model.text() == "nomic-embed-text"
        dlg.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13  SettingsDialog._save
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qt
class TestSave:
    """_save converts widget state back to AppConfig and calls side-effects."""

    _PATCHES = (
        "norvel_writer.ui.theme.apply_theme",
        "norvel_writer.llm.langchain_bridge.reset_singletons",
    )

    def _save(self, dlg, cfg):
        """Run _save with all external side-effects patched."""
        with patch("norvel_writer.config.settings.get_config", return_value=cfg), \
             patch(self._PATCHES[0]), \
             patch(self._PATCHES[1]):
            dlg._save()

    def _dialog(self, qapp, provider="ollama"):
        cfg = _mock_cfg()
        with patch("norvel_writer.config.settings.get_config", return_value=cfg), \
             patch.dict(os.environ, {"LLM_PROVIDER": provider}, clear=False):
            dlg = SettingsDialog()
        return dlg, cfg

    # Slider step → float conversion
    @pytest.mark.parametrize("step,expected_temp", [
        (1, 0.20), (5, 1.00), (10, 2.00),
    ])
    def test_temperature_step_converted_to_float(self, qapp, step, expected_temp):
        dlg, cfg = self._dialog(qapp)
        dlg._slider_temp.setValue(step)
        self._save(dlg, cfg)
        assert cfg.ollama_gen_temperature == pytest.approx(expected_temp)

    @pytest.mark.parametrize("step,expected_top_p", [
        (1, 0.10), (5, 0.50), (10, 1.00),
    ])
    def test_top_p_step_converted_to_float(self, qapp, step, expected_top_p):
        dlg, cfg = self._dialog(qapp)
        dlg._slider_top_p.setValue(step)
        self._save(dlg, cfg)
        assert cfg.ollama_gen_top_p == pytest.approx(expected_top_p)

    @pytest.mark.parametrize("step,expected_min_p", [
        (1, 0.010), (5, 0.050), (10, 0.100),
    ])
    def test_min_p_step_converted_to_float(self, qapp, step, expected_min_p):
        dlg, cfg = self._dialog(qapp)
        dlg._slider_min_p.setValue(step)
        self._save(dlg, cfg)
        assert cfg.ollama_gen_min_p == pytest.approx(expected_min_p)

    @pytest.mark.parametrize("step,expected_repeat", [
        (1, 1.00), (5, 1.40), (10, 1.90),
    ])
    def test_repeat_step_converted_to_float(self, qapp, step, expected_repeat):
        dlg, cfg = self._dialog(qapp)
        dlg._slider_repeat.setValue(step)
        self._save(dlg, cfg)
        assert cfg.ollama_gen_repeat_penalty == pytest.approx(expected_repeat)

    # Seed field
    def test_seed_valid_integer_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._seed_edit.setText("99999")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_seed == 99999

    def test_seed_negative_integer_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._seed_edit.setText("-1")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_seed == -1

    def test_seed_empty_saves_none(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._seed_edit.setText("")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_seed is None

    def test_seed_whitespace_only_saves_none(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._seed_edit.setText("   ")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_seed is None

    def test_seed_non_numeric_saves_none(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._seed_edit.setText("abc")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_seed is None

    def test_seed_float_string_saves_none(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._seed_edit.setText("1.5")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_seed is None

    # num_ctx validation
    def test_num_ctx_valid_value_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_ctx_edit.setText("16384")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_ctx == 16384

    def test_num_ctx_below_minimum_clamped_to_512(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_ctx_edit.setText("100")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_ctx == 512

    def test_num_ctx_exactly_512_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_ctx_edit.setText("512")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_ctx == 512

    def test_num_ctx_empty_saves_none(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_ctx_edit.setText("")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_ctx is None

    def test_num_ctx_non_numeric_saves_none(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_ctx_edit.setText("big")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_ctx is None

    # num_predict validation
    def test_num_predict_valid_value_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_predict_edit.setText("4096")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_predict == 4096

    def test_num_predict_below_minimum_clamped_to_64(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_predict_edit.setText("10")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_predict == 64

    def test_num_predict_exactly_64_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_predict_edit.setText("64")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_predict == 64

    def test_num_predict_empty_saves_none(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_predict_edit.setText("")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_predict is None

    def test_num_predict_non_numeric_saves_none(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._num_predict_edit.setText("many")
        self._save(dlg, cfg)
        assert cfg.ollama_gen_num_predict is None

    # Side-effects
    def test_config_save_called_once(self, qapp):
        dlg, cfg = self._dialog(qapp)
        self._save(dlg, cfg)
        cfg.save.assert_called_once()

    def test_reset_singletons_called_when_ollama(self, qapp):
        dlg, cfg = self._dialog(qapp, provider="ollama")
        with patch("norvel_writer.config.settings.get_config", return_value=cfg), \
             patch(self._PATCHES[0]), \
             patch(self._PATCHES[1]) as mock_reset, \
             patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            dlg._save()
        mock_reset.assert_called_once()

    def test_reset_singletons_not_called_when_not_ollama(self, qapp):
        dlg, cfg = self._dialog(qapp, provider="openai")
        with patch("norvel_writer.config.settings.get_config", return_value=cfg), \
             patch(self._PATCHES[0]), \
             patch(self._PATCHES[1]) as mock_reset, \
             patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            dlg._save()
        mock_reset.assert_not_called()
        dlg.close()

    # Text fields written back
    def test_ollama_url_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._ollama_url.setText("http://10.0.0.1:11434")
        self._save(dlg, cfg)
        assert cfg.ollama_base_url == "http://10.0.0.1:11434"

    def test_chat_model_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._chat_model.setText("llama3.1:8b")
        self._save(dlg, cfg)
        assert cfg.default_chat_model == "llama3.1:8b"

    def test_embed_model_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._embed_model.setText("bge-m3")
        self._save(dlg, cfg)
        assert cfg.default_embed_model == "bge-m3"

    def test_theme_saved(self, qapp):
        dlg, cfg = self._dialog(qapp)
        dlg._theme.setCurrentText("light")
        self._save(dlg, cfg)
        assert cfg.theme == "light"

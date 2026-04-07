"""Central editor panel with autosave and word count."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QAction,
)

from norvel_writer.config.defaults import AUTOSAVE_INTERVAL_MS

# Maps (iso-lang-code, gender) -> edge-tts neural voice name.
# Covers every language in defaults.LANGUAGES.
_VOICE_MAP: dict[tuple[str, str], str] = {
    ("en",    "male"):   "en-US-GuyNeural",
    ("en",    "female"): "en-US-JennyNeural",
    ("zh",    "male"):   "zh-CN-YunxiNeural",
    ("zh",    "female"): "zh-CN-XiaoxiaoNeural",
    ("zh-tw", "male"):   "zh-TW-YunJheNeural",
    ("zh-tw", "female"): "zh-TW-HsiaoChenNeural",
    ("ja",    "male"):   "ja-JP-KeitaNeural",
    ("ja",    "female"): "ja-JP-NanamiNeural",
    ("ko",    "male"):   "ko-KR-InJoonNeural",
    ("ko",    "female"): "ko-KR-SunHiNeural",
    ("es",    "male"):   "es-ES-AlvaroNeural",
    ("es",    "female"): "es-ES-ElviraNeural",
    ("fr",    "male"):   "fr-FR-HenriNeural",
    ("fr",    "female"): "fr-FR-DeniseNeural",
    ("de",    "male"):   "de-DE-ConradNeural",
    ("de",    "female"): "de-DE-KatjaNeural",
    ("ru",    "male"):   "ru-RU-DmitryNeural",
    ("ru",    "female"): "ru-RU-SvetlanaNeural",
    ("pt",    "male"):   "pt-BR-AntonioNeural",
    ("pt",    "female"): "pt-BR-FranciscaNeural",
    ("ar",    "male"):   "ar-SA-HamedNeural",
    ("ar",    "female"): "ar-SA-ZariyahNeural",
    ("hi",    "male"):   "hi-IN-MadhurNeural",
    ("hi",    "female"): "hi-IN-SwaraNeural",
    ("it",    "male"):   "it-IT-DiegoNeural",
    ("it",    "female"): "it-IT-ElsaNeural",
    ("nl",    "male"):   "nl-NL-MaartenNeural",
    ("nl",    "female"): "nl-NL-ColetteNeural",
    ("pl",    "male"):   "pl-PL-MarekNeural",
    ("pl",    "female"): "pl-PL-ZofiaNeural",
    ("tr",    "male"):   "tr-TR-AhmetNeural",
    ("tr",    "female"): "tr-TR-EmelNeural",
    ("vi",    "male"):   "vi-VN-NamMinhNeural",
    ("vi",    "female"): "vi-VN-HoaiMyNeural",
    ("th",    "male"):   "th-TH-NiwatNeural",
    ("th",    "female"): "th-TH-PremwadeeNeural",
}


def _resolve_voice(text: str, gender: str) -> str:
    """Auto-detect language from text and return the matching edge-tts voice."""
    lang = "en"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        detected = detect(text[:500])
        # Normalise langdetect variants to our registry keys
        if detected.startswith("zh"):
            lang = "zh-tw" if "tw" in detected else "zh"
        else:
            lang = detected.split("-")[0]
    except Exception:
        pass
    key = (lang, gender)
    return _VOICE_MAP.get(key) or _VOICE_MAP.get(("en", gender), "en-US-GuyNeural")


class _TTSWorker(QThread):
    """
    Background thread: fetches audio via edge-tts (neural, multi-language),
    then plays it using the Windows MCI API (ctypes, stdlib — no extra deps).
    Supports mid-speech stop.  Requires internet for edge-tts.
    """

    _MCI_ALIAS = "norvel_tts_player"

    finished = Signal()

    def __init__(self, text: str, voice_gender: str, parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self._voice_gender = voice_gender.lower()
        self._stop_flag = False

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._speak_async())
        except Exception:
            pass
        finally:
            loop.close()
            self.finished.emit()

    async def _speak_async(self) -> None:
        import edge_tts

        voice = _resolve_voice(self._text, self._voice_gender)
        communicate = edge_tts.Communicate(self._text, voice)

        audio_data = b""
        async for chunk in communicate.stream():
            if self._stop_flag:
                return
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        if self._stop_flag or not audio_data:
            return

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        try:
            tmp.write(audio_data)
            tmp.close()
            await self._play_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    async def _play_file(self, path: str) -> None:
        if sys.platform == "win32":
            await self._play_win32(path)
        else:
            # macOS / Linux fallback via subprocess
            import subprocess
            player = "afplay" if sys.platform == "darwin" else "mpg123"
            proc = await asyncio.create_subprocess_exec(player, path)
            while proc.returncode is None:
                if self._stop_flag:
                    proc.terminate()
                    break
                await asyncio.sleep(0.1)

    async def _play_win32(self, path: str) -> None:
        import ctypes
        winmm = ctypes.windll.winmm
        alias = self._MCI_ALIAS
        buf = ctypes.create_unicode_buffer(256)
        winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
        winmm.mciSendStringW(f'play {alias}', None, 0, None)
        try:
            while True:
                winmm.mciSendStringW(f'status {alias} mode', buf, 255, None)
                if buf.value not in ("playing", "seeking") or self._stop_flag:
                    break
                await asyncio.sleep(0.1)
        finally:
            winmm.mciSendStringW(f'stop {alias}', None, 0, None)
            winmm.mciSendStringW(f'close {alias}', None, 0, None)

    def stop_speaking(self) -> None:
        self._stop_flag = True
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.winmm.mciSendStringW(
                    f'stop {self._MCI_ALIAS}', None, 0, None
                )
            except Exception:
                pass


class EditorPanel(QWidget):
    """
    Central writing editor.
    Emits content_changed when text changes (debounced).
    Emits autosave_triggered periodically.
    """

    content_changed = Signal(str)     # debounced text change
    autosave_triggered = Signal(str)  # chapter_id, content

    def __init__(self, project_manager, parent=None) -> None:
        super().__init__(parent)
        self._pm = project_manager
        self._current_chapter_id: Optional[str] = None
        self._current_project_id: Optional[str] = None
        self._dirty = False
        self._tts_worker: Optional[_TTSWorker] = None
        self._build_ui()
        self._setup_autosave()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)

        self._chapter_label = QLabel("No chapter selected")
        self._chapter_label.setStyleSheet("padding: 0 8px; color: #89b4fa; font-weight: bold;")

        self._toolbar.addWidget(self._chapter_label)
        self._toolbar.addSeparator()

        btn_save = QPushButton("Save")
        btn_save.setShortcut("Ctrl+S")
        btn_save.setToolTip("Save (Ctrl+S)")
        btn_save.clicked.connect(self._save_now)
        self._toolbar.addWidget(btn_save)

        self._word_count_label = QLabel("0 words")
        self._word_count_label.setObjectName("subtitle")
        self._word_count_label.setStyleSheet("padding: 0 12px; color: #a6adc8;")
        self._toolbar.addWidget(self._word_count_label)

        # Stretch spacer pushes Read controls to the right
        spacer = QWidget()
        spacer.setSizePolicy(
            spacer.sizePolicy().horizontalPolicy(),
            spacer.sizePolicy().verticalPolicy(),
        )
        from PySide6.QtWidgets import QSizePolicy
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._toolbar.addWidget(spacer)

        self._toolbar.addSeparator()

        voice_label = QLabel("Voice:")
        voice_label.setStyleSheet("padding: 0 4px 0 8px; color: #a6adc8;")
        self._toolbar.addWidget(voice_label)

        self._voice_combo = QComboBox()
        self._voice_combo.addItems(["Male", "Female"])
        self._voice_combo.setFixedWidth(80)
        self._toolbar.addWidget(self._voice_combo)

        self._btn_read = QPushButton("Read Aloud")
        self._btn_read.setToolTip("Read chapter aloud (auto-detects language, requires internet)")
        self._btn_read.setStyleSheet(
            "QPushButton { padding: 2px 10px; }"
        )
        self._btn_read.clicked.connect(self._toggle_read)
        self._toolbar.addWidget(self._btn_read)

        layout.addWidget(self._toolbar)

        # Editor
        self._editor = QTextEdit()
        self._editor.setPlaceholderText(
            "Select a chapter from the project panel, or start writing here…"
        )
        self._editor.textChanged.connect(self._on_text_changed)

        # Debounce timer
        self._change_timer = QTimer(self)
        self._change_timer.setSingleShot(True)
        self._change_timer.setInterval(500)
        self._change_timer.timeout.connect(self._emit_content_changed)

        layout.addWidget(self._editor)

    def _setup_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(AUTOSAVE_INTERVAL_MS)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()

    def load_chapter(self, chapter_id: str, project_id: str) -> None:
        self._current_chapter_id = chapter_id
        self._current_project_id = project_id
        chapter = self._pm.get_chapter(chapter_id)
        if chapter:
            self._chapter_label.setText(chapter["title"])
        # Load accepted draft if any
        draft = self._pm.get_accepted_draft(chapter_id)
        self._editor.blockSignals(True)
        self._editor.setPlainText(draft["content"] if draft else "")
        self._editor.blockSignals(False)
        self._dirty = False
        self._update_word_count()

    def insert_text(self, text: str) -> None:
        """Insert text at cursor position (used by draft panel)."""
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n\n" + text)
        self._editor.setTextCursor(cursor)
        self._editor.ensureCursorVisible()

    def get_content(self) -> str:
        return self._editor.toPlainText()

    def set_content(self, text: str) -> None:
        self._editor.setPlainText(text)

    def _on_text_changed(self) -> None:
        self._dirty = True
        self._update_word_count()
        self._change_timer.start()

    def _emit_content_changed(self) -> None:
        self.content_changed.emit(self._editor.toPlainText())

    def _update_word_count(self) -> None:
        from norvel_writer.utils.text_utils import count_words
        text = self._editor.toPlainText()
        count = count_words(text)
        self._word_count_label.setText(f"{count:,} words")

    def _toggle_read(self) -> None:
        if self._tts_worker and self._tts_worker.isRunning():
            self._tts_worker.stop_speaking()
            self._tts_worker.wait()
            return
        text = self._editor.toPlainText().strip()
        if not text:
            return
        gender = self._voice_combo.currentText()
        self._tts_worker = _TTSWorker(text, gender, self)
        self._tts_worker.finished.connect(self._on_tts_finished)
        self._btn_read.setText("Stop Reading")
        self._voice_combo.setEnabled(False)
        self._tts_worker.start()

    def _on_tts_finished(self) -> None:
        self._btn_read.setText("Read Aloud")
        self._voice_combo.setEnabled(True)
        self._tts_worker = None

    def _save_now(self) -> None:
        self._autosave()

    def _autosave(self) -> None:
        if not self._dirty or not self._current_chapter_id:
            return
        content = self._editor.toPlainText()
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        draft_id = self._pm.save_draft(
            chapter_id=self._current_chapter_id,
            content=content,
            model_used="manual",
        )
        self._pm.accept_draft(draft_id)
        self._dirty = False
        from norvel_writer.utils.text_utils import count_words
        wc = count_words(content)
        self._pm.update_chapter(
            self._current_chapter_id, word_count=wc
        )

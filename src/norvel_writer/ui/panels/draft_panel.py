"""Draft generation sidebar panel."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from norvel_writer.ui.widgets.streaming_text import StreamingTextEdit

# ── TTS voice map: (iso-lang, gender) → edge-tts neural voice ────────────────
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
    lang = "en"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        detected = detect(text[:500])
        lang = "zh-tw" if detected.startswith("zh") and "tw" in detected else detected.split("-")[0]
    except Exception:
        pass
    return _VOICE_MAP.get((lang, gender)) or _VOICE_MAP.get(("en", gender), "en-US-GuyNeural")


class _TTSWorker(QThread):
    """Fetches audio via edge-tts and plays it using Windows MCI (no extra deps)."""

    finished = Signal()
    _MCI_ALIAS = "norvel_tts_player"

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
        else:
            player = "afplay" if sys.platform == "darwin" else "mpg123"
            proc = await asyncio.create_subprocess_exec(player, path)
            while proc.returncode is None:
                if self._stop_flag:
                    proc.terminate()
                    break
                await asyncio.sleep(0.1)

    def stop_speaking(self) -> None:
        self._stop_flag = True
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.winmm.mciSendStringW(f'stop {self._MCI_ALIAS}', None, 0, None)
            except Exception:
                pass


class DraftPanel(QDockWidget):
    """
    Sidebar for generating continuations and rewrites.
    Connects to DraftEngine via AsyncWorker.
    """

    insert_into_editor = Signal(str)

    def __init__(self, project_manager, draft_engine, async_worker, parent=None) -> None:
        super().__init__("AI Draft Assistant", parent)
        self._pm = project_manager
        self._engine = draft_engine
        self._worker = async_worker
        self._current_chapter_id: Optional[str] = None
        self._current_project_id: Optional[str] = None
        self._active_future = None
        self._tts_worker: Optional[_TTSWorker] = None
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Instruction input
        layout.addWidget(QLabel("Instruction:"))
        self._instruction = QLineEdit()
        self._instruction.setPlaceholderText("Continue the story…")
        layout.addWidget(self._instruction)

        # Writing language — populated from the shared LANGUAGES registry
        from norvel_writer.config.defaults import LANGUAGES, language_display
        from norvel_writer.config.settings import get_config
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Write in:"))
        self._language = QComboBox()
        self._language.setEditable(True)
        for code, (eng, native) in LANGUAGES.items():
            label = language_display(code)
            self._language.addItem(label, code)
        # Default to the configured content language
        default_lang = get_config().default_content_language
        self._set_language(default_lang)
        lang_row.addWidget(self._language)
        layout.addLayout(lang_row)

        # Style mode
        layout.addWidget(QLabel("Style mode:"))
        self._style_mode = QComboBox()
        self._style_mode.addItems([
            "inspired_by",
            "imitate_closely",
            "preserve_tone_rhythm",
            "avoid_exact_phrasing",
        ])
        layout.addWidget(self._style_mode)

        # Mode selector
        btn_row = QHBoxLayout()
        self._btn_continue = QPushButton("Continue Draft")
        self._btn_continue.setObjectName("primary")
        self._btn_continue.clicked.connect(self._on_continue)

        self._btn_rewrite = QPushButton("Rewrite Selection")
        self._btn_rewrite.clicked.connect(self._on_rewrite)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setEnabled(False)

        btn_row.addWidget(self._btn_continue)
        btn_row.addWidget(self._btn_rewrite)
        btn_row.addWidget(self._btn_stop)
        layout.addLayout(btn_row)

        # Output area
        layout.addWidget(QLabel("Generated text:"))
        self._output = StreamingTextEdit()
        self._output.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._output)

        # Accept / Discard
        accept_row = QHBoxLayout()
        self._btn_insert = QPushButton("Insert into Editor")
        self._btn_insert.clicked.connect(self._on_insert)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(self._output.clear)
        accept_row.addWidget(self._btn_insert)
        accept_row.addWidget(self._btn_clear)
        layout.addLayout(accept_row)

        # Status label
        self._status = QLabel("")
        self._status.setObjectName("subtitle")
        layout.addWidget(self._status)

        # ── Read Aloud ────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244;")
        layout.addWidget(sep)

        layout.addWidget(QLabel("Read Chapter Aloud:"))

        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Voice:"))
        self._voice_combo = QComboBox()
        self._voice_combo.addItems(["Male", "Female"])
        voice_row.addWidget(self._voice_combo)
        voice_row.addStretch()
        layout.addLayout(voice_row)

        self._btn_read = QPushButton("Start Reading")
        self._btn_read.clicked.connect(self._toggle_read)
        layout.addWidget(self._btn_read)

        self.setWidget(container)

    def set_context(
        self,
        chapter_id: str,
        project_id: str,
        get_editor_content,
    ) -> None:
        self._current_chapter_id = chapter_id
        self._current_project_id = project_id
        self._get_editor_content = get_editor_content
        # Auto-set language: project language overrides the app default
        project = self._pm.get_project(project_id)
        if project:
            self._set_language(project.get("language", "en"))

    def _on_continue(self) -> None:
        if not self._current_chapter_id:
            self._status.setText("No chapter selected.")
            return

        instruction = self._instruction.text().strip() or "Continue the story."
        style_mode = self._style_mode.currentText()
        language = self._language.currentText()  # e.g. "Chinese (中文)" — full label for the prompt
        current_text = self._get_editor_content() if hasattr(self, "_get_editor_content") else ""

        self._output.start_stream()
        self._set_generating(True)
        self._status.setText("Generating…")

        async def _run_and_stream():
            try:
                agen = await self._engine.continue_draft(
                    project_id=self._current_project_id,
                    chapter_id=self._current_chapter_id,
                    current_text=current_text,
                    user_instruction=instruction,
                    style_mode=style_mode,
                    language=language,
                )
                async for chunk in agen:
                    yield chunk
            except Exception as exc:
                yield f"\n[Error: {exc}]"

        self._active_future = self._worker.run_stream(
            agen=_run_and_stream(),
            on_chunk=self._output.append_chunk,
            on_done=self._on_stream_done,
            on_error=self._on_stream_error,
        )

    def _on_rewrite(self) -> None:
        if not self._current_chapter_id:
            self._status.setText("No chapter selected.")
            return

        instruction = self._instruction.text().strip() or "Rewrite this passage."
        style_mode = self._style_mode.currentText()
        language = self._language.currentText()  # e.g. "Chinese (中文)" — full label for the prompt

        # Use editor selection or full text
        if hasattr(self, "_get_editor_content"):
            passage = self._get_editor_content()
        else:
            passage = ""

        if not passage.strip():
            self._status.setText("Nothing to rewrite.")
            return

        self._output.start_stream()
        self._set_generating(True)
        self._status.setText("Rewriting…")

        async def _run_and_stream():
            try:
                agen = await self._engine.rewrite_passage(
                    project_id=self._current_project_id,
                    passage=passage,
                    user_instruction=instruction,
                    style_mode=style_mode,
                    language=language,
                )
                async for chunk in agen:
                    yield chunk
            except Exception as exc:
                yield f"\n[Error: {exc}]"

        self._active_future = self._worker.run_stream(
            agen=_run_and_stream(),
            on_chunk=self._output.append_chunk,
            on_done=self._on_stream_done,
            on_error=self._on_stream_error,
        )

    def _on_stop(self) -> None:
        if self._active_future and not self._active_future.done():
            self._active_future.cancel()
        self._on_stream_done()

    def _on_stream_done(self) -> None:
        self._output.finish_stream()
        self._set_generating(False)
        self._status.setText("Done.")

    def _on_stream_error(self, exc: Exception) -> None:
        self._set_generating(False)
        self._status.setText(f"Error: {exc}")

    def _on_insert(self) -> None:
        text = self._output.get_text().strip()
        if text:
            self.insert_into_editor.emit(text)

    def _set_language(self, code: str) -> None:
        """Select a language in the combo by its ISO code."""
        for i in range(self._language.count()):
            if self._language.itemData(i) == code:
                self._language.setCurrentIndex(i)
                return

    def _set_generating(self, generating: bool) -> None:
        self._btn_continue.setEnabled(not generating)
        self._btn_rewrite.setEnabled(not generating)
        self._btn_stop.setEnabled(generating)

    # ── Read Aloud ────────────────────────────────────────────────────────

    def _toggle_read(self) -> None:
        if self._tts_worker and self._tts_worker.isRunning():
            self._tts_worker.stop_speaking()
            self._tts_worker.wait()
            return
        text = ""
        if hasattr(self, "_get_editor_content"):
            text = self._get_editor_content().strip()
        if not text:
            self._status.setText("No chapter text to read.")
            return
        gender = self._voice_combo.currentText()
        self._tts_worker = _TTSWorker(text, gender, self)
        self._tts_worker.finished.connect(self._on_tts_finished)
        self._btn_read.setText("Stop Reading")
        self._voice_combo.setEnabled(False)
        self._status.setText("Reading aloud…")
        self._tts_worker.start()

    def _on_tts_finished(self) -> None:
        self._btn_read.setText("Start Reading")
        self._voice_combo.setEnabled(True)
        self._status.setText("")
        self._tts_worker = None

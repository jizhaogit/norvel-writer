"""Draft generation sidebar panel."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
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

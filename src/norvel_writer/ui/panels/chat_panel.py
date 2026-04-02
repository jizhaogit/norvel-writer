"""Free-form Q&A chat panel for asking questions about the project."""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ChatPanel(QDockWidget):
    """
    Free-form chat panel. The user can ask any question in any language and
    receive a streaming answer with RAG context from the project.

    Examples:
      - "Who is the antagonist?"
      - "Suggest three ways to resolve the conflict in chapter 5."
      - "给我一些关于主角动机的建议" (Chinese)
      - "What does the map of the northern kingdom look like?"
    """

    def __init__(self, project_manager, draft_engine, async_worker, parent=None) -> None:
        super().__init__("Ask / Suggest", parent)
        self._pm = project_manager
        self._engine = draft_engine
        self._worker = async_worker
        self._current_project_id: Optional[str] = None
        self._history: List[Dict[str, str]] = []
        self._active_future = None
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Conversation display
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._chat_display.setPlaceholderText(
            "Ask anything about your project…\n\n"
            "Examples:\n"
            "• Who are the main characters?\n"
            "• Suggest three ways to end chapter 5\n"
            "• What are the rules of magic in my world?\n"
            "• 给我一些关于主角动机的建议\n"
            "• どのようにクライマックスを改善できますか？"
        )
        layout.addWidget(self._chat_display)

        # Input area
        self._input = QPlainTextEdit()
        self._input.setPlaceholderText("Ask a question or request a suggestion… (Enter to send, Shift+Enter for newline)")
        self._input.setMaximumHeight(80)
        self._input.setMinimumHeight(60)
        # Intercept Enter key
        self._input.installEventFilter(self)
        layout.addWidget(self._input)

        btn_row = QHBoxLayout()
        self._btn_send = QPushButton("Ask")
        self._btn_send.setObjectName("primary")
        self._btn_send.clicked.connect(self._on_send)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setEnabled(False)

        self._btn_clear = QPushButton("Clear Chat")
        self._btn_clear.clicked.connect(self._clear_chat)

        btn_row.addWidget(self._btn_send)
        btn_row.addWidget(self._btn_stop)
        btn_row.addWidget(self._btn_clear)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setObjectName("subtitle")
        layout.addWidget(self._status)

        self.setWidget(container)

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            key_event = event  # type: QKeyEvent
            if (key_event.key() == Qt.Key.Key_Return and
                    not (key_event.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    def set_project(self, project_id: str) -> None:
        self._current_project_id = project_id
        self._history.clear()
        self._chat_display.clear()

    def _on_send(self) -> None:
        question = self._input.toPlainText().strip()
        if not question:
            return
        if not self._current_project_id:
            self._status.setText("Select a project first.")
            return

        self._input.clear()
        self._append_user_message(question)
        self._history.append({"role": "user", "content": question})
        self._set_generating(True)
        self._status.setText("Thinking…")

        # Detect language for response guidance — pass the full name so the model understands
        from norvel_writer.utils.text_utils import detect_language
        _LANG_NAMES = {
            "zh": "Chinese (中文)", "zh-cn": "Chinese (中文)", "zh-tw": "Chinese Traditional (繁體中文)",
            "ja": "Japanese (日本語)", "ko": "Korean (한국어)",
            "es": "Spanish (Español)", "fr": "French (Français)", "de": "German (Deutsch)",
            "ru": "Russian (Русский)", "pt": "Portuguese (Português)",
            "ar": "Arabic (العربية)", "hi": "Hindi (हिन्दी)",
            "en": "English",
        }
        lang_code = detect_language(question)
        language = _LANG_NAMES.get(lang_code, lang_code)

        # Start the AI response block
        self._append_ai_prefix()
        self._current_ai_text: list[str] = []

        import asyncio

        async def _run_and_stream():
            try:
                agen = await self._engine.chat_with_context(
                    project_id=self._current_project_id,
                    question=question,
                    history=self._history[:-1],  # exclude the just-added user message
                    language=language,
                )
                async for chunk in agen:
                    yield chunk
            except Exception as exc:
                yield f"\n[Error: {exc}]"

        self._active_future = self._worker.run_stream(
            agen=_run_and_stream(),
            on_chunk=self._on_ai_chunk,
            on_done=self._on_stream_done,
            on_error=self._on_stream_error,
        )

    def _on_stop(self) -> None:
        if self._active_future and not self._active_future.done():
            self._active_future.cancel()
        self._on_stream_done()

    def _on_ai_chunk(self, chunk: str) -> None:
        self._current_ai_text.append(chunk)
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self._chat_display.setTextCursor(cursor)
        self._chat_display.ensureCursorVisible()

    def _on_stream_done(self) -> None:
        full_response = "".join(self._current_ai_text)
        self._history.append({"role": "assistant", "content": full_response})
        self._current_ai_text = []
        # Add a blank line separator
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n\n")
        self._chat_display.setTextCursor(cursor)
        self._set_generating(False)
        self._status.setText("")

    def _on_stream_error(self, exc: Exception) -> None:
        self._set_generating(False)
        self._status.setText(f"Error: {exc}")

    def _append_user_message(self, text: str) -> None:
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # User label in accent colour
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#89b4fa"))
        fmt.setFontWeight(700)
        cursor.setCharFormat(fmt)
        cursor.insertText("You: ")

        # Message text in normal colour
        normal_fmt = QTextCharFormat()
        normal_fmt.setForeground(QColor("#cdd6f4"))
        normal_fmt.setFontWeight(400)
        cursor.setCharFormat(normal_fmt)
        cursor.insertText(text + "\n\n")
        self._chat_display.setTextCursor(cursor)
        self._chat_display.ensureCursorVisible()

    def _append_ai_prefix(self) -> None:
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#a6e3a1"))
        fmt.setFontWeight(700)
        cursor.setCharFormat(fmt)
        cursor.insertText("Assistant: ")
        normal_fmt = QTextCharFormat()
        normal_fmt.setForeground(QColor("#cdd6f4"))
        normal_fmt.setFontWeight(400)
        cursor.setCharFormat(normal_fmt)
        self._chat_display.setTextCursor(cursor)

    def _clear_chat(self) -> None:
        self._history.clear()
        self._chat_display.clear()

    def _set_generating(self, generating: bool) -> None:
        self._btn_send.setEnabled(not generating)
        self._btn_stop.setEnabled(generating)
        self._input.setEnabled(not generating)

"""QTextEdit with streaming token append and stop support."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QTextEdit


class StreamingTextEdit(QTextEdit):
    """
    A QTextEdit that accumulates streamed tokens.
    Tokens are appended without replacing existing content.
    """

    stream_started = Signal()
    stream_finished = Signal(str)  # emits final text

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self._streaming = False
        self._buffer: list[str] = []

    def start_stream(self) -> None:
        self._streaming = True
        self._buffer = []
        self.stream_started.emit()

    @Slot(str)
    def append_chunk(self, chunk: str) -> None:
        if not self._streaming:
            return
        self._buffer.append(chunk)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def finish_stream(self) -> None:
        self._streaming = False
        final = "".join(self._buffer)
        self.stream_finished.emit(final)

    def set_text(self, text: str) -> None:
        self._streaming = False
        self._buffer = []
        self.setPlainText(text)

    def get_text(self) -> str:
        return self.toPlainText()

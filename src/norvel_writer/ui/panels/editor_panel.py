"""Central editor panel with autosave and word count."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QAction,
)

from norvel_writer.config.defaults import AUTOSAVE_INTERVAL_MS


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

        act_save = QAction("Save", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_now)
        self._toolbar.addAction(act_save)

        self._word_count_label = QLabel("0 words")
        self._word_count_label.setObjectName("subtitle")
        self._word_count_label.setStyleSheet("padding: 0 12px; color: #a6adc8;")

        self._toolbar.addWidget(self._word_count_label)

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
        text = self._editor.toPlainText()
        count = len(text.split()) if text.strip() else 0
        self._word_count_label.setText(f"{count:,} words")

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
        wc = len(content.split()) if content.strip() else 0
        self._pm.update_chapter(
            self._current_chapter_id, word_count=wc
        )

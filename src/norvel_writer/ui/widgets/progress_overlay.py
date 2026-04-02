"""Non-blocking progress overlay widget."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class ProgressOverlay(QFrame):
    """
    Semi-transparent overlay with a progress bar and status label.
    Show/hide on top of any widget.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("progressOverlay")
        self.setStyleSheet(
            "#progressOverlay {"
            "  background-color: rgba(0,0,0,140);"
            "  border-radius: 8px;"
            "}"
        )
        self.setVisible(False)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)

        self._label = QLabel("Processing…")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #cdd6f4; font-size: 14px; font-weight: bold;")

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedWidth(300)
        self._bar.setFixedHeight(8)

        layout.addWidget(self._label)
        layout.addWidget(self._bar)

    def show_progress(self, message: str = "Processing…", value: int = 0) -> None:
        self._label.setText(message)
        self._bar.setValue(value)
        self.resize(self.parent().size())  # type: ignore[union-attr]
        self.setVisible(True)
        self.raise_()

    def update_progress(self, value: int, message: str = "") -> None:
        if message:
            self._label.setText(message)
        self._bar.setValue(value)

    def hide_progress(self) -> None:
        self.setVisible(False)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.parent():
            self.resize(self.parent().size())  # type: ignore[union-attr]

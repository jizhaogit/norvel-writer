"""Export dialog: choose format and destination."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QCheckBox,
)


class ExportDialog(QDialog):
    def __init__(self, project_name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Project")
        self.setMinimumWidth(480)
        self._path: Optional[Path] = None
        self._build_ui(project_name)

    def _build_ui(self, project_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel(f"<b>Export: {project_name}</b>"))

        layout.addWidget(QLabel("Format:"))
        self._format = QComboBox()
        self._format.addItems(["Markdown (.md)", "Word (.docx)", "NotebookLM Package (.md)"])
        layout.addWidget(self._format)

        layout.addWidget(QLabel("Destination:"))
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Choose save location…")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self._path_edit)
        path_row.addWidget(btn_browse)
        layout.addLayout(path_row)

        self._open_after = QCheckBox("Open in NotebookLM after export")
        self._open_after.setVisible(False)
        self._format.currentIndexChanged.connect(
            lambda i: self._open_after.setVisible(i == 2)
        )
        layout.addWidget(self._open_after)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        fmt = self._format.currentIndex()
        if fmt == 1:
            path, _ = QFileDialog.getSaveFileName(self, "Save As", "", "Word (*.docx)")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save As", "", "Markdown (*.md)")
        if path:
            self._path_edit.setText(path)

    @property
    def selected_format(self) -> str:
        idx = self._format.currentIndex()
        return ["md", "docx", "notebooklm"][idx]

    @property
    def destination(self) -> Optional[Path]:
        t = self._path_edit.text().strip()
        return Path(t) if t else None

    @property
    def open_in_notebooklm(self) -> bool:
        return self._open_after.isChecked() and self._open_after.isVisible()

"""File ingestion dialog with progress bar."""
from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from norvel_writer.config.defaults import DOC_TYPES, SUPPORTED_FORMATS


class IngestDialog(QDialog):
    """Multi-file ingest dialog with type selector and progress."""

    def __init__(self, project_manager, async_worker, project_id: str, parent=None) -> None:
        super().__init__(parent)
        self._pm = project_manager
        self._worker = async_worker
        self._project_id = project_id
        self._files: List[Path] = []
        self.setWindowTitle("Add Files to Memory")
        self.setMinimumWidth(500)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel("<b>Files to add:</b>"))
        self._file_list = QListWidget()
        layout.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Browse Files…")
        btn_add.clicked.connect(self._browse)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Import as:"))
        self._doc_type = QComboBox()
        for dt in DOC_TYPES:
            self._doc_type.addItem(dt.replace("_", " ").title(), dt)
        type_row.addWidget(self._doc_type)
        layout.addLayout(type_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setObjectName("subtitle")
        layout.addWidget(self._status)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        self._buttons.accepted.connect(self._ingest)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    def _browse(self) -> None:
        exts = " ".join(f"*{e}" for e in SUPPORTED_FORMATS)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", f"Supported ({exts})"
        )
        for p in paths:
            path = Path(p)
            if path not in self._files:
                self._files.append(path)
                self._file_list.addItem(path.name)

    def _clear(self) -> None:
        self._files.clear()
        self._file_list.clear()

    def _ingest(self) -> None:
        if not self._files:
            self.accept()
            return

        doc_type = self._doc_type.currentData()
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._buttons.setEnabled(False)
        self._status.setText(f"Ingesting {len(self._files)} file(s)…")

        from norvel_writer.ingestion.pipeline import IngestPipeline

        files = list(self._files)

        async def _run():
            pipeline = IngestPipeline()
            ok, err = 0, 0
            for i, f in enumerate(files):
                try:
                    await pipeline.run(
                        file_path=f,
                        project_id=self._project_id,
                        doc_type=doc_type,
                    )
                    ok += 1
                except Exception as exc:
                    err += 1
                # Update progress per file
            return ok, err

        def _done(result):
            ok, err = result
            self._status.setText(f"Done: {ok} imported, {err} errors.")
            self._progress.setValue(100)
            self._buttons.setEnabled(True)
            self.accept()

        def _error(exc):
            self._status.setText(f"Error: {exc}")
            self._buttons.setEnabled(True)

        self._worker.run(_run(), on_result=_done, on_error=_error)

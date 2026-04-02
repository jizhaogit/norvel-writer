"""Memory/knowledge base management panel."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from norvel_writer.config.defaults import ALL_SUPPORTED_FORMATS, DOC_TYPES, IMAGE_FORMATS


class MemoryPanel(QDockWidget):
    """
    Panel for managing project memory: ingest files, view documents, delete.
    """

    def __init__(self, project_manager, async_worker, parent=None) -> None:
        super().__init__("Memory / Knowledge Base", parent)
        self._pm = project_manager
        self._worker = async_worker
        self._current_project_id: Optional[str] = None
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Type filter
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._type_filter = QComboBox()
        self._type_filter.addItem("All", None)
        for dt in DOC_TYPES:
            self._type_filter.addItem(dt.replace("_", " ").title(), dt)
        self._type_filter.currentIndexChanged.connect(self._refresh_list)
        filter_row.addWidget(self._type_filter)
        layout.addLayout(filter_row)

        # Document list
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        layout.addWidget(self._list)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("+ Add Files")
        self._btn_add.setObjectName("primary")
        self._btn_add.clicked.connect(self._on_add_files)

        self._btn_delete = QPushButton("Remove")
        self._btn_delete.clicked.connect(self._on_delete)

        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_delete)
        layout.addLayout(btn_row)

        # Doc type selector for import
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Import as:"))
        self._doc_type_combo = QComboBox()
        for dt in DOC_TYPES:
            self._doc_type_combo.addItem(dt.replace("_", " ").title(), dt)
        type_row.addWidget(self._doc_type_combo)
        layout.addLayout(type_row)

        # Image preview
        self._preview = QLabel("")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setFixedHeight(120)
        self._preview.setStyleSheet("border: 1px solid #313244; border-radius: 4px;")
        self._preview.setVisible(False)
        layout.addWidget(self._preview)

        # Status
        self._status = QLabel("")
        self._status.setObjectName("subtitle")
        layout.addWidget(self._status)

        self.setWidget(container)

    def set_project(self, project_id: str) -> None:
        self._current_project_id = project_id
        self._refresh_list()

    def _refresh_list(self) -> None:
        if not self._current_project_id:
            return
        self._list.clear()
        doc_type = self._type_filter.currentData()
        docs = self._pm.list_documents(self._current_project_id, doc_type)
        for doc in docs:
            fmt = doc.get("format", "")
            is_img = f".{fmt}" in IMAGE_FORMATS if fmt else False
            icon_prefix = "🖼 " if is_img else ""
            text = (
                f"{icon_prefix}{doc.get('title') or Path(doc['file_path']).name}"
                f"  [{doc['doc_type']}]  {doc['status']}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, doc["id"])
            item.setData(Qt.ItemDataRole.UserRole + 1, doc["file_path"])
            self._list.addItem(item)
        self._list.currentItemChanged.connect(self._on_item_selected)

    def _on_item_selected(self, current, _previous) -> None:
        """Show thumbnail when an image document is selected."""
        if not current:
            self._preview.setVisible(False)
            return
        file_path = current.data(Qt.ItemDataRole.UserRole + 1)
        if file_path and Path(file_path).suffix.lower() in IMAGE_FORMATS:
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self._preview.width(), 120,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._preview.setPixmap(scaled)
                self._preview.setVisible(True)
                return
        self._preview.setVisible(False)

    def _on_add_files(self) -> None:
        if not self._current_project_id:
            self._status.setText("Select a project first.")
            return

        text_exts = "*.txt *.md *.docx *.pdf *.json"
        img_exts = "*.png *.jpg *.jpeg *.webp *.gif *.bmp"
        ext_filter = (
            f"All supported ({text_exts} {img_exts});;"
            f"Text & Documents ({text_exts});;"
            f"Images — maps & visuals ({img_exts})"
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Files to Memory", "", ext_filter
        )
        if not paths:
            return

        doc_type = self._doc_type_combo.currentData()
        self._status.setText(f"Ingesting {len(paths)} file(s)…")

        from norvel_writer.ingestion.pipeline import IngestPipeline
        from norvel_writer.config.settings import get_config

        vision_model = get_config().vision_model or None

        async def _ingest_all():
            pipeline = IngestPipeline(vision_model=vision_model)
            results = []
            for p in paths:
                try:
                    doc_id = await pipeline.run(
                        file_path=Path(p),
                        project_id=self._current_project_id,
                        doc_type=doc_type,
                    )
                    results.append((p, "ok", doc_id))
                except Exception as exc:
                    results.append((p, "error", str(exc)))
            return results

        def _on_done(results):
            ok = sum(1 for _, s, _ in results if s == "ok")
            err = sum(1 for _, s, _ in results if s == "error")
            self._status.setText(f"Done: {ok} ingested, {err} errors.")
            self._refresh_list()

        def _on_error(exc):
            self._status.setText(f"Error: {exc}")

        self._worker.run(_ingest_all(), on_result=_on_done, on_error=_on_error)

    def _on_delete(self) -> None:
        item = self._list.currentItem()
        if not item:
            return
        doc_id = item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self,
            "Remove Document",
            "Remove this document from memory? The original file is not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._pm.delete_document(doc_id, self._current_project_id)
            self._refresh_list()

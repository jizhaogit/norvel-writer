"""Style profile panel."""
from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class StylePanel(QDockWidget):
    """Shows active style profile and allows building new ones."""

    def __init__(self, project_manager, style_engine, async_worker, parent=None) -> None:
        super().__init__("Style Profile", parent)
        self._pm = project_manager
        self._engine = style_engine
        self._worker = async_worker
        self._current_project_id: Optional[str] = None
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Style Profiles:"))
        self._profile_list = QListWidget()
        self._profile_list.currentItemChanged.connect(self._on_profile_selected)
        layout.addWidget(self._profile_list)

        btn_row = QHBoxLayout()
        self._btn_build = QPushButton("Build from Samples")
        self._btn_build.setObjectName("primary")
        self._btn_build.clicked.connect(self._on_build)

        self._btn_activate = QPushButton("Set Active")
        self._btn_activate.clicked.connect(self._on_activate)

        btn_row.addWidget(self._btn_build)
        btn_row.addWidget(self._btn_activate)
        layout.addLayout(btn_row)

        layout.addWidget(QLabel("Profile details:"))
        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setMaximumHeight(200)
        layout.addWidget(self._details)

        self._status = QLabel("")
        self._status.setObjectName("subtitle")
        layout.addWidget(self._status)

        self.setWidget(container)

    def set_project(self, project_id: str) -> None:
        self._current_project_id = project_id
        self._refresh()

    def _refresh(self) -> None:
        if not self._current_project_id:
            return
        self._profile_list.clear()
        profiles = self._pm.list_style_profiles(self._current_project_id)
        active = self._pm.get_active_style_profile(self._current_project_id)
        active_id = active["id"] if active else None

        for p in profiles:
            label = p["name"]
            if p["id"] == active_id:
                label += " [active]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, p["id"])
            self._profile_list.addItem(item)

    def _on_profile_selected(self, current, _previous) -> None:
        if not current:
            return
        profile_id = current.data(Qt.ItemDataRole.UserRole)
        from norvel_writer.storage.db import get_db
        from norvel_writer.storage.repositories.style_repo import StyleRepo
        repo = StyleRepo(get_db())
        profile = repo.get_style_profile(profile_id)
        if profile:
            try:
                data = json.loads(profile["profile_json"])
                self._details.setPlainText(json.dumps(data, indent=2))
            except Exception:
                self._details.setPlainText(profile["profile_json"])

    def _on_build(self) -> None:
        if not self._current_project_id:
            return
        self._status.setText("Building style profile…")
        self._btn_build.setEnabled(False)

        async def _build():
            return await self._engine.build_profile(
                project_id=self._current_project_id,
                profile_name="Style Profile",
            )

        def _done(profile_id):
            self._pm.set_active_style_profile(self._current_project_id, profile_id)
            self._refresh()
            self._status.setText("Style profile built and activated.")
            self._btn_build.setEnabled(True)

        def _error(exc):
            self._status.setText(f"Error: {exc}")
            self._btn_build.setEnabled(True)

        self._worker.run(_build(), on_result=_done, on_error=_error)

    def _on_activate(self) -> None:
        item = self._profile_list.currentItem()
        if not item:
            return
        profile_id = item.data(Qt.ItemDataRole.UserRole)
        self._pm.set_active_style_profile(self._current_project_id, profile_id)
        self._refresh()
        self._status.setText("Profile activated.")

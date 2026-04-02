"""Project/chapter tree panel."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QDockWidget,
    QInputDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROLE = Qt.ItemDataRole.UserRole + 1
CHAPTER_ROLE = Qt.ItemDataRole.UserRole + 2
ITEM_TYPE_ROLE = Qt.ItemDataRole.UserRole + 3


class ProjectPanel(QDockWidget):
    project_selected = Signal(str)   # project_id
    chapter_selected = Signal(str, str)  # chapter_id, project_id
    new_project_requested = Signal()

    def __init__(self, project_manager, parent=None) -> None:
        super().__init__("Projects", parent)
        self._pm = project_manager
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize().__class__(16, 16))

        act_new = QAction("+ Project", self)
        act_new.triggered.connect(self._on_new_project)
        toolbar.addAction(act_new)

        self._model = QStandardItemModel()
        self._model.setHorizontalHeaderLabels([""])

        self._tree = QTreeView()
        self._tree.setModel(self._model)
        self._tree.setHeaderHidden(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.selectionModel().currentChanged.connect(self._on_selection_changed)
        self._tree.setExpandsOnDoubleClick(True)

        layout.addWidget(toolbar)
        layout.addWidget(self._tree)
        self.setWidget(container)

    def refresh(self) -> None:
        self._model.clear()
        self._model.setHorizontalHeaderLabels([""])
        projects = self._pm.list_projects()
        for proj in projects:
            proj_item = QStandardItem(proj["name"])
            proj_item.setData(proj["id"], PROJECT_ROLE)
            proj_item.setData("project", ITEM_TYPE_ROLE)
            proj_item.setEditable(False)
            proj_item.setToolTip(proj.get("description") or "")

            chapters = self._pm.list_chapters(proj["id"])
            for ch in chapters:
                ch_item = QStandardItem(f"  {ch['title']}")
                ch_item.setData(ch["id"], CHAPTER_ROLE)
                ch_item.setData(proj["id"], PROJECT_ROLE)
                ch_item.setData("chapter", ITEM_TYPE_ROLE)
                ch_item.setEditable(False)
                proj_item.appendRow(ch_item)

            self._model.appendRow(proj_item)
        self._tree.expandAll()

    def _on_selection_changed(self, current, previous) -> None:
        if not current.isValid():
            return
        item = self._model.itemFromIndex(current)
        item_type = item.data(ITEM_TYPE_ROLE)
        if item_type == "project":
            self.project_selected.emit(item.data(PROJECT_ROLE))
        elif item_type == "chapter":
            self.chapter_selected.emit(
                item.data(CHAPTER_ROLE), item.data(PROJECT_ROLE)
            )

    def _on_new_project(self) -> None:
        from norvel_writer.ui.dialogs.new_project_dialog import NewProjectDialog
        dlg = NewProjectDialog(self)
        if dlg.exec() == NewProjectDialog.DialogCode.Accepted:
            self._pm.create_project(
                name=dlg.project_name,
                description=dlg.description,
                language=dlg.language_code,
            )
            self.refresh()

    def _on_context_menu(self, pos) -> None:
        index = self._tree.indexAt(pos)
        if not index.isValid():
            return
        item = self._model.itemFromIndex(index)
        item_type = item.data(ITEM_TYPE_ROLE)

        menu = QMenu(self)

        if item_type == "project":
            project_id = item.data(PROJECT_ROLE)
            act_chapter = menu.addAction("Add Chapter")
            act_chapter.triggered.connect(lambda: self._add_chapter(project_id))
            menu.addSeparator()
            act_delete = menu.addAction("Delete Project…")
            act_delete.triggered.connect(lambda: self._delete_project(project_id, item.text()))

        elif item_type == "chapter":
            chapter_id = item.data(CHAPTER_ROLE)
            project_id = item.data(PROJECT_ROLE)
            act_rename = menu.addAction("Rename")
            act_rename.triggered.connect(lambda: self._rename_chapter(chapter_id, item.text()))
            menu.addSeparator()
            act_delete = menu.addAction("Delete Chapter…")
            act_delete.triggered.connect(lambda: self._delete_chapter(chapter_id))

        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _add_chapter(self, project_id: str) -> None:
        title, ok = QInputDialog.getText(self, "New Chapter", "Chapter title:")
        if ok and title.strip():
            self._pm.create_chapter(project_id, title.strip())
            self.refresh()

    def _delete_project(self, project_id: str, name: str) -> None:
        reply = QMessageBox.question(
            self,
            "Delete Project",
            f"Delete project '{name}' and all its data? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._pm.delete_project(project_id)
            self.refresh()

    def _rename_chapter(self, chapter_id: str, current_title: str) -> None:
        title, ok = QInputDialog.getText(
            self, "Rename Chapter", "New title:", text=current_title.strip()
        )
        if ok and title.strip():
            self._pm.update_chapter(chapter_id, title=title.strip())
            self.refresh()

    def _delete_chapter(self, chapter_id: str) -> None:
        reply = QMessageBox.question(
            self,
            "Delete Chapter",
            "Delete this chapter and all its drafts?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._pm.delete_chapter(chapter_id)
            self.refresh()

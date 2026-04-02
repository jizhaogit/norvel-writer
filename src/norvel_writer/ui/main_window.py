"""Main application window."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from norvel_writer.core.draft_engine import DraftEngine
from norvel_writer.core.export_engine import DocxExporter, MarkdownExporter, NotebookLMExporter
from norvel_writer.core.project import ProjectManager
from norvel_writer.core.style_profile import StyleProfileEngine
from norvel_writer.ui.panels.chat_panel import ChatPanel
from norvel_writer.ui.panels.draft_panel import DraftPanel
from norvel_writer.ui.panels.editor_panel import EditorPanel
from norvel_writer.ui.panels.memory_panel import MemoryPanel
from norvel_writer.ui.panels.project_panel import ProjectPanel
from norvel_writer.ui.panels.style_panel import StylePanel
from norvel_writer.utils.async_worker import AsyncWorker

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Norvel Writer")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)

        # Core singletons
        self._worker = AsyncWorker.instance()
        self._pm = ProjectManager()
        self._draft_engine = DraftEngine(project_manager=self._pm)
        self._style_engine = StyleProfileEngine()

        self._current_project_id: Optional[str] = None
        self._current_chapter_id: Optional[str] = None

        self._build_menu()
        self._build_panels()
        self._build_status_bar()
        self._restore_last_project()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_panels(self) -> None:
        # Central editor
        self._editor = EditorPanel(self._pm, self)
        self.setCentralWidget(self._editor)

        # Project panel (left)
        self._project_panel = ProjectPanel(self._pm, self)
        self._project_panel.project_selected.connect(self._on_project_selected)
        self._project_panel.chapter_selected.connect(self._on_chapter_selected)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._project_panel)

        # Memory panel (left, tabbed with project)
        self._memory_panel = MemoryPanel(self._pm, self._worker, self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._memory_panel)
        self.tabifyDockWidget(self._project_panel, self._memory_panel)

        # Draft panel (right)
        self._draft_panel = DraftPanel(
            self._pm, self._draft_engine, self._worker, self
        )
        self._draft_panel.insert_into_editor.connect(self._editor.insert_text)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._draft_panel)

        # Style panel (right, tabbed with draft)
        self._style_panel = StylePanel(self._pm, self._style_engine, self._worker, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._style_panel)
        self.tabifyDockWidget(self._draft_panel, self._style_panel)

        # Chat / Q&A panel (right, tabbed with draft and style)
        self._chat_panel = ChatPanel(self._pm, self._draft_engine, self._worker, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._chat_panel)
        self.tabifyDockWidget(self._style_panel, self._chat_panel)

        # Show draft panel by default
        self._draft_panel.raise_()
        self._project_panel.raise_()

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        act_new_project = QAction("New Project…", self)
        act_new_project.setShortcut("Ctrl+Shift+N")
        act_new_project.triggered.connect(self._new_project)
        file_menu.addAction(act_new_project)

        file_menu.addSeparator()

        act_export = QAction("Export…", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._export)
        file_menu.addAction(act_export)

        act_import_edit = QAction("Import Externally Edited File…", self)
        act_import_edit.triggered.connect(self._import_external_edit)
        file_menu.addAction(act_import_edit)

        file_menu.addSeparator()

        act_settings = QAction("Settings…", self)
        act_settings.triggered.connect(self._open_settings)
        file_menu.addAction(act_settings)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(act_quit)

        # Edit
        edit_menu = menubar.addMenu("&Edit")

        act_continuity = QAction("Check Continuity", self)
        act_continuity.triggered.connect(self._check_continuity)
        edit_menu.addAction(act_continuity)

        act_summarise = QAction("Summarise Chapter", self)
        act_summarise.triggered.connect(self._summarise_chapter)
        edit_menu.addAction(act_summarise)

        # View
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self._project_panel.toggleViewAction())
        view_menu.addAction(self._memory_panel.toggleViewAction())
        view_menu.addAction(self._draft_panel.toggleViewAction())
        view_menu.addAction(self._style_panel.toggleViewAction())
        view_menu.addAction(self._chat_panel.toggleViewAction())

        # Help
        help_menu = menubar.addMenu("&Help")
        act_about = QAction("About Norvel Writer", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _build_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

        # Ollama status check
        QTimer.singleShot(1000, self._check_ollama_status)

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_project_selected(self, project_id: str) -> None:
        self._current_project_id = project_id
        self._memory_panel.set_project(project_id)
        self._style_panel.set_project(project_id)
        self._chat_panel.set_project(project_id)
        project = self._pm.get_project(project_id)
        if project:
            self.setWindowTitle(f"Norvel Writer — {project['name']}")
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        cfg.last_opened_project_id = project_id
        cfg.save()

    def _on_chapter_selected(self, chapter_id: str, project_id: str) -> None:
        self._current_chapter_id = chapter_id
        self._current_project_id = project_id
        self._editor.load_chapter(chapter_id, project_id)
        self._draft_panel.set_context(
            chapter_id, project_id, self._editor.get_content
        )
        chapter = self._pm.get_chapter(chapter_id)
        if chapter:
            self._status_bar.showMessage(f"Chapter: {chapter['title']}")

    # ── Actions ───────────────────────────────────────────────────────────

    def _new_project(self) -> None:
        from norvel_writer.ui.dialogs.new_project_dialog import NewProjectDialog
        dlg = NewProjectDialog(self)
        if dlg.exec() == NewProjectDialog.DialogCode.Accepted:
            pid = self._pm.create_project(
                name=dlg.project_name,
                description=dlg.description,
                language=dlg.language_code,
            )
            self._project_panel.refresh()

    def _export(self) -> None:
        if not self._current_project_id:
            QMessageBox.information(self, "Export", "Please select a project first.")
            return
        from norvel_writer.ui.dialogs.export_dialog import ExportDialog
        project = self._pm.get_project(self._current_project_id)
        dlg = ExportDialog(project["name"] if project else "Project", self)
        if dlg.exec() != ExportDialog.DialogCode.Accepted:
            return
        dest = dlg.destination
        if not dest:
            return
        fmt = dlg.selected_format
        try:
            if fmt == "md":
                MarkdownExporter().export(self._current_project_id, dest)
            elif fmt == "docx":
                DocxExporter().export(self._current_project_id, dest)
            elif fmt == "notebooklm":
                NotebookLMExporter().export(self._current_project_id, dest)
                if dlg.open_in_notebooklm:
                    NotebookLMExporter().open_in_browser()
            QMessageBox.information(self, "Export", f"Exported to:\n{dest}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _import_external_edit(self) -> None:
        if not self._current_chapter_id:
            QMessageBox.information(self, "Import", "Please select a chapter first.")
            return
        from PySide6.QtWidgets import QFileDialog
        from norvel_writer.core.diff_engine import DiffEngine
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Edited File", "", "Text files (*.md *.txt)"
        )
        if not path:
            return
        try:
            engine = DiffEngine()
            diff_chunks = engine.import_edited(
                self._current_chapter_id, file_path=Path(path)
            )
            content = Path(path).read_text(encoding="utf-8")
            self._editor.set_content(content)
            # Save as new draft
            draft_id = self._pm.save_draft(
                chapter_id=self._current_chapter_id,
                content=content,
                model_used="external_edit",
            )
            self._pm.accept_draft(draft_id)
            changed = sum(1 for c in diff_chunks if c.tag != "equal")
            QMessageBox.information(
                self, "Import Complete",
                f"Imported. {changed} diff regions detected.\n"
                "The revised text is now the active draft."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))

    def _check_continuity(self) -> None:
        if not self._current_chapter_id or not self._current_project_id:
            return
        passage = self._editor.get_content()
        if not passage.strip():
            return

        self._status_bar.showMessage("Checking continuity…")

        async def _check():
            return await self._draft_engine.check_continuity(
                project_id=self._current_project_id,
                passage=passage,
            )

        def _done(result):
            self._status_bar.showMessage("Continuity check complete.")
            QMessageBox.information(self, "Continuity Check", result)

        self._worker.run(_check(), on_result=_done)

    def _summarise_chapter(self) -> None:
        if not self._current_chapter_id:
            return
        text = self._editor.get_content()
        if not text.strip():
            return
        self._status_bar.showMessage("Summarising…")

        async def _summarise():
            return await self._draft_engine.summarise_chapter(text)

        def _done(summary):
            self._pm.update_chapter(self._current_chapter_id, summary=summary)
            self._status_bar.showMessage("Summary saved.")
            QMessageBox.information(self, "Chapter Summary", summary)

        self._worker.run(_summarise(), on_result=_done)

    def _open_settings(self) -> None:
        from norvel_writer.ui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        dlg.exec()

    def _about(self) -> None:
        from norvel_writer import __version__
        QMessageBox.about(
            self,
            "About Norvel Writer",
            f"<h3>Norvel Writer</h3>"
            f"<p>Version {__version__}</p>"
            f"<p>Local-first writing assistant powered by Ollama.</p>"
            f"<p>Your writing stays on your device.</p>",
        )

    def _restore_last_project(self) -> None:
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        if cfg.last_opened_project_id:
            project = self._pm.get_project(cfg.last_opened_project_id)
            if project:
                self._on_project_selected(cfg.last_opened_project_id)

    def _check_ollama_status(self) -> None:
        from norvel_writer.llm.ollama_client import get_client

        async def _ping():
            return await get_client().ping()

        def _done(ok):
            if ok:
                self._status_bar.showMessage("Ollama: connected", 3000)
            else:
                self._status_bar.showMessage(
                    "Ollama: not reachable — check that Ollama is running"
                )

        self._worker.run(_ping(), on_result=_done)

    def closeEvent(self, event) -> None:
        self._worker.shutdown()
        event.accept()

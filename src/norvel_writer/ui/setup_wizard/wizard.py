"""First-run setup wizard."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWizard

from norvel_writer.ui.setup_wizard.page_welcome import WelcomePage
from norvel_writer.ui.setup_wizard.page_ollama import OllamaPage
from norvel_writer.ui.setup_wizard.page_models import ModelsPage
from norvel_writer.ui.setup_wizard.page_finish import FinishPage


class SetupWizard(QWizard):
    """Guides the user through first-run setup: Ollama detection, model pull."""

    PAGE_WELCOME = 0
    PAGE_OLLAMA = 1
    PAGE_MODELS = 2
    PAGE_FINISH = 3

    def __init__(self, async_worker, parent=None) -> None:
        super().__init__(parent)
        self._worker = async_worker
        self.setWindowTitle("Norvel Writer — Setup")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(640, 480)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)

        self.setPage(self.PAGE_WELCOME, WelcomePage(self))
        self.setPage(self.PAGE_OLLAMA, OllamaPage(async_worker, self))
        self.setPage(self.PAGE_MODELS, ModelsPage(async_worker, self))
        self.setPage(self.PAGE_FINISH, FinishPage(self))

    def accept(self) -> None:
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        cfg.first_run_complete = True
        cfg.save()
        super().accept()

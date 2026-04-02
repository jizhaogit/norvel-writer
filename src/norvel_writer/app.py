"""Application bootstrap: QApplication setup, theme, first-run check."""
from __future__ import annotations

import sys
import logging

log = logging.getLogger(__name__)


def run() -> int:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    # Must create QApplication before importing Qt widgets
    app = QApplication(sys.argv)
    app.setApplicationName("NorvelWriter")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("NorvelWriter")

    # High-DPI
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Bootstrap config and directories
    from norvel_writer.config.settings import get_config
    cfg = get_config()
    cfg.ensure_dirs()

    # Logging
    from norvel_writer.utils.logging_config import setup_logging
    setup_logging(cfg.logs_path)

    # Database
    from norvel_writer.storage.db import init_db
    init_db(cfg.db_path)

    # Apply theme
    from norvel_writer.ui.theme import apply_theme
    apply_theme(app, cfg.theme)

    # Pre-warm AsyncWorker
    from norvel_writer.utils.async_worker import AsyncWorker
    AsyncWorker.instance()

    # First-run wizard or main window
    if not cfg.first_run_complete:
        from norvel_writer.ui.setup_wizard.wizard import SetupWizard
        worker = AsyncWorker.instance()
        wizard = SetupWizard(worker)
        result = wizard.exec()
        if result != SetupWizard.DialogCode.Accepted:
            return 0

    from norvel_writer.ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    return app.exec()

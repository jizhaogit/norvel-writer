from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWizardPage


class WelcomePage(QWizardPage):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Welcome to Norvel Writer")
        self.setSubTitle(
            "Your local-first writing assistant powered by Ollama.\n"
            "This wizard will help you get set up in a few steps."
        )
        layout = QVBoxLayout(self)
        info = QLabel(
            "<p>Norvel Writer runs entirely on your machine. "
            "Your writing never leaves your device unless you explicitly share it.</p>"
            "<p>You will need:</p>"
            "<ul>"
            "<li><b>Ollama</b> installed (free, open-source)</li>"
            "<li>At least one AI model downloaded (~1–8 GB depending on model)</li>"
            "</ul>"
            "<p>Click <b>Next</b> to check your Ollama installation.</p>"
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(info)

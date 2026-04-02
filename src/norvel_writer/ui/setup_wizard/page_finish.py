from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWizardPage


class FinishPage(QWizardPage):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("You're ready!")
        self.setSubTitle("Norvel Writer is configured and ready to use.")
        layout = QVBoxLayout(self)
        info = QLabel(
            "<p>Setup is complete. Here's what to do next:</p>"
            "<ol>"
            "<li><b>Create a project</b> from the Projects panel (left sidebar).</li>"
            "<li><b>Add chapters</b> by right-clicking your project.</li>"
            "<li><b>Import your reference files</b> (codex, beats, notes) "
            "from the Memory panel.</li>"
            "<li>Click <b>Continue Draft</b> in the AI Assistant panel to start writing.</li>"
            "</ol>"
            "<p>Click <b>Finish</b> to open the app.</p>"
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(info)

"""New project dialog — name, description, language."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
)

from norvel_writer.config.defaults import LANGUAGES, language_display


class NewProjectDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(420)
        self._build_ui()
        self._load_defaults()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._name = QLineEdit()
        self._name.setPlaceholderText("My Novel")
        form.addRow("Project name *:", self._name)

        self._description = QTextEdit()
        self._description.setPlaceholderText("Optional short description…")
        self._description.setMaximumHeight(70)
        form.addRow("Description:", self._description)

        self._language = QComboBox()
        for code in LANGUAGES:
            self._language.addItem(language_display(code), code)
        lang_note = QLabel("Sets the default writing language for this project.")
        lang_note.setObjectName("subtitle")
        lang_note.setWordWrap(True)
        form.addRow("Writing language:", self._language)
        form.addRow("", lang_note)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_defaults(self) -> None:
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        # Pre-select the configured default project language
        for i in range(self._language.count()):
            if self._language.itemData(i) == cfg.default_project_language:
                self._language.setCurrentIndex(i)
                break

    def _on_accept(self) -> None:
        if not self._name.text().strip():
            self._name.setPlaceholderText("Name is required!")
            return
        self.accept()

    @property
    def project_name(self) -> str:
        return self._name.text().strip()

    @property
    def description(self) -> str:
        return self._description.toPlainText().strip()

    @property
    def language_code(self) -> str:
        return self._language.currentData()

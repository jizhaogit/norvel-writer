"""Wizard page: detect and start Ollama."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWizardPage,
    QHBoxLayout,
    QProgressBar,
)


class OllamaPage(QWizardPage):
    def __init__(self, async_worker, parent=None) -> None:
        super().__init__(parent)
        self._worker = async_worker
        self._ready = False
        self.setTitle("Ollama Setup")
        self.setSubTitle("Norvel Writer uses Ollama to run AI models locally.")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._status_label = QLabel("Checking Ollama…")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedHeight(8)
        layout.addWidget(self._progress)

        btn_row = QHBoxLayout()
        self._btn_download = QPushButton("Download Ollama")
        self._btn_download.setVisible(False)
        self._btn_download.clicked.connect(self._open_download)

        self._btn_retry = QPushButton("Retry")
        self._btn_retry.setVisible(False)
        self._btn_retry.clicked.connect(self._check)

        self._btn_start = QPushButton("Start Ollama")
        self._btn_start.setVisible(False)
        self._btn_start.clicked.connect(self._start_ollama)

        btn_row.addWidget(self._btn_download)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_retry)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def initializePage(self) -> None:
        self._check()

    def _check(self) -> None:
        self._btn_download.setVisible(False)
        self._btn_retry.setVisible(False)
        self._btn_start.setVisible(False)
        self._progress.setRange(0, 0)
        self._status_label.setText("Checking Ollama…")
        self._ready = False
        self.completeChanged.emit()

        from norvel_writer.llm.model_manager import get_ollama_status

        def _done(status):
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            if not status.installed:
                self._status_label.setText(
                    "Ollama is not installed. Please download and install it, "
                    "then click Retry."
                )
                self._btn_download.setVisible(True)
                self._btn_retry.setVisible(True)
            elif not status.running:
                self._status_label.setText(
                    "Ollama is installed but not running. "
                    "Click 'Start Ollama' to launch it."
                )
                self._btn_start.setVisible(True)
                self._btn_retry.setVisible(True)
            else:
                model_count = len(status.models)
                self._status_label.setText(
                    f"Ollama is running. {model_count} model(s) installed.\n"
                    f"Version: {status.version}"
                )
                self._ready = True
                self.completeChanged.emit()

        def _error(exc):
            self._progress.setRange(0, 1)
            self._status_label.setText(f"Error checking Ollama: {exc}")
            self._btn_retry.setVisible(True)

        self._worker.run(get_ollama_status(), on_result=_done, on_error=_error)

    def _open_download(self) -> None:
        from norvel_writer.llm.model_manager import launch_ollama_download_page
        launch_ollama_download_page()

    def _start_ollama(self) -> None:
        self._status_label.setText("Starting Ollama…")
        self._progress.setRange(0, 0)

        from norvel_writer.llm.model_manager import start_ollama_serve

        def _done(ok):
            if ok:
                self._check()
            else:
                self._status_label.setText(
                    "Could not start Ollama automatically. "
                    "Please start it manually, then click Retry."
                )
                self._btn_retry.setVisible(True)

        self._worker.run(start_ollama_serve(), on_result=_done)

    def isComplete(self) -> bool:
        return self._ready

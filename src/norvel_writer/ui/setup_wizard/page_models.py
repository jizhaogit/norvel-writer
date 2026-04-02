"""Wizard page: select and pull models."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWizardPage,
)

from norvel_writer.config.defaults import (
    DEFAULT_EMBED_MODEL,
    FALLBACK_CHAT_MODELS,
    RECOMMENDED_EMBED_MODEL,
)


class ModelsPage(QWizardPage):
    def __init__(self, async_worker, parent=None) -> None:
        super().__init__(parent)
        self._worker = async_worker
        self._ready = False
        self.setTitle("Model Selection")
        self.setSubTitle(
            "Choose which AI models to use. Models are downloaded once and stored locally."
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Chat model
        layout.addWidget(QLabel("<b>Chat model</b> (for writing):"))
        chat_row = QHBoxLayout()
        self._chat_combo = QComboBox()
        self._chat_combo.setEditable(True)
        for m in FALLBACK_CHAT_MODELS:
            self._chat_combo.addItem(m)
        chat_row.addWidget(self._chat_combo)
        self._btn_pull_chat = QPushButton("Download")
        self._btn_pull_chat.clicked.connect(self._pull_chat)
        chat_row.addWidget(self._btn_pull_chat)
        layout.addLayout(chat_row)
        self._chat_status = QLabel("")
        self._chat_status.setObjectName("subtitle")
        layout.addWidget(self._chat_status)

        # Embed model
        layout.addWidget(QLabel(f"<b>Embedding model</b> (recommended: {RECOMMENDED_EMBED_MODEL}):"))
        embed_row = QHBoxLayout()
        self._embed_combo = QComboBox()
        self._embed_combo.setEditable(True)
        self._embed_combo.addItem(DEFAULT_EMBED_MODEL)
        embed_row.addWidget(self._embed_combo)
        self._btn_pull_embed = QPushButton("Download")
        self._btn_pull_embed.clicked.connect(self._pull_embed)
        embed_row.addWidget(self._btn_pull_embed)
        layout.addLayout(embed_row)
        self._embed_status = QLabel("")
        self._embed_status.setObjectName("subtitle")
        layout.addWidget(self._embed_status)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._main_status = QLabel("")
        self._main_status.setWordWrap(True)
        layout.addWidget(self._main_status)

        self._btn_skip = QPushButton("Skip — I'll download models later")
        self._btn_skip.clicked.connect(self._skip)
        layout.addWidget(self._btn_skip)

        layout.addStretch()

    def initializePage(self) -> None:
        self._refresh_installed()

    def _refresh_installed(self) -> None:
        from norvel_writer.llm.ollama_client import get_client

        async def _list():
            return await get_client().list_models()

        def _done(models):
            names = {m.name for m in models}
            names.update({m.name.split(":")[0] for m in models})

            chat = self._chat_combo.currentText()
            if chat in names or chat.split(":")[0] in names:
                self._chat_status.setText("✓ Installed")
                self._chat_status.setStyleSheet("color: #a6e3a1;")
            else:
                self._chat_status.setText("Not installed — click Download")

            embed = self._embed_combo.currentText()
            if embed in names or embed.split(":")[0] in names:
                self._embed_status.setText("✓ Installed")
                self._embed_status.setStyleSheet("color: #a6e3a1;")
                self._ready = True
                self.completeChanged.emit()
            else:
                self._embed_status.setText("Not installed — click Download")

        self._worker.run(_list(), on_result=_done)

    def _pull_chat(self) -> None:
        model = self._chat_combo.currentText().strip()
        if not model:
            return
        self._main_status.setText(f"Downloading {model}…")
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_pull_chat.setEnabled(False)

        from norvel_writer.llm.model_manager import ensure_model
        from norvel_writer.config.settings import get_config

        def _progress_cb(pct: int) -> None:
            self._progress.setValue(pct)

        async def _pull():
            ok = await ensure_model(model, progress_cb=_progress_cb)
            return ok

        def _done(ok):
            self._btn_pull_chat.setEnabled(True)
            self._progress.setVisible(False)
            if ok:
                self._chat_status.setText("✓ Installed")
                self._chat_status.setStyleSheet("color: #a6e3a1;")
                self._main_status.setText(f"{model} ready.")
                # Save to config
                cfg = get_config()
                cfg.default_chat_model = model
                cfg.save()
                self._check_complete()
            else:
                self._main_status.setText(
                    f"Could not download {model}. "
                    "Make sure Ollama is running (check the system tray), then try again. "
                    "Or click Skip to set up models later."
                )

        self._worker.run(_pull(), on_result=_done)

    def _pull_embed(self) -> None:
        model = self._embed_combo.currentText().strip()
        if not model:
            return
        self._main_status.setText(f"Downloading {model}…")
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_pull_embed.setEnabled(False)

        from norvel_writer.llm.model_manager import ensure_model
        from norvel_writer.config.settings import get_config

        def _progress_cb(pct: int) -> None:
            self._progress.setValue(pct)

        async def _pull():
            return await ensure_model(model, progress_cb=_progress_cb)

        def _done(ok):
            self._btn_pull_embed.setEnabled(True)
            self._progress.setVisible(False)
            if ok:
                self._embed_status.setText("✓ Installed")
                self._embed_status.setStyleSheet("color: #a6e3a1;")
                self._main_status.setText(f"{model} ready.")
                cfg = get_config()
                cfg.default_embed_model = model
                cfg.save()
                self._ready = True
                self.completeChanged.emit()
            else:
                self._main_status.setText(
                    f"Could not download {model}. "
                    "Make sure Ollama is running (check the system tray), then try again. "
                    "Or click Skip to set up models later."
                )

        self._worker.run(_pull(), on_result=_done)

    def _skip(self) -> None:
        self._main_status.setText(
            "Skipped. You can download models later via File → Settings "
            "or by running: ollama pull gemma3:4b && ollama pull nomic-embed-text"
        )
        self._ready = True
        self.completeChanged.emit()

    def _check_complete(self) -> None:
        self._ready = True
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._ready

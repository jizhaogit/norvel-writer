"""Ollama model detection, listing, and pull management."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from norvel_writer.llm.ollama_client import ModelInfo, OllamaConnectionError, get_client

log = logging.getLogger(__name__)

OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"


@dataclass
class OllamaStatus:
    installed: bool = False
    running: bool = False
    version: str = ""
    models: List[ModelInfo] = field(default_factory=list)


async def get_ollama_status() -> OllamaStatus:
    status = OllamaStatus()

    # Check if ollama binary is on PATH
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            status.installed = True
            status.version = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not status.installed:
        return status

    # Check if service is reachable
    client = get_client()
    status.running = await client.ping()
    if status.running:
        try:
            status.models = await client.list_models()
        except OllamaConnectionError:
            status.running = False

    return status


async def ensure_model(
    model_name: str,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> bool:
    """Pull model if not present. Returns True if model is ready."""
    client = get_client()
    try:
        models = await client.list_models()
        model_names = {m.name for m in models}
        # Also check short names (without :latest suffix)
        model_names.update({m.name.split(":")[0] for m in models})
        if model_name in model_names or model_name.split(":")[0] in model_names:
            log.info("Model %r already installed", model_name)
            return True
    except OllamaConnectionError:
        return False

    log.info("Pulling model %r...", model_name)
    try:
        await client.pull_model(model_name, progress_cb)
        return True
    except Exception as exc:
        log.error("Failed to pull model %r: %s", model_name, exc)
        return False


def launch_ollama_download_page() -> None:
    """Open the Ollama download page in the default browser."""
    import webbrowser
    webbrowser.open(OLLAMA_DOWNLOAD_URL)


async def start_ollama_serve() -> bool:
    """Attempt to start 'ollama serve' as a background process."""
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        # Give it a moment to start
        for _ in range(10):
            await asyncio.sleep(0.5)
            client = get_client()
            if await client.ping():
                return True
        return False
    except Exception as exc:
        log.warning("Could not start ollama serve: %s", exc)
        return False

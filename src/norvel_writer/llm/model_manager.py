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

    # Check if ollama binary is on PATH (optional — only for version string).
    # On Windows, Ollama often runs as a tray app and is NOT on PATH even when
    # the service is fully reachable, so we never bail out early here.
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

    # Always check the HTTP service, regardless of binary availability.
    # The service may be running even when the CLI is not on PATH.
    client = get_client()
    status.running = await client.ping()
    if status.running:
        status.installed = True  # service is up → treat as installed
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
    """Attempt to start Ollama as a background process.

    On Windows, tries the Ollama tray app (Ollama.exe) first, which is the
    standard installation. Falls back to 'ollama serve' on other platforms.
    """
    no_window = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    if sys.platform == "win32":
        # Windows: launch the Ollama tray app from common install locations
        import os
        candidates = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\Ollama.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\Ollama\Ollama.exe"),
        ]
        launched = False
        for exe in candidates:
            if os.path.exists(exe):
                try:
                    subprocess.Popen(
                        [exe],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=no_window,
                    )
                    launched = True
                    log.info("Launched Ollama tray app: %s", exe)
                    break
                except Exception as exc:
                    log.warning("Could not launch %s: %s", exe, exc)

        if not launched:
            # Fall back to ollama serve
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=no_window,
                )
                launched = True
            except Exception as exc:
                log.warning("Could not start ollama serve: %s", exc)
                return False
    else:
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            log.warning("Could not start ollama serve: %s", exc)
            return False

    # Wait up to 15 seconds for the service to respond
    client = get_client()
    for _ in range(30):
        await asyncio.sleep(0.5)
        if await client.ping():
            return True
    return False

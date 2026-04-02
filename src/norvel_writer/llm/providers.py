"""LLM provider manager: reads llm.ini and routes calls to the right backend."""
from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── INI discovery ─────────────────────────────────────────────────────────────

_INI_FILENAME = "llm.ini"
_EXAMPLE_FILENAME = "llm.ini.example"


def _config_dir() -> Path:
    from platformdirs import user_config_dir
    from norvel_writer.config.defaults import APP_NAME, APP_AUTHOR
    return Path(user_config_dir(APP_NAME, APP_AUTHOR))


def _app_root() -> Path:
    """Return the project root (where run.bat lives)."""
    return Path(__file__).parent.parent.parent.parent.parent


def find_ini_path() -> Optional[Path]:
    """Return the first llm.ini found, or None."""
    candidates = [
        _config_dir() / _INI_FILENAME,
        _app_root() / _INI_FILENAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def ensure_ini_exists() -> Path:
    """
    Create llm.ini in the user config dir from the bundled example if it
    does not already exist.  Returns the path to the (possibly new) file.
    """
    dest = _config_dir() / _INI_FILENAME
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Try to copy from the project root example
    example = _app_root() / _EXAMPLE_FILENAME
    if example.exists():
        dest.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        log.info("Created llm.ini at %s", dest)
    else:
        # Write the minimal default inline
        dest.write_text(_DEFAULT_INI, encoding="utf-8")
        log.info("Created default llm.ini at %s", dest)

    return dest


_DEFAULT_INI = """\
# ══════════════════════════════════════════════════════════════════════════════
# Norvel Writer — LLM configuration  (llm.ini)
# ══════════════════════════════════════════════════════════════════════════════
#
# This file controls which AI backend Norvel Writer uses.
# Edit it with any text editor, then restart the app (or click Save in
# Settings → LLM Config) for changes to take effect.
#
# Supported providers
#   ollama    — local models via Ollama  (free, offline, private)
#   openai    — OpenAI API              (requires API key, sends data online)
#   anthropic — Anthropic Claude API    (requires API key, sends data online)
#   gemini    — Google Gemini API       (requires API key, sends data online)
#
# ══════════════════════════════════════════════════════════════════════════════

[provider]
# Which backend to use for chat/writing generation.
# Valid values: ollama | openai | anthropic | gemini
chat = ollama

# Which backend to use for embeddings (RAG / similarity search).
# Currently only "ollama" and "openai" support embeddings.
# If you use a cloud provider for chat but want free local embeddings,
# set chat = anthropic (or gemini) and embeddings = ollama.
embeddings = ollama


# ──────────────────────────────────────────────────────────────────────────────
# [ollama]  Local models via Ollama  (https://ollama.com)
# ──────────────────────────────────────────────────────────────────────────────
# Install Ollama from https://ollama.com/download, then pull models with:
#
#   ollama pull gemma3:4b          # recommended chat model (~2.5 GB)
#   ollama pull nomic-embed-text   # required for embeddings (~274 MB)
#
# Other good chat models to try:
#   gemma3:1b      (lightweight, ~800 MB)
#   llama3.2:3b    (Meta, ~2 GB)
#   llama3.1:8b    (Meta, ~4.7 GB)
#   mistral:7b     (Mistral AI, ~4.1 GB)
#   phi3:mini      (Microsoft, ~2.3 GB)
#
# Vision models (for image description, optional):
#   llava:7b           (~4.5 GB)
#   llama3.2-vision    (~7.9 GB)
#   moondream          (~1.7 GB)
#
[ollama]
# URL of the Ollama HTTP service.  Change only if you run Ollama on a
# non-standard port or on a remote machine.
base_url     = http://127.0.0.1:11434

# Model used for all writing/chat generation.
chat_model   = gemma3:4b

# Model used for embedding (RAG).  nomic-embed-text is strongly recommended.
embed_model  = nomic-embed-text

# Optional vision model for image description.  Leave blank to disable.
vision_model =


# ──────────────────────────────────────────────────────────────────────────────
# [openai]  OpenAI API  (https://platform.openai.com)
# ──────────────────────────────────────────────────────────────────────────────
# Get your API key at: https://platform.openai.com/api-keys
# To use OpenAI set [provider] chat = openai
#
# Chat model options (as of 2025):
#   gpt-4o             (most capable)
#   gpt-4o-mini        (fast & cheap, recommended default)
#   gpt-4-turbo
#   o3-mini            (reasoning model)
#
# Embedding model options:
#   text-embedding-3-small   (default, cheap)
#   text-embedding-3-large   (higher quality)
#
# base_url: leave blank for the official OpenAI endpoint.
#           Set to a custom URL to use any OpenAI-compatible API
#           (e.g. LM Studio, vLLM, Together AI, Groq, Fireworks …).
#
# use_assistant: set to true to use the Assistants API instead of Chat
#                Completions (advanced — useful for persistent threads).
#
[openai]
api_key       =
chat_model    = gpt-4o-mini
embed_model   = text-embedding-3-small
base_url      =
use_assistant = false


# ──────────────────────────────────────────────────────────────────────────────
# [anthropic]  Anthropic Claude API  (https://www.anthropic.com)
# ──────────────────────────────────────────────────────────────────────────────
# Get your API key at: https://console.anthropic.com
# To use Anthropic set [provider] chat = anthropic
#
# Note: Anthropic does NOT provide an embeddings API.
#       Keep [provider] embeddings = ollama when using Anthropic for chat.
#
# Chat model options (as of 2025):
#   claude-3-5-haiku-20241022    (fast & cheap, recommended default)
#   claude-3-5-sonnet-20241022   (balanced)
#   claude-3-opus-20240229       (most capable)
#
[anthropic]
api_key    =
chat_model = claude-3-5-haiku-20241022


# ──────────────────────────────────────────────────────────────────────────────
# [gemini]  Google Gemini API  (https://ai.google.dev)
# ──────────────────────────────────────────────────────────────────────────────
# Get your API key at: https://aistudio.google.com/app/apikey
# To use Gemini set [provider] chat = gemini
#
# Note: Gemini embeddings are not yet integrated.
#       Keep [provider] embeddings = ollama when using Gemini for chat.
#
# Chat model options (as of 2025):
#   gemini-1.5-flash     (fast & cheap, recommended default)
#   gemini-1.5-pro       (most capable)
#   gemini-2.0-flash     (next-gen flash)
#
[gemini]
api_key    =
chat_model = gemini-1.5-flash
"""


def read_ini() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    path = find_ini_path()
    if path:
        cfg.read(path, encoding="utf-8")
    return cfg


def get_section(name: str) -> dict:
    cfg = read_ini()
    return dict(cfg[name]) if cfg.has_section(name) else {}


# ── Public helpers ────────────────────────────────────────────────────────────

def chat_provider() -> str:
    return read_ini().get("provider", "chat", fallback="ollama").lower()


def embeddings_provider() -> str:
    return read_ini().get("provider", "embeddings", fallback="ollama").lower()


# ── ProviderRouter ────────────────────────────────────────────────────────────

class ProviderRouter:
    """
    Drop-in replacement for OllamaClient.
    Reads llm.ini on every call (cached per-process via read_ini) and
    dispatches to the configured provider.  The 'model' parameter that
    callers pass is ignored — the INI-configured model is always used.
    """

    # ── Chat ──────────────────────────────────────────────────────────────────

    def _chat_backend(self) -> Tuple[Any, str]:
        """Return (client, model) for the active chat provider."""
        provider = chat_provider()
        cfg = get_section(provider)

        if provider == "ollama":
            from norvel_writer.llm.ollama_client import OllamaClient
            from norvel_writer.config.settings import get_config
            app_cfg = get_config()
            url = cfg.get("base_url", app_cfg.ollama_base_url)
            model = cfg.get("chat_model", app_cfg.default_chat_model)
            return OllamaClient(base_url=url), model

        if provider == "openai":
            api_key = cfg.get("api_key", "")
            base_url = cfg.get("base_url", "")
            model = cfg.get("chat_model", "gpt-4o-mini")
            use_assistant = cfg.get("use_assistant", "false").strip().lower() == "true"
            if use_assistant:
                from norvel_writer.llm.openai_assistant import OpenAIAssistantClient
                return OpenAIAssistantClient(api_key=api_key, base_url=base_url), model
            from norvel_writer.llm.openai_client import OpenAIClient
            return OpenAIClient(api_key=api_key, base_url=base_url), model

        if provider == "anthropic":
            from norvel_writer.llm.anthropic_client import AnthropicClient
            return (
                AnthropicClient(api_key=cfg.get("api_key", "")),
                cfg.get("chat_model", "claude-3-5-haiku-20241022"),
            )

        if provider == "gemini":
            from norvel_writer.llm.gemini_client import GeminiClient
            return (
                GeminiClient(api_key=cfg.get("api_key", "")),
                cfg.get("chat_model", "gemini-1.5-flash"),
            )

        log.warning("Unknown chat provider %r — falling back to Ollama", provider)
        from norvel_writer.llm.ollama_client import OllamaClient
        from norvel_writer.config.settings import get_config
        app_cfg = get_config()
        return OllamaClient(app_cfg.ollama_base_url), app_cfg.default_chat_model

    # ── Embed ─────────────────────────────────────────────────────────────────

    def _embed_backend(self) -> Tuple[Any, str]:
        """Return (client, model) for the active embeddings provider."""
        provider = embeddings_provider()
        cfg = get_section(provider)

        if provider == "openai":
            from norvel_writer.llm.openai_client import OpenAIClient
            return (
                OpenAIClient(
                    api_key=cfg.get("api_key", ""),
                    base_url=cfg.get("base_url", ""),
                ),
                cfg.get("embed_model", "text-embedding-3-small"),
            )

        # Default: Ollama
        from norvel_writer.llm.ollama_client import OllamaClient
        from norvel_writer.config.settings import get_config
        app_cfg = get_config()
        ollama_cfg = get_section("ollama")
        url = ollama_cfg.get("base_url", app_cfg.ollama_base_url)
        model = ollama_cfg.get("embed_model", app_cfg.default_embed_model)
        return OllamaClient(base_url=url), model

    # ── Public interface (mirrors OllamaClient) ───────────────────────────────

    async def ping(self) -> bool:
        try:
            client, _ = self._chat_backend()
            return await client.ping()
        except Exception:
            return False

    async def list_models(self):
        try:
            client, _ = self._chat_backend()
            return await client.list_models()
        except Exception:
            return []

    async def chat_stream(self, model: str, messages, options=None) -> AsyncIterator[str]:
        client, active_model = self._chat_backend()
        return await client.chat_stream(active_model, messages, options)

    async def chat_complete(self, model: str, messages, options=None) -> str:
        client, active_model = self._chat_backend()
        return await client.chat_complete(active_model, messages, options)

    async def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        client, active_model = self._embed_backend()
        return await client.embed(active_model, texts)

    async def describe_image(
        self,
        image_path,
        model: str,
        prompt: str = "",
        language: str = "English",
    ) -> str:
        """Vision is only supported by Ollama; other providers return a placeholder."""
        provider = chat_provider()
        ollama_cfg = get_section("ollama")
        vision_model = ollama_cfg.get("vision_model", "").strip()

        if not vision_model:
            return f"[Image: {Path(image_path).name}] (no vision_model set in llm.ini)"

        from norvel_writer.llm.ollama_client import OllamaClient
        from norvel_writer.config.settings import get_config
        url = ollama_cfg.get("base_url", get_config().ollama_base_url)
        client = OllamaClient(base_url=url)
        return await client.describe_image(image_path, vision_model, prompt, language)


# ── Singleton ─────────────────────────────────────────────────────────────────

_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router

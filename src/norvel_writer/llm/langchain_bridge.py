"""
LangChain LLM bridge for Norvel Writer.

All LLM provider configuration is read from a .env file.
Supported providers: ollama | openai | anthropic | gemini

.env is searched in this order:
  1. <project root>/.env   (next to run.bat)
  2. <user config dir>/.env

After editing .env, call reset_singletons() (or restart the app)
for changes to take effect.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import AsyncIterator, List, Optional

log = logging.getLogger(__name__)

# ── .env discovery & loading ───────────────────────────────────────────────

def _app_root() -> Path:
    """Return the project root directory (where run.bat lives)."""
    # src/norvel_writer/llm/langchain_bridge.py
    # parent 1 → llm/   parent 2 → norvel_writer/   parent 3 → src/   parent 4 → project root
    return Path(__file__).parent.parent.parent.parent


def _config_dir() -> Path:
    from platformdirs import user_config_dir
    from norvel_writer.config.defaults import APP_NAME, APP_AUTHOR
    return Path(user_config_dir(APP_NAME, APP_AUTHOR))


def find_env_path() -> Optional[Path]:
    """Return the first .env found, or None."""
    for candidate in [_app_root() / ".env", _config_dir() / ".env"]:
        if candidate.exists():
            return candidate
    return None


def env_dest() -> Path:
    """Return the .env path to write to.  Prefers project root if it exists."""
    root_env = _app_root() / ".env"
    if root_env.exists():
        return root_env
    return _config_dir() / ".env"


def _load_env() -> None:
    env_path = find_env_path()
    if env_path:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
        log.debug("Loaded .env from %s", env_path)


_load_env()


def ensure_env_exists() -> Path:
    """Create a default .env if none exists. Returns the path."""
    dest = env_dest()
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_DEFAULT_ENV, encoding="utf-8")
    log.info("Created .env at %s", dest)
    return dest


# ── Config helpers ─────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# ── LLM singleton ──────────────────────────────────────────────────────────

_llm = None
_embeddings = None


def get_llm():
    """Return the LangChain chat model for the active provider (lazy init)."""
    global _llm
    if _llm is not None:
        return _llm

    provider = _env("LLM_PROVIDER", "ollama").lower()
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            kw: dict = {
                "model": _env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                "api_key": _env("OPENAI_API_KEY"),
                "temperature": float(_env("OPENAI_TEMPERATURE", "0.7")),
                "max_tokens": int(_env("OPENAI_MAX_TOKENS", "4096")),
                "streaming": True,
            }
            base_url = _env("OPENAI_BASE_URL")
            if base_url:
                kw["base_url"] = base_url
            _llm = ChatOpenAI(**kw)
            log.info("LLM: OpenAI %s (temp=%.2f)", kw["model"], kw["temperature"])

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            _llm = ChatAnthropic(
                model=_env("ANTHROPIC_CHAT_MODEL", "claude-3-5-haiku-20241022"),
                api_key=_env("ANTHROPIC_API_KEY"),
                temperature=float(_env("ANTHROPIC_TEMPERATURE", "0.7")),
                max_tokens=int(_env("ANTHROPIC_MAX_TOKENS", "4096")),
            )
            log.info("LLM: Anthropic %s (temp=%.2f)", _llm.model, _llm.temperature)

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm = ChatGoogleGenerativeAI(
                model=_env("GEMINI_CHAT_MODEL", "gemini-1.5-flash"),
                google_api_key=_env("GEMINI_API_KEY"),
                temperature=float(_env("GEMINI_TEMPERATURE", "0.7")),
                max_output_tokens=int(_env("GEMINI_MAX_TOKENS", "4096")),
            )
            log.info("LLM: Gemini %s (temp=%.2f)", _llm.model, _llm.temperature)

        else:  # ollama (default)
            from langchain_ollama import ChatOllama
            _ollama_kw: dict = {
                "base_url":      _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                "model":         _env("OLLAMA_CHAT_MODEL", "gemma3:4b"),
                # Sampling — Google recommends temp=1.0 / top_p=0.95 / top_k=64 for Gemma 4.
                # Defaults here are conservative (0.7) for older models; override in .env.
                "temperature":   float(_env("OLLAMA_TEMPERATURE",    "0.7")),
                # Large context window so system-prompt + chapter + beats all
                # fit in a single pass without truncation (truncation causes loops).
                "num_ctx":       int(_env("OLLAMA_NUM_CTX",          "8192")),
                # Hard cap on generated tokens.
                "num_predict":   int(_env("OLLAMA_NUM_PREDICT",      "4096")),
                # Repetition suppression: penalty applied to recent tokens.
                # repeat_last_n — how many tokens back to scan (default 64 is too short
                #   for long chapter rewrites; 512 catches multi-paragraph loops).
                # repeat_penalty > 1.0 discourages reusing those tokens.
                "repeat_last_n": int(_env("OLLAMA_REPEAT_LAST_N",   "512")),
                "repeat_penalty":float(_env("OLLAMA_REPEAT_PENALTY", "1.1")),
            }
            # top_p / top_k — only set if explicitly configured (avoid overriding
            # Ollama's own defaults when not needed, e.g. for non-Gemma models).
            _top_p = _env("OLLAMA_TOP_P")
            _top_k = _env("OLLAMA_TOP_K")
            if _top_p:
                _ollama_kw["top_p"] = float(_top_p)
            if _top_k:
                _ollama_kw["top_k"] = int(_top_k)
            _llm = ChatOllama(**_ollama_kw)
            log.info(
                "LLM: Ollama %s @ %s (ctx=%d, temp=%.2f)",
                _llm.model, _llm.base_url,
                _ollama_kw["num_ctx"], _ollama_kw["temperature"],
            )

    except Exception as exc:
        log.error("LLM init failed for provider %r: %s", provider, exc)
        _llm = None

    return _llm


# ── Embeddings singleton ───────────────────────────────────────────────────

def get_embeddings_fn():
    """Return the LangChain embeddings model for the active provider (lazy init)."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    embed_provider = _env("EMBEDDINGS_PROVIDER", "ollama").lower()
    try:
        if embed_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            _embeddings = OpenAIEmbeddings(
                api_key=_env("OPENAI_API_KEY"),
                model=_env("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            )
            log.info("Embeddings: OpenAI %s", _embeddings.model)
        else:  # ollama (default)
            from langchain_ollama import OllamaEmbeddings
            _embeddings = OllamaEmbeddings(
                base_url=_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                model=_env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            )
            log.info(
                "Embeddings: Ollama %s @ %s",
                _embeddings.model,
                _embeddings.base_url,
            )
    except Exception as exc:
        log.error("Embeddings init failed: %s", exc)
        _embeddings = None

    return _embeddings


# ── Message conversion ─────────────────────────────────────────────────────

def _to_lc_messages(messages: list[dict]):
    """Convert OpenAI-style dicts to LangChain message objects."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _extract_content(chunk) -> str:
    """Pull plain text out of a LangChain chunk/message (handles all providers)."""
    c = chunk.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # Anthropic extended format: list of content blocks
        return "".join(
            x.get("text", "") for x in c if isinstance(x, dict)
        )
    return ""


# ── Public LLM API ─────────────────────────────────────────────────────────

async def chat_stream(messages: list[dict]) -> AsyncIterator[str]:
    """
    Coroutine that returns an async generator of text tokens.

    Usage::
        stream = await chat_stream(messages)
        async for token in stream:
            ...
    """
    llm = get_llm()
    if llm is None:
        provider = _env("LLM_PROVIDER", "ollama")
        raise RuntimeError(
            f"LLM provider {provider!r} failed to initialize. "
            "Check your .env configuration."
        )
    lc_messages = _to_lc_messages(messages)

    async def _gen():
        async for chunk in llm.astream(lc_messages):
            text = _extract_content(chunk)
            if text:
                yield text

    return _gen()


async def chat_complete(messages: list[dict]) -> str:
    """Single-shot (non-streaming) chat completion."""
    llm = get_llm()
    if llm is None:
        provider = _env("LLM_PROVIDER", "ollama")
        raise RuntimeError(
            f"LLM provider {provider!r} failed to initialize. "
            "Check your .env configuration."
        )
    lc_messages = _to_lc_messages(messages)
    result = await llm.ainvoke(lc_messages)
    return _extract_content(result)


# ── Lifecycle ──────────────────────────────────────────────────────────────

def get_context_limits() -> dict:
    """
    Return token budget limits used when assembling system prompts.

    All values are in *tokens* (1 token ≈ 4 characters for English prose).
    They control how much of each context section is fed to the LLM.

    Defaults are tuned for gemma3:4b (OLLAMA_NUM_CTX=8192).
    When using a large-context model such as Gemma 4 (128 K), raise
    OLLAMA_NUM_CTX and all CONTEXT_* values accordingly — see the
    'Large-context models' block in .env for recommended values.

    Rule of thumb for sizing:
      input_budget  = OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT - ~1000 (prompt overhead)
      RAG_BUDGET    ≈ input_budget × 0.40
      STYLE_BUDGET  ≈ input_budget × 0.15
      TEXT_BUDGET   ≈ input_budget × 0.35
      (remaining ≈ 0.10 for beats, editor note, QA note, persona)
    """
    return {
        "rag_budget":   int(_env("CONTEXT_RAG_BUDGET",   "3500")),
        "style_budget": int(_env("CONTEXT_STYLE_BUDGET", "1500")),
        "text_budget":  int(_env("CONTEXT_TEXT_BUDGET",  "3000")),
    }


def reset_singletons() -> None:
    """
    Clear cached LLM and embeddings instances.
    Call after writing a new .env so the next request picks up the changes.
    """
    global _llm, _embeddings
    _llm = None
    _embeddings = None
    _load_env()


# ── Default .env template ──────────────────────────────────────────────────

_DEFAULT_ENV = """\
# ══════════════════════════════════════════════════════════════════════════════
# Norvel Writer — LLM configuration  (.env)
# ══════════════════════════════════════════════════════════════════════════════
#
# This file controls which AI backend Norvel Writer uses.
# Edit it here or via Settings → LLM Config, then restart the app.
#
# Supported providers
#   ollama    — local models via Ollama  (free, offline, private)
#   openai    — OpenAI API              (requires API key, sends data online)
#   anthropic — Anthropic Claude API    (requires API key, sends data online)
#   gemini    — Google Gemini API       (requires API key, sends data online)
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Active provider ────────────────────────────────────────────────────────
# Which backend to use for chat/writing generation.
# Valid: ollama | openai | anthropic | gemini
LLM_PROVIDER=ollama

# Which backend to use for embeddings (RAG / semantic search).
# Only "ollama" and "openai" support embeddings in this app.
# Set chat to anthropic/gemini but keep embeddings=ollama for free local RAG.
EMBEDDINGS_PROVIDER=ollama


# ── Ollama (local — https://ollama.com) ───────────────────────────────────
#
# Install Ollama: https://ollama.com/download
# Then pull models from a terminal:
#
#   ollama pull gemma3:4b          # recommended chat model  (~2.5 GB)
#   ollama pull nomic-embed-text   # required for embeddings (~274 MB)
#
# Other good chat models:
#   gemma3:1b      (lightweight, ~800 MB)
#   llama3.2:3b    (Meta, ~2 GB)
#   llama3.1:8b    (Meta, ~4.7 GB)
#   mistral:7b     (Mistral AI, ~4.1 GB)
#
# Vision models (for image description — optional):
#   llava:7b           (~4.5 GB)
#   llama3.2-vision    (~7.9 GB)
#   moondream          (~1.7 GB)
#
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=gemma3:4b
OLLAMA_EMBED_MODEL=nomic-embed-text
# OLLAMA_VISION_MODEL=llava:7b
#
# ── Sampling parameters ───────────────────────────────────────────────────
# OLLAMA_TEMPERATURE  creativity (0.0=deterministic → 1.0=most creative)
#   gemma3:4b  → 0.7 (safe default for small models)
#   gemma4:*   → 1.0 (Google's official recommendation for all Gemma 4 sizes)
# OLLAMA_TOP_P / OLLAMA_TOP_K  — leave blank to use Ollama defaults
#   gemma4:*   → TOP_P=0.95, TOP_K=64 (Google's official recommendation)
# OLLAMA_NUM_CTX   context window — must be ≤ your model's hard maximum
#   gemma3:4b  → 8192    (model maximum)
#   gemma4:e2b → 32768   (sweet spot for 8 GB VRAM; absolute max is 131072)
# OLLAMA_NUM_PREDICT   max tokens generated per call (~750 words per 1000 tokens)
# OLLAMA_REPEAT_LAST_N / REPEAT_PENALTY   repetition suppression
#   Gemma 4 is better at avoiding loops — 1.1 is enough (was 1.18 for Gemma 3)
OLLAMA_TEMPERATURE=0.7
OLLAMA_TOP_P=
OLLAMA_TOP_K=
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=4096
OLLAMA_REPEAT_LAST_N=512
OLLAMA_REPEAT_PENALTY=1.1

# ── Context budgets (all providers) ───────────────────────────────────────
#
# Controls how many tokens of each section are fed into every prompt.
# Formula:  input_budget = NUM_CTX - NUM_PREDICT - 1000 (overhead)
#           RAG_BUDGET   ≈ input_budget × 0.40
#           STYLE_BUDGET ≈ input_budget × 0.15
#           TEXT_BUDGET  ≈ input_budget × 0.35
#           (remaining ≈ 0.10 for beats, editor note, QA note, persona)
#
# ┌─────────────────┬──────────┬─────────────┬───────────────────────────┐
# │ Model / Config  │ NUM_CTX  │ NUM_PREDICT │ RAG / STYLE / TEXT        │
# ├─────────────────┼──────────┼─────────────┼───────────────────────────┤
# │ gemma3:4b       │  8 192   │    4 096    │ 3500 / 1500 / 3000        │
# │ gemma4:e2b 8GB  │ 32 768   │    8 192    │ 9000 / 3500 / 8000        │
# │ gemma4:e2b 16GB │ 65 536   │    8 192    │ 22000 / 8000 / 19000      │
# │ gemma4:e2b 24GB+│ 131 072  │    8 192    │ 48000 / 18000 / 42000     │
# └─────────────────┴──────────┴─────────────┴───────────────────────────┘
#
CONTEXT_RAG_BUDGET=3500
CONTEXT_STYLE_BUDGET=1500
CONTEXT_TEXT_BUDGET=3000


# ── OpenAI (https://platform.openai.com) ──────────────────────────────────
#
# API key: https://platform.openai.com/api-keys
# Set LLM_PROVIDER=openai to activate.
#
# Chat models: gpt-4o | gpt-4o-mini | gpt-4-turbo | o3-mini
# Embed models: text-embedding-3-small | text-embedding-3-large
#
# OPENAI_BASE_URL: leave blank for official OpenAI endpoint.
#   Set to a custom URL to use any OpenAI-compatible API
#   (LM Studio, vLLM, Together AI, Groq, Fireworks, Mistral, …)
#
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_BASE_URL=


# ── Anthropic (https://www.anthropic.com) ─────────────────────────────────
#
# API key: https://console.anthropic.com
# Set LLM_PROVIDER=anthropic to activate.
# Note: Anthropic has no embeddings API — keep EMBEDDINGS_PROVIDER=ollama.
#
# Chat models:
#   claude-3-5-haiku-20241022    (fast & cheap, recommended)
#   claude-3-5-sonnet-20241022   (balanced)
#   claude-3-opus-20240229       (most capable)
#
ANTHROPIC_API_KEY=
ANTHROPIC_CHAT_MODEL=claude-3-5-haiku-20241022


# ── Google Gemini (https://ai.google.dev) ─────────────────────────────────
#
# API key: https://aistudio.google.com/app/apikey
# Set LLM_PROVIDER=gemini to activate.
# Note: Gemini embeddings not yet integrated — keep EMBEDDINGS_PROVIDER=ollama.
#
# Chat models:
#   gemini-1.5-flash    (fast & cheap, recommended)
#   gemini-1.5-pro      (most capable)
#   gemini-2.0-flash    (next-gen)
#
GEMINI_API_KEY=
GEMINI_CHAT_MODEL=gemini-1.5-flash
"""

"""Async Ollama API client wrapper."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

log = logging.getLogger(__name__)


class OllamaConnectionError(RuntimeError):
    """Ollama service is unreachable."""


class OllamaModelNotFoundError(RuntimeError):
    """Requested model is not installed."""


@dataclass
class ModelInfo:
    name: str
    size: int  # bytes
    digest: str
    family: str = ""


class OllamaClient:
    """
    Thin async wrapper around the ollama Python SDK.

    All network errors are normalised to OllamaConnectionError or
    OllamaModelNotFoundError so callers only handle two exception types.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url
        self._client: Optional[Any] = None

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.AsyncClient(host=self._base_url)
        return self._client

    async def ping(self) -> bool:
        """Return True if the Ollama service responds.

        Uses urllib directly (same approach as working apps) rather than the
        ollama SDK, which can fail silently on Windows due to IPv6/IPv4 issues.
        Tries 127.0.0.1 explicitly to bypass localhost DNS resolution problems.
        Also respects the OLLAMA_HOST environment variable.
        """
        import os
        import urllib.request
        import urllib.parse

        candidates = []

        # 1. OLLAMA_HOST env var (e.g. "127.0.0.1:11434" or "0.0.0.0:11435")
        ollama_host = os.environ.get("OLLAMA_HOST", "").strip()
        if ollama_host:
            if "://" not in ollama_host:
                ollama_host = "http://" + ollama_host
            parsed = urllib.parse.urlparse(ollama_host)
            env_port = parsed.port or 11434
            candidates.append(("127.0.0.1", env_port))
            candidates.append(("localhost", env_port))

        # 2. Port from llm.ini / configured base_url
        cfg_port = urllib.parse.urlparse(self._base_url).port or 11434
        candidates.append(("127.0.0.1", cfg_port))
        candidates.append(("localhost", cfg_port))

        for host, port in candidates:
            try:
                url = f"http://{host}:{port}/"
                req = urllib.request.urlopen(url, timeout=3)
                req.close()
                return True
            except Exception:
                continue

        return False

    async def list_models(self) -> List[ModelInfo]:
        import json
        import os
        import urllib.request
        import urllib.parse

        candidates = []

        ollama_host = os.environ.get("OLLAMA_HOST", "").strip()
        if ollama_host:
            if "://" not in ollama_host:
                ollama_host = "http://" + ollama_host
            parsed = urllib.parse.urlparse(ollama_host)
            env_port = parsed.port or 11434
            candidates.append(("127.0.0.1", env_port))
            candidates.append(("localhost", env_port))

        cfg_port = urllib.parse.urlparse(self._base_url).port or 11434
        candidates.append(("127.0.0.1", cfg_port))
        candidates.append(("localhost", cfg_port))

        last_exc: Exception = RuntimeError("No hosts tried")

        for host, port in candidates:
            try:
                url = f"http://{host}:{port}/api/tags"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                models = []
                for m in data.get("models", []):
                    details = m.get("details", {})
                    models.append(ModelInfo(
                        name=m.get("model") or m.get("name", ""),
                        size=m.get("size", 0),
                        digest=m.get("digest", ""),
                        family=details.get("family", "") if details else "",
                    ))
                return models
            except Exception as exc:
                last_exc = exc
                continue

        raise OllamaConnectionError(f"Cannot reach Ollama at {self._base_url}") from last_exc

    async def pull_model(
        self,
        name: str,
        progress_cb: Optional[Any] = None,
    ) -> None:
        """Pull a model, calling progress_cb(pct: int) periodically."""
        try:
            import inspect
            client = self._get_client()
            result = client.pull(name, stream=True)
            # Depending on the ollama library version, pull(stream=True) may
            # return either a coroutine (needs await) or an async generator
            # directly (must NOT be awaited). Handle both.
            if inspect.isawaitable(result):
                result = await result
            async for status in result:
                if progress_cb and status.total and status.completed:
                    pct = int((status.completed / status.total) * 100)
                    progress_cb(pct)
        except Exception as exc:
            raise OllamaConnectionError(f"Failed to pull model {name!r}") from exc

    async def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion. Returns an async generator of string chunks.

        Uses the same ``return _gen()`` pattern as the other clients so that
        ``await client.chat_stream(...)`` always resolves to an async generator,
        regardless of which backend is active.
        """
        ollama_client = self._get_client()

        async def _gen():
            try:
                import inspect
                result = ollama_client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options=options or {},
                )
                # Depending on the ollama library version, chat(stream=True)
                # may return a coroutine (needs await) or an async generator
                # directly. Handle both so we work with any installed version.
                if inspect.isawaitable(result):
                    result = await result
                async for chunk in result:
                    content = chunk.message.content
                    if content:
                        yield content
            except Exception as exc:
                err_str = str(exc).lower()
                if "not found" in err_str or "pull" in err_str:
                    raise OllamaModelNotFoundError(
                        f"Model {model!r} not found. Pull it first."
                    ) from exc
                raise OllamaConnectionError(
                    f"Ollama error during chat: {exc}"
                ) from exc

        return _gen()

    async def chat_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Non-streaming chat completion. Returns full response text."""
        parts: List[str] = []
        async for chunk in await self.chat_stream(model, messages, options):
            parts.append(chunk)
        return "".join(parts)

    async def describe_image(
        self,
        image_path: "Path",
        model: str,
        prompt: str = "",
        language: str = "English",
    ) -> str:
        """
        Send an image to a vision-capable Ollama model and return a text description.

        Requires a multimodal model such as llava:7b, llama3.2-vision, or moondream.
        The image is base64-encoded and sent via the Ollama chat API images field.
        """
        import base64
        from pathlib import Path as _Path

        path = _Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        if not prompt:
            prompt = (
                f"Describe this image in detail in {language}. "
                "Focus on everything that would be useful for a novelist: "
                "locations, characters, objects, spatial relationships, atmosphere, "
                "colours, and any text visible in the image. "
                "Be thorough and specific."
            )

        try:
            client = self._get_client()
            response = await client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64],
                    }
                ],
            )
            return response.message.content or ""
        except Exception as exc:
            err_str = str(exc).lower()
            if "not found" in err_str or "pull" in err_str:
                raise OllamaModelNotFoundError(
                    f"Vision model {model!r} not found. Pull it first (e.g. ollama pull {model})."
                ) from exc
            raise OllamaConnectionError(f"Vision description failed: {exc}") from exc

    async def embed(
        self,
        model: str,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Embed a list of texts. Returns list of embedding vectors.
        Batches automatically if the list is large.
        """
        if not texts:
            return []
        try:
            import ollama
            client = self._get_client()
            results: List[List[float]] = []
            # Process one at a time — Ollama's embed endpoint takes single input
            for text in texts:
                resp = await client.embed(model=model, input=text)
                emb = resp.embeddings
                if emb:
                    results.append(emb[0])
                else:
                    results.append([])
            return results
        except Exception as exc:
            err_str = str(exc).lower()
            if "not found" in err_str:
                raise OllamaModelNotFoundError(
                    f"Embedding model {model!r} not found."
                ) from exc
            raise OllamaConnectionError(f"Embedding failed: {exc}") from exc


_client: Optional[OllamaClient] = None


def get_client():
    """
    Return the active LLM provider router.
    Reads llm.ini on first call; returns a ProviderRouter that dispatches
    to Ollama, OpenAI, Anthropic, or Gemini depending on configuration.
    Falls back to a direct OllamaClient if the provider system is unavailable.
    """
    try:
        from norvel_writer.llm.providers import get_router
        return get_router()
    except Exception:
        global _client
        if _client is None:
            from norvel_writer.config.settings import get_config
            _client = OllamaClient(get_config().ollama_base_url)
        return _client

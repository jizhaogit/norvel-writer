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

        Uses a TCP port check first to avoid localhost IPv6/IPv4 resolution
        issues on Windows where 'localhost' may resolve to ::1 but Ollama
        only listens on 127.0.0.1.
        """
        import urllib.parse
        port = urllib.parse.urlparse(self._base_url).port or 11434

        # Reliable TCP check on 127.0.0.1 directly
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port), timeout=3.0
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except Exception:
            pass

        # Fallback: SDK call
        try:
            client = self._get_client()
            await asyncio.wait_for(client.list(), timeout=5.0)
            return True
        except Exception:
            return False

    async def list_models(self) -> List[ModelInfo]:
        try:
            client = self._get_client()
            response = await client.list()
            models = []
            for m in response.models:
                models.append(ModelInfo(
                    name=m.model or m.name,
                    size=m.size or 0,
                    digest=m.digest or "",
                    family=getattr(m, "details", None) and
                           getattr(m.details, "family", "") or "",
                ))
            return models
        except Exception as exc:
            raise OllamaConnectionError(f"Cannot reach Ollama at {self._base_url}") from exc

    async def pull_model(
        self,
        name: str,
        progress_cb: Optional[Any] = None,
    ) -> None:
        """Pull a model, calling progress_cb(pct: int) periodically."""
        try:
            import ollama
            client = self._get_client()
            async for status in await client.pull(name, stream=True):
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
        """Stream chat completion, yielding string chunks."""
        try:
            import ollama
            client = self._get_client()
            async for chunk in await client.chat(
                model=model,
                messages=messages,
                stream=True,
                options=options or {},
            ):
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

    async def chat_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Non-streaming chat completion. Returns full response text."""
        parts: List[str] = []
        async for chunk in self.chat_stream(model, messages, options):
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


def get_client() -> OllamaClient:
    global _client
    if _client is None:
        from norvel_writer.config.settings import get_config
        _client = OllamaClient(get_config().ollama_base_url)
    return _client

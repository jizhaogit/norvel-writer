"""OpenAI / ChatGPT provider client using httpx for async streaming."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIClient:
    def __init__(self, api_key: str, base_url: str = "") -> None:
        self._api_key = api_key
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def ping(self) -> bool:
        """Check connectivity by listing available models."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self):
        """Return an empty list — model management is done outside the app."""
        return []

    async def chat_stream(
        self, model: str, messages: List[Dict], options: Optional[Any] = None
    ) -> AsyncIterator[str]:
        import httpx

        body = {"model": model, "messages": messages, "stream": True}

        async def _gen():
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream(
                        "POST",
                        f"{self._base_url}/chat/completions",
                        headers=self._headers(),
                        json=body,
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            if payload.strip() == "[DONE]":
                                return
                            try:
                                data = json.loads(payload)
                                content = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                pass
            except Exception as exc:
                log.error("OpenAI stream error: %s", exc)
                yield f"\n[OpenAI error: {exc}]"

        return _gen()

    async def chat_complete(
        self, model: str, messages: List[Dict], options: Optional[Any] = None
    ) -> str:
        parts: List[str] = []
        async for chunk in await self.chat_stream(model, messages, options):
            parts.append(chunk)
        return "".join(parts)

    async def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        import httpx

        results: List[List[float]] = []
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    f"{self._base_url}/embeddings",
                    headers=self._headers(),
                    json={"model": model, "input": texts},
                )
                r.raise_for_status()
                data = r.json()
                # Sort by index to preserve order
                items = sorted(data["data"], key=lambda x: x["index"])
                results = [item["embedding"] for item in items]
        except Exception as exc:
            log.error("OpenAI embed error: %s", exc)
            results = [[0.0] * 1536] * len(texts)
        return results

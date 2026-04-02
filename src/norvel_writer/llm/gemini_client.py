"""Google Gemini provider client using httpx for async streaming."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

log = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _convert_messages(self, messages: List[Dict]) -> tuple:
        """Convert OpenAI-style messages to Gemini contents format."""
        system_parts = []
        contents = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system_parts.append({"text": content})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
        return system_parts, contents

    async def ping(self) -> bool:
        try:
            import httpx
            url = f"{_BASE_URL}?key={self._api_key}"
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(url)
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self):
        return []

    async def chat_stream(
        self, model: str, messages: List[Dict], options: Optional[Any] = None
    ) -> AsyncIterator[str]:
        import httpx

        system_parts, contents = self._convert_messages(messages)
        body: Dict[str, Any] = {"contents": contents}
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        url = f"{_BASE_URL}/{model}:streamGenerateContent?key={self._api_key}&alt=sse"

        async def _gen():
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream("POST", url, json=body) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            try:
                                data = json.loads(line[6:])
                                for candidate in data.get("candidates", []):
                                    for part in candidate.get("content", {}).get("parts", []):
                                        text = part.get("text", "")
                                        if text:
                                            yield text
                            except json.JSONDecodeError:
                                pass
            except Exception as exc:
                log.error("Gemini stream error: %s", exc)
                yield f"\n[Gemini error: {exc}]"

        return _gen()

    async def chat_complete(
        self, model: str, messages: List[Dict], options: Optional[Any] = None
    ) -> str:
        parts: List[str] = []
        async for chunk in await self.chat_stream(model, messages, options):
            parts.append(chunk)
        return "".join(parts)

    async def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        """Gemini embedding not compatible with ChromaDB's expected dimensions."""
        raise NotImplementedError(
            "Gemini embeddings are not supported. "
            "Set [provider] embeddings = ollama in llm.ini."
        )

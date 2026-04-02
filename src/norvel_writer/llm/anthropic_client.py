"""Anthropic (Claude) provider client using httpx for async streaming."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

log = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_MAX_TOKENS = 4096


class AnthropicClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
        }

    def _convert_messages(self, messages: List[Dict]) -> tuple:
        """Split system message from conversation messages (Anthropic API format)."""
        system = ""
        conv = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role in ("user", "assistant"):
                conv.append({"role": role, "content": content})
        return system, conv

    async def ping(self) -> bool:
        try:
            import httpx
            # Minimal request to verify the API key works
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.post(
                    _API_URL,
                    headers=self._headers(),
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                return r.status_code in (200, 400)  # 400 = bad request but key is valid
        except Exception:
            return False

    async def list_models(self):
        return []

    async def chat_stream(
        self, model: str, messages: List[Dict], options: Optional[Any] = None
    ) -> AsyncIterator[str]:
        import httpx

        system, conv = self._convert_messages(messages)
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "messages": conv,
            "stream": True,
        }
        if system:
            body["system"] = system

        async def _gen():
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream(
                        "POST",
                        _API_URL,
                        headers=self._headers(),
                        json=body,
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "content_block_delta":
                                    text = data.get("delta", {}).get("text", "")
                                    if text:
                                        yield text
                            except json.JSONDecodeError:
                                pass
            except Exception as exc:
                log.error("Anthropic stream error: %s", exc)
                yield f"\n[Anthropic error: {exc}]"

        return _gen()

    async def chat_complete(
        self, model: str, messages: List[Dict], options: Optional[Any] = None
    ) -> str:
        parts: List[str] = []
        async for chunk in await self.chat_stream(model, messages, options):
            parts.append(chunk)
        return "".join(parts)

    async def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        """Anthropic does not provide an embeddings API."""
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Set [provider] embeddings = ollama in llm.ini."
        )

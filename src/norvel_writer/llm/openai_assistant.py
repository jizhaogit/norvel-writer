"""
OpenAI Assistants API integration for Norvel Writer.

Design principles:
- All documents stay in LOCAL ChromaDB — nothing is uploaded to OpenAI.
- The assistant provides a persistent writing identity with tuned instructions.
- RAG context retrieved locally is injected into each thread message.
- A new thread is created per generation call (stateless from OpenAI's side).
- The assistant object itself is created once and reused (cached locally).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

log = logging.getLogger(__name__)

_ASSISTANT_NAME = "Norvel Writer"
_ASSISTANT_INSTRUCTIONS = """\
You are Norvel Writer, an expert creative writing assistant embedded in a local desktop app.

Your role:
- Continue stories naturally from where the author left off
- Rewrite passages to improve style, clarity, or pacing
- Answer questions about the story world, characters, and plot
- Help resolve continuity issues and suggest story developments
- Write in whatever language the author requests

When project context (codex entries, character sheets, chapter beats, style notes) is provided
in the [Context] section of a message, use it to stay consistent with the established world.
Never contradict established facts. Match the author's voice and tone.
"""


def _cache_path() -> Path:
    from platformdirs import user_config_dir
    from norvel_writer.config.defaults import APP_NAME, APP_AUTHOR
    return Path(user_config_dir(APP_NAME, APP_AUTHOR)) / "openai_assistant.json"


def _load_cached_id() -> Optional[str]:
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
        return data.get("assistant_id")
    except Exception:
        return None


def _save_cached_id(assistant_id: str) -> None:
    try:
        _cache_path().parent.mkdir(parents=True, exist_ok=True)
        _cache_path().write_text(
            json.dumps({"assistant_id": assistant_id}), encoding="utf-8"
        )
    except Exception:
        pass


class OpenAIAssistantClient:
    """
    Wraps the OpenAI Assistants API.

    Documents are stored locally.  Each call:
    1. Retrieves RAG context from local ChromaDB (done by DraftEngine before calling here).
    2. Injects that context into the user message sent to the thread.
    3. Streams the assistant's response back token by token.
    4. Deletes the ephemeral thread after the response completes.
    """

    def __init__(self, api_key: str, base_url: str = "") -> None:
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._assistant_id: Optional[str] = _load_cached_id()

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2",
        }

    # ── Assistant lifecycle ───────────────────────────────────────────────────

    async def _get_or_create_assistant(self, model: str) -> str:
        """Return the cached assistant ID, creating it if necessary."""
        import httpx

        # Verify cached assistant still exists
        if self._assistant_id:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        f"{self._base_url}/assistants/{self._assistant_id}",
                        headers=self._auth_headers(),
                    )
                    if r.status_code == 200:
                        log.debug("Reusing assistant %s", self._assistant_id)
                        return self._assistant_id
            except Exception:
                pass

        # Create a new assistant
        log.info("Creating Norvel Writer assistant (one-time setup)…")
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self._base_url}/assistants",
                headers=self._auth_headers(),
                json={
                    "name": _ASSISTANT_NAME,
                    "instructions": _ASSISTANT_INSTRUCTIONS,
                    "model": model,
                },
            )
            r.raise_for_status()
            assistant_id = r.json()["id"]

        self._assistant_id = assistant_id
        _save_cached_id(assistant_id)
        log.info("Created assistant %s", assistant_id)
        return assistant_id

    # ── Message conversion ────────────────────────────────────────────────────

    @staticmethod
    def _build_thread_message(messages: List[Dict]) -> str:
        """
        Flatten OpenAI-style messages into a single thread message.

        The system prompt (which contains RAG context injected by DraftEngine)
        becomes a [Context] block.  The last user message is the actual request.
        """
        context_parts: List[str] = []
        request_parts: List[str] = []

        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                context_parts.append(content)
            elif role == "user":
                request_parts.append(content)
            # assistant turns (history) — prepend as quoted context
            elif role == "assistant":
                context_parts.append(f"[Previous assistant reply]\n{content}")

        parts = []
        if context_parts:
            parts.append("[Context]\n" + "\n\n---\n\n".join(context_parts))
        if request_parts:
            parts.append("[Request]\n" + "\n\n".join(request_parts))

        return "\n\n".join(parts)

    # ── Public interface (matches OllamaClient) ───────────────────────────────

    async def ping(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list:
        return []

    async def chat_stream(
        self, model: str, messages: List[Dict], options: Optional[Any] = None
    ) -> AsyncIterator[str]:
        import httpx

        assistant_id = await self._get_or_create_assistant(model)
        user_message = self._build_thread_message(messages)

        async def _gen():
            thread_id: Optional[str] = None
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    # 1. Create ephemeral thread
                    r = await client.post(
                        f"{self._base_url}/threads",
                        headers=self._auth_headers(),
                        json={},
                    )
                    r.raise_for_status()
                    thread_id = r.json()["id"]

                    # 2. Add user message (contains locally-retrieved RAG context)
                    r = await client.post(
                        f"{self._base_url}/threads/{thread_id}/messages",
                        headers=self._auth_headers(),
                        json={"role": "user", "content": user_message},
                    )
                    r.raise_for_status()

                    # 3. Stream the run
                    async with client.stream(
                        "POST",
                        f"{self._base_url}/threads/{thread_id}/runs",
                        headers=self._auth_headers(),
                        json={"assistant_id": assistant_id, "stream": True},
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                data = json.loads(payload)
                                if data.get("object") == "thread.message.delta":
                                    for part in data.get("delta", {}).get("content", []):
                                        text = (
                                            part.get("text", {}).get("value", "")
                                            if isinstance(part.get("text"), dict)
                                            else ""
                                        )
                                        if text:
                                            yield text
                            except json.JSONDecodeError:
                                pass

            except Exception as exc:
                log.error("OpenAI Assistant stream error: %s", exc)
                yield f"\n[OpenAI Assistant error: {exc}]"
            finally:
                # 4. Delete the ephemeral thread to keep the account tidy
                if thread_id:
                    try:
                        import httpx as _hx
                        async with _hx.AsyncClient(timeout=5) as cl:
                            await cl.delete(
                                f"{self._base_url}/threads/{thread_id}",
                                headers=self._auth_headers(),
                            )
                    except Exception:
                        pass

        return _gen()

    async def chat_complete(
        self, model: str, messages: List[Dict], options: Optional[Any] = None
    ) -> str:
        parts: List[str] = []
        async for chunk in await self.chat_stream(model, messages, options):
            parts.append(chunk)
        return "".join(parts)

    async def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        """Delegate to the standard OpenAI embeddings endpoint."""
        from norvel_writer.llm.openai_client import OpenAIClient
        client = OpenAIClient(api_key=self._api_key, base_url=self._base_url)
        return await client.embed(model, texts)

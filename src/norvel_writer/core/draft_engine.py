"""DraftEngine: continue, rewrite, and summarise workflows."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

log = logging.getLogger(__name__)


class DraftEngine:
    """
    Handles continuation, rewrite, and summarise workflows.
    All primary methods are async generators that yield string tokens.
    """

    def __init__(
        self,
        project_manager=None,
        model: Optional[str] = None,
    ) -> None:
        from norvel_writer.core.project import ProjectManager
        from norvel_writer.config.settings import get_config
        self._pm = project_manager or ProjectManager()
        self._model = model or get_config().default_chat_model

    async def continue_draft(
        self,
        project_id: str,
        chapter_id: str,
        current_text: str,
        user_instruction: str = "Continue the story from where it left off.",
        style_mode: str = "inspired_by",
        constraints: Optional[List[str]] = None,
        language: str = "en",
        active_doc_types: Optional[List[str]] = None,
        beats: str = "",
    ) -> AsyncIterator[str]:
        """Stream continuation tokens for the current draft."""
        from norvel_writer.llm.ollama_client import get_client
        from norvel_writer.llm.prompt_builder import build_continuation_messages
        from norvel_writer.utils.text_utils import truncate_to_tokens

        # Retrieve context
        last_para = _last_paragraphs(current_text, n_tokens=512)
        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=last_para,
            n_results=8,
            doc_types=active_doc_types or ["codex", "beats", "draft", "research"],
            chapter_id=chapter_id,
        )
        style_results = await self._pm.retrieve_style_examples(
            project_id=project_id,
            query=last_para,
            n_results=4,
        )

        rag_chunks = [r["text"] for r in rag_results]
        style_chunks = [r["text"] for r in style_results]

        style_profile_data = self._pm.get_active_style_profile(project_id)
        style_profile = None
        if style_profile_data:
            try:
                style_profile = json.loads(style_profile_data["profile_json"])
            except Exception:
                pass

        proj = self._pm.get_project(project_id)
        persona = (proj.get("persona") or "").strip() if proj else ""

        context_text = truncate_to_tokens(current_text, max_tokens=2048)
        messages = build_continuation_messages(
            current_text=context_text,
            rag_chunks=rag_chunks,
            style_chunks=style_chunks,
            style_profile=style_profile,
            user_instruction=user_instruction,
            language=language,
            style_mode=style_mode,
            constraints=constraints,
            persona=persona,
            beats=beats,
        )

        client = get_client()
        return client.chat_stream(self._model, messages)

    async def rewrite_passage(
        self,
        project_id: str,
        passage: str,
        user_instruction: str = "Rewrite this passage in the same style.",
        style_mode: str = "preserve_tone_rhythm",
        language: str = "en",
    ) -> AsyncIterator[str]:
        """Stream rewritten passage tokens."""
        from norvel_writer.llm.ollama_client import get_client
        from norvel_writer.llm.prompt_builder import build_rewrite_messages

        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=passage,
            n_results=4,
        )
        style_results = await self._pm.retrieve_style_examples(
            project_id=project_id,
            query=passage,
            n_results=4,
        )

        style_profile_data = self._pm.get_active_style_profile(project_id)
        style_profile = None
        if style_profile_data:
            try:
                style_profile = json.loads(style_profile_data["profile_json"])
            except Exception:
                pass

        proj = self._pm.get_project(project_id)
        persona = (proj.get("persona") or "").strip() if proj else ""

        messages = build_rewrite_messages(
            passage=passage,
            rag_chunks=[r["text"] for r in rag_results],
            style_chunks=[r["text"] for r in style_results],
            style_profile=style_profile,
            user_instruction=user_instruction,
            language=language,
            style_mode=style_mode,
            persona=persona,
        )

        client = get_client()
        return client.chat_stream(self._model, messages)

    async def summarise_chapter(
        self,
        chapter_text: str,
        language: str = "en",
    ) -> str:
        """Return a 1-3 sentence summary of a chapter."""
        from norvel_writer.llm.ollama_client import get_client
        from norvel_writer.utils.text_utils import truncate_to_tokens

        text = truncate_to_tokens(chapter_text, max_tokens=3000)
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a writing assistant. Summarise the following chapter "
                    f"in 1-3 sentences in {language}. Be concise and factual."
                ),
            },
            {"role": "user", "content": text},
        ]
        client = get_client()
        return await client.chat_complete(self._model, messages)

    async def check_continuity(
        self,
        project_id: str,
        passage: str,
        language: str = "en",
    ) -> str:
        """Check passage for contradictions with project codex/beats."""
        from norvel_writer.llm.ollama_client import get_client

        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=passage,
            n_results=6,
            doc_types=["codex", "beats"],
        )
        context = "\n\n---\n\n".join(r["text"] for r in rag_results)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a continuity checker. Given the project reference material, "
                    "identify any contradictions or inconsistencies in the passage. "
                    f"Respond in {language}. If no issues found, say so briefly."
                ),
            },
            {
                "role": "user",
                "content": f"REFERENCE MATERIAL:\n{context}\n\nPASSAGE TO CHECK:\n{passage}",
            },
        ]
        client = get_client()
        return await client.chat_complete(self._model, messages)

    async def chat_with_context(
        self,
        project_id: str,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        language: str = "en",
    ) -> AsyncIterator[str]:
        """
        Free-form Q&A: answer any question about the project using RAG context.
        Supports multi-turn history. Returns an async generator of string tokens.

        Examples of questions the user can ask in any language:
          - "Who is the antagonist in my story?"
          - "What is the geography of the northern kingdom?"
          - "Suggest three ways to end chapter 5."
          - "给我一些关于主角动机的建议" (Chinese)
        """
        from norvel_writer.llm.ollama_client import get_client

        # Retrieve relevant context for the question
        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=question,
            n_results=6,
        )
        context = "\n\n---\n\n".join(r["text"] for r in rag_results)

        system_prompt = (
            "You are a knowledgeable writing assistant with full access to the author's "
            "project materials. Answer questions, offer suggestions, brainstorm ideas, "
            "and help the author think through their story. "
            f"Always respond in the same language as the user's question (detected: {language}). "
            "Be concise but thorough. If you use project material in your answer, say so briefly."
        )
        if context:
            system_prompt += f"\n\n## Project Reference Material\n{context}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        client = get_client()
        return client.chat_stream(self._model, messages)


def _last_paragraphs(text: str, n_tokens: int = 512) -> str:
    """Return the last ~n_tokens worth of text for use as RAG query."""
    max_chars = n_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]

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

    LLM provider and model are configured via .env (LLM_PROVIDER + model keys).
    The optional ``model`` parameter is kept for API compatibility but is ignored;
    the active model is always determined by the LangChain bridge.
    """

    def __init__(
        self,
        project_manager=None,
        model: Optional[str] = None,
    ) -> None:
        from norvel_writer.core.project import ProjectManager
        self._pm = project_manager or ProjectManager()

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
        text_after_cursor: str = "",
    ) -> AsyncIterator[str]:
        """Stream continuation tokens for the current draft."""
        from norvel_writer.llm.langchain_bridge import chat_stream
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
            text_after_cursor=text_after_cursor,
        )

        return await chat_stream(messages)

    async def rewrite_passage(
        self,
        project_id: str,
        passage: str,
        user_instruction: str = "Rewrite this passage in the same style.",
        style_mode: str = "preserve_tone_rhythm",
        language: str = "en",
        beats: str = "",
    ) -> AsyncIterator[str]:
        """Stream rewritten passage tokens."""
        from norvel_writer.llm.langchain_bridge import chat_stream
        from norvel_writer.llm.prompt_builder import build_rewrite_messages

        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=passage,
            n_results=6,
            doc_types=["codex", "beats", "draft", "research"],
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
            beats=beats,
        )

        return await chat_stream(messages)

    async def summarise_chapter(
        self,
        chapter_text: str,
        language: str = "en",
    ) -> str:
        """Return a 1-3 sentence summary of a chapter."""
        from norvel_writer.llm.langchain_bridge import chat_complete
        from norvel_writer.llm.prompt_builder import _lang_display
        from norvel_writer.utils.text_utils import truncate_to_tokens

        text = truncate_to_tokens(chapter_text, max_tokens=3000)
        lang = _lang_display(language)
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a writing assistant. Summarise the following chapter "
                    f"in 1-3 sentences. Write your summary in {lang}. Be concise and factual."
                ),
            },
            {"role": "user", "content": text},
        ]
        return await chat_complete(messages)

    async def check_continuity(
        self,
        project_id: str,
        passage: str,
        language: str = "en",
    ) -> str:
        """Check passage for contradictions with project codex/beats."""
        from norvel_writer.llm.langchain_bridge import chat_complete
        from norvel_writer.llm.prompt_builder import _lang_display

        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=passage,
            n_results=6,
            doc_types=["codex", "beats"],
        )
        context = "\n\n---\n\n".join(r["text"] for r in rag_results)
        lang = _lang_display(language)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a continuity checker. Given the project reference material, "
                    "identify any contradictions or inconsistencies in the passage. "
                    f"Write your response in {lang}. If no issues found, say so briefly."
                ),
            },
            {
                "role": "user",
                "content": f"REFERENCE MATERIAL:\n{context}\n\nPASSAGE TO CHECK:\n{passage}",
            },
        ]
        return await chat_complete(messages)

    async def chat_with_context(
        self,
        project_id: str,
        question: str,
        chapter_id: str = "",
        role: str = "editor",
        history: Optional[List[Dict[str, str]]] = None,
        language: str = "en",
    ) -> AsyncIterator[str]:
        """
        Role-based chat with full project context.
        role: "editor" | "writer" | "qa"
        Responds in the same language the user writes in.
        """
        from norvel_writer.llm.langchain_bridge import chat_stream
        from norvel_writer.utils.text_utils import detect_language, strip_html, truncate_to_tokens
        from norvel_writer.llm.prompt_builder import _lang_display

        # ── Auto-detect response language ──────────────────────────────────
        lang_display = _lang_display(detect_language(question))

        # ── Load current chapter text ──────────────────────────────────────
        chapter_text = ""
        chapter_title = ""
        if chapter_id:
            try:
                from norvel_writer.storage.repositories.project_repo import ProjectRepo
                from norvel_writer.storage.db import get_db
                ch_row = ProjectRepo(get_db()).get_chapter(chapter_id)
                if ch_row:
                    chapter_title = ch_row.get("title") or "Untitled Chapter"
                    raw = ch_row.get("content") or ""
                    chapter_text = truncate_to_tokens(strip_html(raw), max_tokens=3000)
            except Exception:
                pass

        # ── RAG retrieval (codex / beats / research) ───────────────────────
        rag_query = f"{chapter_title}: {question}" if chapter_title else question
        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=rag_query,
            n_results=8,
            doc_types=["codex", "beats", "research"],
        )
        rag_context = "\n\n---\n\n".join(r["text"] for r in rag_results)

        # ── Extra context for Writer role ──────────────────────────────────
        persona = ""
        style_chunks: List[str] = []
        if role == "writer":
            proj = self._pm.get_project(project_id)
            persona = (proj.get("persona") or "").strip() if proj else ""

            style_results = await self._pm.retrieve_style_examples(
                project_id=project_id,
                query=question,
                n_results=4,
            )
            style_chunks = [r["text"] for r in style_results]

        # ── Build role-specific system prompt ──────────────────────────────
        lang_line = (
            f"ALWAYS respond in the same language the user writes in. "
            f"The user appears to be writing in {lang_display} — respond in {lang_display}."
        )

        if role == "editor":
            system_prompt = (
                "You are a senior professional book editor with 20+ years of experience "
                "at major publishing houses. Your role is to give the author honest, "
                "constructive, publisher-level editorial feedback.\n\n"
                "Your focus areas:\n"
                "• Narrative structure & pacing — does the chapter flow well? are scene transitions clear?\n"
                "• Character voice & consistency — does each character sound distinct and believable?\n"
                "• Dialogue — is it natural? does it serve the scene? does it reveal character?\n"
                "• Show vs tell — flag where emotions/actions are told rather than shown\n"
                "• Prose clarity & style — unclear sentences, overwriting, repetition\n"
                "• Tension & reader engagement — does the chapter hold attention?\n"
                "• Marketability — does it meet genre and audience expectations?\n\n"
                "When giving feedback:\n"
                "- Quote directly from the chapter text to anchor each point\n"
                "- Explain WHY something works or doesn't work\n"
                "- Give specific, actionable revision suggestions\n"
                "- Balance positive observations with areas for improvement\n"
                "- Be honest but respectful — the goal is to strengthen the work\n\n"
                + lang_line
            )

        elif role == "writer":
            system_prompt = (
                "You are a skilled professional co-author and writing collaborator. "
                "Your role is to help the author write new content — scenes, dialogue, "
                "descriptions, chapter continuations — while faithfully following their "
                "established style, voice, characters, and story rules.\n\n"
                "You MUST honour:\n"
                "1. The author's persona & voice instructions (PRIMARY — override everything else)\n"
                "2. The chapter beats (advance the plot in order — do not skip or add beats)\n"
                "3. The project codex (character traits, world rules, lore, naming conventions)\n"
                "4. The style samples (match tone, rhythm, sentence structure)\n\n"
                "When writing:\n"
                "- Do NOT add meta-commentary or explain your choices\n"
                "- Do NOT use character names differently from the codex\n"
                "- Maintain the established POV and tense\n"
                "- Write directly usable prose — not outlines or summaries\n\n"
                + lang_line
            )

        else:  # qa
            system_prompt = (
                "You are a meticulous QA (Quality Assurance) reviewer for creative fiction. "
                "Your role is to systematically audit the author's chapter for errors, "
                "inconsistencies, and problems — not to rewrite, just to identify issues.\n\n"
                "Check all of the following:\n"
                "• Codex compliance — do character names, traits, abilities, and relationships "
                "match the codex exactly? Are world rules respected?\n"
                "• Beat adherence — does the chapter follow the planned beats in order? "
                "Are any beats missing, skipped, or contradicted?\n"
                "• Logic & causality — do events follow logically? Are character decisions "
                "consistent with their established motivations?\n"
                "• Continuity — does anything contradict what was established earlier "
                "(timeline, locations, objects, relationships)?\n"
                "• Descriptive consistency — do settings, appearances, and physical objects "
                "stay consistent throughout the chapter?\n"
                "• Chaos or confusion — are there scenes that are hard to follow, "
                "unclear action blocking, or confusing POV shifts?\n\n"
                "Format your response as a structured report:\n"
                "- ✅ PASS items (things that are correct)\n"
                "- ⚠️ ISSUES (specific problems with location in text and explanation)\n"
                "- 📋 SUMMARY (overall verdict)\n\n"
                + lang_line
            )

        # ── Attach chapter content ─────────────────────────────────────────
        if chapter_text:
            system_prompt += (
                f"\n\n## Current Chapter: {chapter_title}\n"
                f"{chapter_text}"
            )

        # ── Attach persona (Writer only) ───────────────────────────────────
        if persona:
            system_prompt += (
                f"\n\n## Author's Personal Style Instructions (PRIMARY DIRECTIVE)\n"
                f"{persona}"
            )

        # ── Attach style samples (Writer only) ────────────────────────────
        if style_chunks:
            system_prompt += "\n\n## Style Reference Samples\n"
            for chunk in style_chunks:
                system_prompt += f"---\n{chunk}\n"

        # ── Attach codex / beats / research ───────────────────────────────
        if rag_context:
            system_prompt += (
                f"\n\n## Project Reference Material (Codex / Beats / Research)\n"
                f"{rag_context}"
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        return await chat_stream(messages)


def _last_paragraphs(text: str, n_tokens: int = 512) -> str:
    """Return the last ~n_tokens worth of text for use as RAG query."""
    max_chars = n_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]

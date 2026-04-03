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
        editor_note: str = "",
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

        # ── Resolve which chapter the user is asking about ─────────────────
        # Priority: explicitly open chapter_id → chapter mentioned by name/number in question
        chapter_text = ""
        chapter_title = ""
        resolved_chapter_id = chapter_id

        # Use self._pm directly — it already holds a working DB connection
        if not resolved_chapter_id:
            try:
                all_chapters = self._pm.list_chapters(project_id)
                resolved_chapter_id = _detect_chapter_id(question, all_chapters)
            except Exception as exc:
                log.warning("chat: chapter list failed: %s", exc)

        if resolved_chapter_id:
            try:
                ch_row = self._pm.get_chapter(resolved_chapter_id)
                if ch_row:
                    chapter_title = ch_row.get("title") or "Untitled Chapter"
                # Chapter prose lives in the drafts table, not the chapters row
                draft = self._pm.get_accepted_draft(resolved_chapter_id)
                if draft:
                    raw = draft.get("content") or ""
                    chapter_text = truncate_to_tokens(strip_html(raw), max_tokens=3000)
                    log.debug("chat: loaded chapter %r (%d chars)", chapter_title, len(chapter_text))
                else:
                    log.debug("chat: no accepted draft for chapter %r", resolved_chapter_id)
            except Exception as exc:
                log.warning("chat: chapter load failed for %r: %s", resolved_chapter_id, exc)

        # ── Detect topic focus from question keywords ───────────────────────
        # Lets the user say "check the codex" or "第一章的节拍" and get the right content
        q_lower = question.lower()
        topic_wants_codex = any(kw in q_lower for kw in [
            "codex", "character", "world", "lore", "rule", "setting",
            "世界观", "角色", "设定", "人物", "コーデックス", "キャラ", "世界設定",
        ])
        topic_wants_beats = any(kw in q_lower for kw in [
            "beat", "plot", "outline", "structure", "story arc",
            "节拍", "情节", "大纲", "结构", "ビート", "プロット", "構成",
        ])

        # ── Role-specific RAG doc_types ────────────────────────────────────
        # Editor: beats only by default (knows chapter intent without codex noise).
        #         Expand to codex if user explicitly asks about codex/characters.
        # QA:     codex + beats (compliance checking)
        # Writer: everything
        if role == "editor":
            rag_doc_types = ["beats"]
            if topic_wants_codex:
                rag_doc_types.append("codex")
        elif role == "qa":
            rag_doc_types = ["codex", "beats"]
        else:
            rag_doc_types = ["codex", "beats", "research"]

        # If the user is specifically asking about beats/plot, always include beats
        if topic_wants_beats and "beats" not in rag_doc_types:
            rag_doc_types.append("beats")

        rag_query = f"{chapter_title}: {question}" if chapter_title else question
        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=rag_query,
            n_results=6,
            doc_types=rag_doc_types,
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

        # ── Language instruction ───────────────────────────────────────────
        lang_line = (
            f"ALWAYS respond in the same language the user writes in. "
            f"The user appears to be writing in {lang_display} — respond in {lang_display}."
        )

        # ── Build role-specific system prompt ──────────────────────────────
        if role == "editor":
            system_prompt = (
                "You are a senior professional book editor with 20+ years of experience "
                "at major publishing houses. Your ONLY job right now is to give the author "
                "honest, constructive, publisher-level editorial feedback on the chapter text "
                "provided below.\n\n"
                "DO NOT discuss the codex, world-building documents, or project metadata. "
                "Focus exclusively on the prose, structure, and craft of the chapter itself.\n\n"
                "Your focus areas:\n"
                "• Narrative structure & pacing — does the chapter flow? are transitions clear?\n"
                "• Character voice & consistency — does each character sound distinct?\n"
                "• Dialogue — is it natural? does it serve the scene? reveal character?\n"
                "• Show vs tell — flag where emotions/actions are told rather than shown\n"
                "• Prose clarity & style — unclear sentences, overwriting, repetition\n"
                "• Tension & reader engagement — does the chapter hold attention?\n"
                "• Marketability — does it meet genre and audience expectations?\n\n"
                "When giving feedback:\n"
                "- Quote directly from the chapter text to anchor each point\n"
                "- Explain WHY something works or doesn't\n"
                "- Give specific, actionable revision suggestions\n"
                "- Balance positive observations with areas for improvement\n\n"
                + lang_line
            )
            # Chapter content is the PRIMARY subject — place it prominently
            if not chapter_text:
                system_prompt += (
                    "\n\n⚠️ No chapter content is loaded. Tell the user: "
                    "'Please open and select a chapter in the editor panel first, "
                    "then I can give you editorial feedback on its content.'"
                )
            else:
                system_prompt += (
                    f"\n\n════════════════════════════════════════\n"
                    f"CHAPTER TO REVIEW: {chapter_title}\n"
                    f"════════════════════════════════════════\n"
                    f"{chapter_text}\n"
                    f"════════════════════════════════════════\n"
                    f"(End of chapter. Give feedback on the text above only.)"
                )
            if rag_context:
                system_prompt += (
                    f"\n\n## Planned Chapter Beats (for context only)\n"
                    f"{rag_context}"
                )

        elif role == "writer":
            system_prompt = (
                "You are a skilled professional co-author and writing collaborator. "
                "Your role is to help the author write new content — scenes, dialogue, "
                "descriptions, chapter continuations — while faithfully following their "
                "established style, voice, characters, and story rules.\n\n"
                "You MUST honour (in priority order):\n"
                "1. The author's persona & voice instructions (PRIMARY — overrides everything)\n"
                "2. Pinned editor suggestions — apply every point when rewriting\n"
                "3. The chapter beats (advance the plot in order — do not skip or add beats)\n"
                "4. The project codex (character traits, world rules, lore, naming)\n"
                "5. The style samples (match tone, rhythm, sentence structure)\n\n"
                "When writing:\n"
                "- Do NOT add meta-commentary or explain your choices\n"
                "- Maintain the established POV and tense\n"
                "- Write directly usable prose — not outlines or summaries\n\n"
                + lang_line
            )
            if persona:
                system_prompt += (
                    f"\n\n## PRIMARY DIRECTIVE — Author's Voice\n{persona}"
                )
            if editor_note:
                system_prompt += (
                    f"\n\n╔══════════════════════════════════════════╗\n"
                    f"║  📌 PINNED EDITOR SUGGESTIONS — APPLY ALL ║\n"
                    f"╚══════════════════════════════════════════╝\n"
                    f"{editor_note}\n"
                    f"(Every point above must be addressed in your rewrite. "
                    f"Do not ignore any suggestion.)"
                )
            if rag_context:
                system_prompt += (
                    f"\n\n## Project Reference Material (Codex / Beats / Research)\n"
                    f"{rag_context}"
                )
            if style_chunks:
                system_prompt += "\n\n## Style Reference Samples\n"
                for chunk in style_chunks:
                    system_prompt += f"---\n{chunk}\n"
            if chapter_text:
                system_prompt += (
                    f"\n\n## Current Chapter Content (for context)\n"
                    f"{chapter_text}"
                )

        else:  # qa
            system_prompt = (
                "You are a meticulous QA (Quality Assurance) reviewer for creative fiction. "
                "Your role is to systematically audit the chapter text below for errors "
                "and inconsistencies — not to rewrite, only to identify issues.\n\n"
                "Check all of the following:\n"
                "• Codex compliance — character names, traits, abilities, relationships, world rules\n"
                "• Beat adherence — does the chapter follow the planned beats in order?\n"
                "• Logic & causality — do events follow logically? are decisions motivated?\n"
                "• Continuity — timeline, locations, objects, relationships vs earlier chapters\n"
                "• Descriptive consistency — settings, appearances, physical objects\n"
                "• Chaos / confusion — unclear blocking, confusing POV shifts, hard-to-follow scenes\n\n"
                "Format your response as a structured report:\n"
                "- ✅ PASS items\n"
                "- ⚠️ ISSUES (quote the exact location + explain the problem)\n"
                "- 📋 SUMMARY\n\n"
                + lang_line
            )
            if not chapter_text:
                system_prompt += (
                    "\n\n⚠️ No chapter content is loaded. Tell the user: "
                    "'Please open and select a chapter in the editor panel first, "
                    "then I can run a QA check on its content.'"
                )
            else:
                system_prompt += (
                    f"\n\n════════════════════════════════════════\n"
                    f"CHAPTER TO AUDIT: {chapter_title}\n"
                    f"════════════════════════════════════════\n"
                    f"{chapter_text}\n"
                    f"════════════════════════════════════════"
                )
            if rag_context:
                system_prompt += (
                    f"\n\n## Reference Material to Check Against (Codex / Beats)\n"
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


def _detect_chapter_id(question: str, chapters: list) -> str:
    """
    Try to identify which chapter the user is asking about by scanning
    their question for chapter numbers or titles.  Works cross-language.

    Strategies (in order):
    1. Numeric chapter reference: "chapter 1", "第1章", "チャプター2", "章节1" etc.
    2. Ordinal words mapped to numbers (first/second/第一/第二/一/二…)
    3. Chapter title substring match (case-insensitive)

    Returns the chapter_id string if found, else "".
    """
    import re

    if not chapters:
        return ""

    q = question.lower()

    # ── Map ordinal words → integer ────────────────────────────────────────
    ORDINALS: dict = {
        # English
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        # Chinese / Japanese shared characters
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        # Korean
        "첫": 1, "둘": 2, "셋": 3,
    }

    # ── Pattern 1: explicit numeric references ─────────────────────────────
    # Matches: "chapter 3", "ch3", "第3章", "チャプター3", "章节3", "cap 3",
    #          "kap 3", "kapitel 3", "chapitre 3", "capitulo 3" etc.
    num_patterns = [
        r"(?:chapter|chap|ch|第|章节|チャプター|챕터|capitulo|chapitre|kapitel|kap|cap)[.\s\-_]*(\d+)",
        r"(\d+)(?:st|nd|rd|th)?\s*(?:chapter|chap|章|章节)",
    ]
    for pat in num_patterns:
        m = re.search(pat, q, re.IGNORECASE | re.UNICODE)
        if m:
            n = int(m.group(1))
            # Match by position (1-based) or by numeric suffix in title
            if 1 <= n <= len(chapters):
                return chapters[n - 1]["id"]
            # Also try matching title containing the number
            for ch in chapters:
                title = (ch.get("title") or "").lower()
                if str(n) in title:
                    return ch["id"]

    # ── Pattern 2: ordinal words ───────────────────────────────────────────
    for word, n in ORDINALS.items():
        if word in q:
            # Check it's near a chapter-related word
            ctx_pattern = rf"{re.escape(word)}.{{0,20}}(?:chapter|chap|章|章节|チャプター|챕터)"
            ctx_pattern2 = rf"(?:chapter|chap|章|章节|チャプター|챕터).{{0,20}}{re.escape(word)}"
            if re.search(ctx_pattern, q, re.IGNORECASE | re.UNICODE) or \
               re.search(ctx_pattern2, q, re.IGNORECASE | re.UNICODE):
                if 1 <= n <= len(chapters):
                    return chapters[n - 1]["id"]

    # ── Pattern 3: chapter title substring match ───────────────────────────
    for ch in chapters:
        title = (ch.get("title") or "").strip().lower()
        if title and title in q:
            return ch["id"]

    return ""

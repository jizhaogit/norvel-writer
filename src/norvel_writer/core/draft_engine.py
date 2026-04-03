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
        editor_note: str = "",
        qa_note: str = "",
    ) -> AsyncIterator[str]:
        """Stream continuation tokens — uses the Writer role skill, same priority as chat Writer."""
        from norvel_writer.llm.langchain_bridge import chat_stream
        from norvel_writer.llm.prompt_builder import _lang_display
        from norvel_writer.utils.text_utils import truncate_to_tokens

        lang_display = _lang_display(language)
        last_para = _last_paragraphs(current_text, n_tokens=512)

        # RAG — same doc types as Writer chat
        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=last_para,
            n_results=8,
            doc_types=active_doc_types or ["codex", "beats", "research", "notes"],
            chapter_id=chapter_id,
        )
        style_results = await self._pm.retrieve_style_examples(
            project_id=project_id,
            query=last_para,
            n_results=4,
        )
        rag_context = "\n\n---\n\n".join(r["text"] for r in rag_results)
        style_chunks = [r["text"] for r in style_results]

        proj = self._pm.get_project(project_id)
        persona = (proj.get("persona") or "").strip() if proj else ""

        # Image descriptions — same as Writer chat
        image_context = _fetch_image_context(self._pm._db, project_id, chapter_id)

        context_text = truncate_to_tokens(current_text, max_tokens=2048)

        system_prompt = _build_writer_system_prompt(
            lang_display=lang_display,
            persona=persona,
            editor_note=editor_note,
            rag_context=rag_context,
            image_context=image_context,
            qa_note=qa_note,
            style_chunks=style_chunks,
            beats=beats,
            existing_text=context_text,
            mode="continue",
            text_after_cursor=text_after_cursor.strip(),
            style_mode=style_mode,
            constraints=constraints,
        )

        # Construct user message with cursor marker when inserting mid-text
        if text_after_cursor.strip():
            draft_block = (
                f"{context_text.rstrip()}\n\n✍ ← INSERT HERE\n\n"
                f"--- Text that continues AFTER your insertion ---\n{text_after_cursor.strip()}\n---"
            )
        else:
            draft_block = context_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_instruction}\n\n---\n{draft_block}"},
        ]
        return await chat_stream(messages)

    async def rewrite_passage(
        self,
        project_id: str,
        passage: str,
        chapter_id: str = "",
        user_instruction: str = "Rewrite this passage in the same style.",
        style_mode: str = "preserve_tone_rhythm",
        language: str = "en",
        beats: str = "",
        editor_note: str = "",
        qa_note: str = "",
    ) -> AsyncIterator[str]:
        """Stream rewritten passage tokens — uses the Writer role skill, same priority as chat Writer."""
        from norvel_writer.llm.langchain_bridge import chat_stream
        from norvel_writer.llm.prompt_builder import _lang_display

        lang_display = _lang_display(language)

        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=passage,
            n_results=6,
            doc_types=["codex", "beats", "research", "notes"],
        )
        style_results = await self._pm.retrieve_style_examples(
            project_id=project_id,
            query=passage,
            n_results=4,
        )
        rag_context = "\n\n---\n\n".join(r["text"] for r in rag_results)
        style_chunks = [r["text"] for r in style_results]

        proj = self._pm.get_project(project_id)
        persona = (proj.get("persona") or "").strip() if proj else ""

        image_context = _fetch_image_context(self._pm._db, project_id, chapter_id)

        system_prompt = _build_writer_system_prompt(
            lang_display=lang_display,
            persona=persona,
            editor_note=editor_note,
            rag_context=rag_context,
            image_context=image_context,
            qa_note=qa_note,
            style_chunks=style_chunks,
            beats=beats,
            existing_text="",
            mode="rewrite",
            style_mode=style_mode,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_instruction}\n\n---\n{passage}"},
        ]
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
        qa_note: str = "",
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
            rag_doc_types = ["codex", "beats", "research", "notes"]

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

        # ── Image description context (project-level + chapter-level) ─────────
        # Priority: same as codex — injected alongside rag_context
        image_context = _fetch_image_context(self._pm._db, project_id, resolved_chapter_id)

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

        # ── Load role definition from TOML file ───────────────────────────
        from norvel_writer.core.role_loader import load_role

        def _bullets(items: list, prefix: str = "•") -> str:
            return "\n".join(f"{prefix} {item}" for item in items)

        # ── Build role-specific system prompt ──────────────────────────────
        if role == "editor":
            rd = load_role("editor")
            background = rd.get("identity", {}).get("background", "").strip() or (
                "You are a senior professional book editor with 20+ years of experience "
                "at major publishing houses."
            )
            focus_areas = rd.get("focus", {}).get("areas", [
                "Narrative structure & pacing — does the chapter flow? are transitions clear?",
                "Character voice & consistency — does each character sound distinct?",
                "Dialogue — is it natural? does it serve the scene? reveal character?",
                "Show vs tell — flag where emotions/actions are told rather than shown",
                "Prose clarity & style — unclear sentences, overwriting, repetition",
                "Tension & reader engagement — does the chapter hold attention?",
                "Marketability — does it meet genre and audience expectations?",
            ])
            feedback_style = rd.get("feedback", {}).get("style", "").strip() or (
                "Quote directly from the chapter text to anchor each point.\n"
                "Explain WHY something works or doesn't.\n"
                "Give specific, actionable revision suggestions.\n"
                "Balance positive observations with areas for improvement."
            )

            system_prompt = (
                f"{background}\n\n"
                "Your ONLY job right now is to give the author honest, constructive, "
                "publisher-level editorial feedback on the chapter text provided below.\n"
                "DO NOT discuss the codex, world-building documents, or project metadata. "
                "Focus exclusively on the prose, structure, and craft of the chapter itself.\n\n"
                f"Your focus areas:\n{_bullets(focus_areas)}\n\n"
                f"{feedback_style}\n\n"
                + lang_line
            )
            if not chapter_text:
                system_prompt += (
                    "\n\n⚠️ No chapter content is loaded. Tell the user: "
                    "Please open and select a chapter in the editor panel first, "
                    "then I can give you editorial feedback on its content."
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
                system_prompt += f"\n\n## Planned Chapter Beats (for context only)\n{rag_context}"
            if image_context:
                system_prompt += f"\n\n## Visual Reference Descriptions (for context only)\n{image_context}"

        elif role == "writer":
            # Use the shared writer prompt builder — identical to Draft AI
            # Beats for the resolved chapter are fetched from the DB
            ch_beats = ""
            if resolved_chapter_id:
                try:
                    ch_row = self._pm.get_chapter(resolved_chapter_id)
                    ch_beats = (ch_row.get("beats") or "").strip() if ch_row else ""
                except Exception:
                    pass

            system_prompt = _build_writer_system_prompt(
                lang_display=lang_display,
                persona=persona,
                editor_note=editor_note,
                rag_context=rag_context,
                image_context=image_context,
                qa_note=qa_note,
                style_chunks=style_chunks,
                beats=ch_beats,
                existing_text=chapter_text,
                mode="chat",
            )

        else:  # qa
            rd = load_role("qa")
            background = rd.get("identity", {}).get("background", "").strip() or (
                "You are a meticulous QA (Quality Assurance) reviewer for creative fiction. "
                "Your role is to systematically audit the chapter text for errors and "
                "inconsistencies — not to rewrite, only to identify issues."
            )
            check_areas = rd.get("checks", {}).get("areas", [
                "Codex compliance — character names, traits, abilities, relationships, world rules",
                "Beat adherence — does the chapter follow the planned beats in order?",
                "Logic & causality — do events follow logically? are decisions motivated?",
                "Continuity — timeline, locations, objects, relationships vs earlier chapters",
                "Descriptive consistency — settings, appearances, physical objects",
                "Chaos / confusion — unclear blocking, confusing POV shifts, hard-to-follow scenes",
            ])
            report_format = rd.get("report", {}).get("format", "").strip() or (
                "Format your response as a structured report:\n"
                "- ✅ PASS items\n"
                "- ⚠️ ISSUES (quote the exact location + explain the problem + suggest the fix)\n"
                "- 📋 SUMMARY (overall verdict and top 3 priorities to fix)"
            )

            system_prompt = (
                f"{background}\n\n"
                f"Check all of the following:\n{_bullets(check_areas)}\n\n"
                f"{report_format}\n\n"
                + lang_line
            )
            if not chapter_text:
                system_prompt += (
                    "\n\n⚠️ No chapter content is loaded. Tell the user: "
                    "Please open and select a chapter in the editor panel first, "
                    "then I can run a QA check on its content."
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
                system_prompt += f"\n\n## Reference Material to Check Against (Codex / Beats)\n{rag_context}"
            if image_context:
                system_prompt += f"\n\n## Visual Reference Descriptions (check visual consistency)\n{image_context}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        return await chat_stream(messages)


def _fetch_image_context(db: Any, project_id: str, chapter_id: str = "") -> str:
    """Query project + chapter image descriptions from the DB."""
    parts: List[str] = []
    try:
        proj_imgs = db.execute(
            "SELECT title, ai_description FROM project_images "
            "WHERE project_id=? AND ai_description != '' ORDER BY created_at ASC",
            (project_id,),
        )
        for img in proj_imgs:
            label = (img["title"] or "Visual Reference").strip()
            parts.append(f"[Project Visual — {label}]\n{img['ai_description']}")
    except Exception as exc:
        log.debug("_fetch_image_context project: %s", exc)

    if chapter_id:
        try:
            ch_imgs = db.execute(
                "SELECT title, ai_description FROM chapter_images "
                "WHERE chapter_id=? AND ai_description != '' ORDER BY created_at ASC",
                (chapter_id,),
            )
            for img in ch_imgs:
                label = (img["title"] or "Chapter Image").strip()
                parts.append(f"[Chapter Visual — {label}]\n{img['ai_description']}")
        except Exception as exc:
            log.debug("_fetch_image_context chapter: %s", exc)

    return "\n\n---\n\n".join(parts)


def _build_writer_system_prompt(
    lang_display: str,
    persona: str,
    editor_note: str,
    rag_context: str,
    image_context: str,
    qa_note: str,
    style_chunks: List[str],
    beats: str,
    existing_text: str,
    mode: str,          # "continue" | "rewrite"
    text_after_cursor: str = "",
    style_mode: str = "",
    constraints: Optional[List[str]] = None,
) -> str:
    """
    Build the Writer system prompt with the same priority ordering and TOML role
    as the chat_with_context Writer role — so Draft AI and Chat Writer are identical.

    Priority order (strict):
      1. User's request (in the user message)
      2. Persona / voice
      3. Editor suggestions
      4. Codex / beats / notes / research / visual references
      5. QA issues
      6. Style samples
    """
    from norvel_writer.core.role_loader import load_role

    def _bullets(items: list, prefix: str = "-") -> str:
        return "\n".join(f"{prefix} {item}" for item in items)

    rd = load_role("writer")
    background = rd.get("identity", {}).get("background", "").strip() or (
        "You are a skilled professional co-author and writing collaborator. "
        "Your role is to help the author write new content — scenes, dialogue, "
        "descriptions, chapter continuations — while faithfully following their "
        "established style, voice, characters, and story rules."
    )
    priorities = rd.get("priorities", {})
    p1 = priorities.get("p1", "The author's CURRENT REQUEST — what you are being asked to do right now")
    p2 = priorities.get("p2", "The author's persona & voice instructions — overrides stylistic choices")
    p3 = priorities.get("p3", "Pinned editor suggestions — apply every improvement point to your writing")
    p4 = priorities.get("p4", "All memory documents — codex, beats, notes, research, visual references — follow them strictly")
    p5 = priorities.get("p5", "Pinned QA issues — fix every flagged problem; do not reintroduce them")
    p6 = priorities.get("p6", "Style samples — match the established tone, rhythm, and sentence structure")
    rules = rd.get("rules", {}).get("items", [
        "Output ONLY the prose — no beat labels, no beat numbers, no headings, no annotations, nothing except the story text",
        "Do NOT add meta-commentary, preambles, or explain your choices",
        "Maintain the established POV and tense throughout",
        "Write directly usable prose — not outlines or summaries",
        "Cover each beat EXACTLY ONCE — once written, move on",
        "NEVER repeat a sentence, paragraph, or scene you have already written",
        "When you reach the final beat, end the chapter naturally and STOP",
    ])

    # Mode-specific task description
    if mode == "continue":
        if text_after_cursor:
            task_line = (
                "Write new content to INSERT at the cursor position (marked ✍ in the draft below). "
                "Your text must flow naturally FROM the content above AND lead smoothly INTO the text that follows."
            )
        else:
            task_line = "Continue the story directly from where the current draft ends."
    elif mode == "rewrite":
        task_line = (
            f"Rewrite the passage provided by the author. "
            f"Style guidance: {style_mode.replace('_', ' ')}. "
            "Preserve the narrative content and plot events, but produce SUBSTANTIALLY DIFFERENT prose. "
            "Improve sentence structure, word choice, rhythm, imagery, and overall prose quality. "
            "The rewrite MUST NOT be a near-copy, a paraphrase, or a lightly edited version of the original. "
            "If the original has weak verbs — strengthen them. If it tells instead of showing — show. "
            "If sentences are repetitive or flat — vary and energise them. "
            "Return ONLY the rewritten prose — no preamble, no explanation, no sign-off, "
            "and do NOT reproduce the original text."
        )
    else:  # chat
        task_line = (
            "Collaborate with the author on their request. "
            "Write only what is asked — new scenes, dialogue, descriptions, revisions, or other content as directed."
        )

    # Rewrite mode: append critical differentiator rule
    rewrite_rules: List[str] = []
    if mode == "rewrite":
        rewrite_rules = [
            "⚠ REWRITE RULE: Your output MUST be substantially and noticeably different from the original passage",
            "Do NOT copy sentences verbatim or reproduce the original structure word-by-word",
            "Do NOT produce a superficial paraphrase — genuinely improve the prose",
            "You may restructure paragraphs, change sentence order, alter imagery, or vary rhythm freely",
            "The STORY EVENTS and CHARACTER ACTIONS must remain the same — only the prose changes",
        ]

    all_rules = rules + rewrite_rules if rewrite_rules else rules

    prompt = (
        f"{background}\n\n"
        f"Task: {task_line}\n\n"
        "You MUST honour the following (in strict priority order):\n"
        f"1. {p1}\n"
        f"2. {p2}\n"
        f"3. {p3}\n"
        f"4. {p4}\n"
        f"5. {p5}\n"
        f"6. {p6}\n\n"
        f"When writing:\n{_bullets(all_rules)}\n\n"
        f"ALWAYS write in {lang_display}."
    )

    # Priority 2 — Persona
    if persona:
        prompt += (
            f"\n\n╔══════════════════════════════════════════╗\n"
            f"║  PRIORITY 2 — AUTHOR'S VOICE & PERSONA     ║\n"
            f"╚══════════════════════════════════════════╝\n"
            f"The following persona instructions define this author's unique voice and MUST be "
            f"followed above all else. They override any style samples (Priority 6) and any "
            f"stylistic defaults in your training.\n\n"
            f"{persona}"
        )

    # Priority 3 — Editor suggestions
    if editor_note:
        prompt += (
            f"\n\n╔══════════════════════════════════════════╗\n"
            f"║  PRIORITY 3 — EDITOR SUGGESTIONS (APPLY ALL) ║\n"
            f"╚══════════════════════════════════════════╝\n"
            f"{editor_note}\n"
            f"(Every point above must be addressed in your writing.)"
        )

    # Priority 4 — Memory (codex / beats / notes / research) + visual references
    combined_memory = rag_context
    if image_context:
        sep = "\n\n---\n\n" if combined_memory else ""
        combined_memory += f"{sep}### Visual Reference Descriptions\n{image_context}"
    if combined_memory:
        prompt += f"\n\n## PRIORITY 4 — Project Memory (Codex / Beats / Notes / Research / Visuals)\n{combined_memory}"

    # Priority 5 — QA issues
    if qa_note:
        prompt += (
            f"\n\n╔══════════════════════════════════════════╗\n"
            f"║  PRIORITY 5 — QA ISSUES (FIX ALL OF THESE) ║\n"
            f"╚══════════════════════════════════════════╝\n"
            f"{qa_note}\n"
            f"(Every issue above must be corrected. Do not reintroduce any of them.)"
        )

    # Priority 6 — Style samples
    if style_chunks:
        defer_note = " (secondary — defer to Priority 2 Persona if one is set)" if persona else ""
        prompt += f"\n\n## PRIORITY 6 — Style Reference Samples{defer_note}\n"
        prompt += "Use these excerpts as stylistic reference. Match their tone, rhythm, and sentence structure:\n"
        for chunk in style_chunks:
            prompt += f"---\n{chunk}\n"

    # Chapter beats (belongs with memory context but shown separately for clarity)
    if beats:
        prompt += (
            f"\n\n## Chapter Beats — FOLLOW THESE EXACTLY\n"
            f"Cover each beat in order. Do NOT skip any. Do NOT add unlisted beats.\n\n"
            f"{beats}"
        )

    # Existing draft / chapter text (context — always shown when present)
    if existing_text:
        label = "Current Draft (for context — do NOT repeat this)" if mode == "continue" else "Current Chapter Content (existing draft — for context only)"
        prompt += f"\n\n## {label}\n{existing_text}"

    # Continue-specific: style mode + constraints
    if mode == "continue":
        prompt += f"\n\nStyle guidance: {style_mode.replace('_', ' ')}"
        if constraints:
            prompt += "\n\n## Additional Constraints\n" + _bullets(constraints)

    return prompt


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

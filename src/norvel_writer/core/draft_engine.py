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

    async def _rag_retrieve(
        self,
        project_id: str,
        chapter_id: Optional[str],
        query: str,
        n_results: int,
        doc_types: Optional[List[str]] = None,
        include_project: bool = False,
    ) -> List[Dict]:
        """Retrieve RAG chunks with chapter-first priority.

        Default: chapter-scoped only.  Falls back to project-level when the
        chapter has no matching documents.  If include_project=True, merges
        both (chapter results first = higher priority in _cap_rag).
        """
        ch_results: List[Dict] = []
        if chapter_id:
            ch_results = await self._pm.retrieve_context(
                project_id=project_id,
                query=query,
                n_results=n_results,
                doc_types=doc_types,
                chapter_id=chapter_id,
                scope="chapter",
            )

        proj_results: List[Dict] = []
        if include_project or not ch_results:
            proj_results = await self._pm.retrieve_context(
                project_id=project_id,
                query=query,
                n_results=n_results,
                doc_types=doc_types,
                scope="project",
            )

        if include_project:
            # Chapter docs first → they win the _cap_rag budget competition
            return ch_results + proj_results
        return ch_results if ch_results else proj_results

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
        from norvel_writer.llm.langchain_bridge import chat_stream, get_context_limits
        from norvel_writer.llm.prompt_builder import _lang_display
        from norvel_writer.utils.text_utils import truncate_to_tokens

        limits = get_context_limits()
        lang_display = _lang_display(language)
        last_para = _last_paragraphs(current_text, n_tokens=512)

        # Detect write-from-beats mode: no existing text but beats are present.
        is_beats_mode = bool(beats.strip()) and not current_text.strip()

        # ── Previous-chapter tail (beats mode only) ────────────────────────
        # If this is not the first chapter of the project, we fetch the last
        # ~600 tokens of the previous chapter's accepted draft and inject it
        # as a "where the story left off" anchor.  This prevents the model from
        # inventing a fresh opening instead of continuing naturally, while still
        # letting the beats define what actually happens next.
        prev_chapter_tail = ""
        if is_beats_mode:
            try:
                from norvel_writer.utils.text_utils import strip_html
                all_chapters = self._pm.list_chapters(project_id)
                ch_ids = [c["id"] for c in all_chapters]
                if chapter_id in ch_ids:
                    idx = ch_ids.index(chapter_id)
                    if idx > 0:
                        prev_ch_id = ch_ids[idx - 1]
                        prev_draft = self._pm.get_accepted_draft(prev_ch_id)
                        if prev_draft:
                            prev_raw = strip_html(prev_draft.get("content") or "")
                            # Take only the tail — the transition / ending paragraphs
                            tail_chars = 2400  # ≈ 600 tokens
                            prev_chapter_tail = prev_raw[-tail_chars:].strip()
                            log.debug(
                                "beats: injecting %d chars from previous chapter %r",
                                len(prev_chapter_tail), prev_ch_id,
                            )
            except Exception as exc:
                log.warning("beats: could not fetch previous chapter: %s", exc)

        # RAG query strategy — mirrors chat_with_context which uses the user's
        # actual words as the semantic query, not just the last paragraph.
        # • beats mode  → beats text (what's about to be written)
        # • normal mode → combine user instruction + last paragraph so that
        #   a request like "write a tense confrontation" pulls the right characters
        #   and world rules, not just whatever prose was written last.
        _default_instr = "Continue the story from where it left off."
        if is_beats_mode:
            rag_query = beats
        elif user_instruction and user_instruction.strip() != _default_instr:
            rag_query = f"{user_instruction}\n{last_para}".strip()
        else:
            rag_query = last_para

        # RAG — fetch extra candidates then cap to token budget so small local
        # models aren't silently overflowed (budget ≈ 3500 tok ≈ 14 000 chars).
        _all_types = active_doc_types or ["codex", "beats", "research", "notes"]

        if is_beats_mode:
            # In beats mode, codex documents can be very large and are only
            # useful when they directly relate to the beats being written.
            # Strategy: retrieve beats/notes/research unconditionally (they are
            # always structurally relevant), then retrieve codex separately and
            # filter out chunks that are semantically distant from the beats query.
            _cx_threshold = limits["codex_distance_threshold"]
            _non_codex    = [t for t in _all_types if t != "codex"]
            _wants_codex  = "codex" in _all_types

            if _non_codex:
                _bn_results = await self._rag_retrieve(
                    project_id, chapter_id, rag_query, 8, _non_codex,
                )
            else:
                _bn_results = []

            _cx_results = []
            if _wants_codex:
                _cx_raw = await self._rag_retrieve(
                    project_id, chapter_id, rag_query, 10, ["codex"],
                )
                # Only keep codex chunks close enough to the beats query
                _cx_results = [r for r in _cx_raw if r.get("distance", 1.0) <= _cx_threshold]
                log.debug(
                    "beats RAG: %d/%d codex chunks passed distance threshold %.2f",
                    len(_cx_results), len(_cx_raw), _cx_threshold,
                )

            # Merge and re-sort by distance so _cap_rag's relevance ordering is
            # correct across both sub-lists.  Without this, non-codex chunks
            # (appended first) would consume the entire budget and all codex
            # chunks would be silently abandoned.
            rag_results = sorted(
                _bn_results + _cx_results,
                key=lambda r: r.get("distance", 0.0),
            )
        else:
            rag_results = await self._rag_retrieve(
                project_id, chapter_id, rag_query, 14, _all_types,
            )

        style_results = await self._pm.retrieve_style_examples(
            project_id=project_id,
            query=rag_query,
            n_results=8,
        )
        # In beats mode, apply the codex distance threshold across the entire
        # merged list so that irrelevant non-codex chunks (research / notes with
        # high distance) don't crowd out relevant codex entries.
        # In normal mode no distance filter is applied (max_distance default=1.0).
        _rag_max_dist = limits["codex_distance_threshold"] if is_beats_mode else 1.0
        rag_context = "\n\n---\n\n".join(
            r["text"] for r in _cap_rag(
                rag_results,
                budget_tokens=limits["rag_budget"],
                max_distance=_rag_max_dist,
            )
        )
        style_chunks = [r["text"] for r in _cap_rag(style_results, budget_tokens=limits["style_budget"])]

        proj = self._pm.get_project(project_id)
        persona = (proj.get("persona") or "").strip() if proj else ""

        # Image descriptions — same as Writer chat
        image_context = _fetch_image_context(self._pm._db, project_id, chapter_id)

        context_text = truncate_to_tokens(current_text, max_tokens=limits["text_budget"])

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
            mode="beats" if is_beats_mode else "continue",
            text_after_cursor=text_after_cursor.strip(),
            style_mode=style_mode,
            constraints=constraints,
            prev_chapter_tail=prev_chapter_tail,
        )

        # Construct user message with cursor marker when inserting mid-text
        if text_after_cursor.strip():
            draft_block = (
                f"{context_text.rstrip()}\n\n✍ ← INSERT HERE\n\n"
                f"--- Text that continues AFTER your insertion ---\n{text_after_cursor.strip()}\n---"
            )
        else:
            draft_block = context_text

        # In beats mode the user message doubles as a final beats reminder.
        # Because small models weight the most recently seen tokens highly,
        # repeating the constraint here (after all the system-prompt context)
        # significantly reduces the chance of the model going off-script.
        if is_beats_mode:
            beat_fence = (
                "⚠ BEATS CONSTRAINT: Write ONLY the scenes and events described in the "
                "Chapter Blueprint above. Do NOT invent new scenes, characters, or plot "
                "points that are not in the beats. Start at Beat 1. Stop after the final beat."
            )
            user_content = f"{user_instruction}\n\n{beat_fence}"
            if draft_block.strip():
                user_content += f"\n\n---\n{draft_block}"
        else:
            user_content = f"{user_instruction}\n\n---\n{draft_block}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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
        from norvel_writer.llm.langchain_bridge import chat_stream, get_context_limits
        from norvel_writer.llm.prompt_builder import _lang_display

        limits = get_context_limits()
        lang_display = _lang_display(language)

        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=passage,
            n_results=14,
            doc_types=["codex", "beats", "research", "notes"],
        )
        style_results = await self._pm.retrieve_style_examples(
            project_id=project_id,
            query=passage,
            n_results=8,
        )
        rag_context = "\n\n---\n\n".join(
            r["text"] for r in _cap_rag(rag_results, budget_tokens=limits["rag_budget"])
        )
        style_chunks = [r["text"] for r in _cap_rag(style_results, budget_tokens=limits["style_budget"])]

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
        from norvel_writer.llm.langchain_bridge import chat_complete, get_context_limits
        from norvel_writer.llm.prompt_builder import _lang_display
        from norvel_writer.utils.text_utils import truncate_to_tokens

        limits = get_context_limits()
        text = truncate_to_tokens(chapter_text, max_tokens=limits["text_budget"])
        lang = _lang_display(language)
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a writing assistant. "
                    f"Summarise the following chapter in 1-3 sentences. "
                    f"IMPORTANT: Write your entire response in {lang}. "
                    f"Do not respond in English if the target language is not English. "
                    f"Be concise and factual."
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
        from norvel_writer.llm.langchain_bridge import chat_complete, get_context_limits
        from norvel_writer.llm.prompt_builder import _lang_display

        limits = get_context_limits()
        rag_results = await self._pm.retrieve_context(
            project_id=project_id,
            query=passage,
            n_results=14,
            doc_types=["codex", "beats"],
        )
        context = "\n\n---\n\n".join(
            r["text"] for r in _cap_rag(rag_results, budget_tokens=limits["rag_budget"])
        )
        lang = _lang_display(language)

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a continuity checker. "
                    f"IMPORTANT: Write your ENTIRE response in {lang}. "
                    f"Do not respond in English if the target language is not English. "
                    f"Given the project reference material below, identify any contradictions "
                    f"or inconsistencies in the passage. "
                    f"If no issues are found, say so briefly in {lang}."
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
        Responds in the project language by default; switches if the user
        explicitly requests a different language (e.g. "respond in Japanese").
        """
        from norvel_writer.llm.langchain_bridge import chat_stream, get_context_limits
        from norvel_writer.utils.text_utils import strip_html, truncate_to_tokens
        from norvel_writer.llm.prompt_builder import _lang_display

        limits = get_context_limits()

        # ── Resolve response language ──────────────────────────────────────
        # Priority: explicit override in the user's message > project language
        override = _detect_language_override(question)
        effective_lang = override if override else (language or "en")
        lang_display = _lang_display(effective_lang)
        log.debug("chat: lang=%r (override=%r, project=%r)", effective_lang, override, language)

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
                    chapter_text = truncate_to_tokens(strip_html(raw), max_tokens=limits["text_budget"])
                    log.debug("chat: loaded chapter %r (%d chars)", chapter_title, len(chapter_text))
                else:
                    log.debug("chat: no accepted draft for chapter %r", resolved_chapter_id)
            except Exception as exc:
                log.warning("chat: chapter load failed for %r: %s", resolved_chapter_id, exc)

        # ── Detect topic focus from question keywords ───────────────────────
        # Lets the user say "check the codex" or "第一章的节拍" and get the right content
        # Keywords cover: English, Chinese (Simplified/Traditional), Japanese, Korean,
        # French, German, Spanish, Italian, Portuguese
        q_lower = question.lower()
        topic_wants_codex = any(kw in q_lower for kw in [
            # English
            "codex", "character", "world", "lore", "rule", "setting", "worldbuilding",
            # Chinese (Simplified + Traditional)
            "世界观", "角色", "设定", "人物", "规则", "世界設定", "人設",
            # Japanese
            "コーデックス", "キャラ", "世界設定", "設定", "キャラクター",
            # Korean
            "세계관", "캐릭터", "설정", "인물", "규칙",
            # French
            "personnage", "monde", "univers", "règle", "cadre",
            # German
            "charakter", "welt", "regel", "einstellung", "weltenbau",
            # Spanish
            "personaje", "mundo", "regla", "ambientación",
            # Italian
            "personaggio", "mondo", "regola", "ambientazione",
            # Portuguese
            "personagem", "mundo", "regra", "ambientação",
        ])
        topic_wants_beats = any(kw in q_lower for kw in [
            # English
            "beat", "plot", "outline", "structure", "story arc", "pacing",
            # Chinese
            "节拍", "情节", "大纲", "结构", "節拍", "情節",
            # Japanese
            "ビート", "プロット", "構成", "あらすじ",
            # Korean
            "비트", "플롯", "구성", "개요",
            # French
            "intrigue", "structure", "trame", "plan",
            # German
            "handlung", "struktur", "gliederung", "plot",
            # Spanish
            "trama", "estructura", "argumento",
            # Italian
            "trama", "struttura", "intreccio",
            # Portuguese
            "trama", "estrutura", "enredo",
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

        # ── Pre-load chapter beats (writer role only) ──────────────────────
        # Must happen BEFORE the RAG query so we can use the beats text as the
        # semantic query when write-from-beats intent is detected — identical to
        # the strategy used in continue_draft's beats mode.
        ch_beats = ""
        if role == "writer" and resolved_chapter_id:
            try:
                _ch_row_pre = self._pm.get_chapter(resolved_chapter_id)
                ch_beats = (_ch_row_pre.get("beats") or "").strip() if _ch_row_pre else ""
            except Exception:
                pass

        # ── Intent detection (writer role) ────────────────────────────────
        # Detect two distinct write intents so the correct prompt mode is used:
        #   • rewrite  → chapter text is the target (rewrite the existing draft)
        #   • write    → beats are the target (generate fresh prose from beats)
        # Both are detected here so the RAG query can be adjusted before retrieval.
        _rewrite_kws = [
            "rewrite", "re-write", "re write", "rewrite the chapter",
            "重写", "改写", "重新写", "重新改写",
            "réécrire", "umschreiben", "riscrivere", "reescribir",
        ]
        _write_kws = [
            # English
            "write the chapter", "write this chapter", "write chapter",
            "write from beats", "write from my beats", "write from the beats",
            "write based on beats", "generate the chapter", "generate chapter",
            "draft the chapter", "draft from beats",
            # Chinese (Simplified + Traditional)
            "写这章", "写章节", "写这个章节", "按节拍写", "根据节拍写", "写第",
            "寫這章", "寫章節", "按節拍寫", "根據節拍寫",
            # Japanese
            "章を書いて", "チャプターを書いて",
            # Korean
            "챕터를 써줘", "장을 써줘", "챕터 작성",
            # French
            "écrire le chapitre", "écris le chapitre", "rédige le chapitre",
            # German
            "schreibe das kapitel", "schreib das kapitel",
            # Spanish
            "escribe el capítulo", "redacta el capítulo",
            # Italian
            "scrivi il capitolo",
            # Portuguese
            "escreve o capítulo", "escreva o capítulo",
        ]
        _is_chapter_rewrite = bool(
            role == "writer" and chapter_text
            and any(kw in q_lower for kw in _rewrite_kws)
        )
        _is_write_from_beats = bool(
            role == "writer" and ch_beats
            and not _is_chapter_rewrite
            and any(kw in q_lower for kw in _write_kws)
        )

        # Detect explicit request for project-level (centre) memory.
        # By default the writer uses chapter memory only; project docs are included
        # only when the user explicitly asks for them.
        _project_memory_kws = [
            # English
            "project document", "project memory", "center memory", "centre memory",
            "global memory", "all documents", "all my documents",
            "include project", "use project", "project codex", "project notes",
            "project knowledge", "project files",
            # Chinese Simplified
            "项目文档", "项目记忆", "中央记忆", "全局记忆", "项目资料",
            "项目笔记", "项目设定", "包含项目", "使用项目",
            # Chinese Traditional
            "項目文檔", "項目記憶", "中央記憶", "全局記憶", "項目資料",
            # Japanese
            "プロジェクト文書", "プロジェクトメモリ",
            # Korean
            "프로젝트 문서", "전체 문서",
        ]
        _wants_project_memory = role == "writer" and any(
            kw in q_lower for kw in _project_memory_kws
        )

        # ── RAG query & retrieval ──────────────────────────────────────────
        # write-from-beats: use beats text as query (same as continue_draft)
        # otherwise:        chapter-title-prefixed question
        if _is_write_from_beats:
            rag_query = ch_beats
        else:
            rag_query = f"{chapter_title}: {question}" if chapter_title else question

        # write-from-beats: split codex / non-codex retrieval with distance
        # filtering — identical to continue_draft's beats-mode strategy.
        if _is_write_from_beats:
            _cx_threshold = limits["codex_distance_threshold"]
            _non_codex    = [t for t in rag_doc_types if t != "codex"]
            _wants_codex  = "codex" in rag_doc_types
            _bn_results: list = []
            if _non_codex:
                _bn_results = await self._rag_retrieve(
                    project_id, resolved_chapter_id, rag_query, 8, _non_codex,
                    include_project=_wants_project_memory,
                )
            _cx_results: list = []
            if _wants_codex:
                _cx_raw = await self._rag_retrieve(
                    project_id, resolved_chapter_id, rag_query, 10, ["codex"],
                    include_project=_wants_project_memory,
                )
                _cx_results = [r for r in _cx_raw if r.get("distance", 1.0) <= _cx_threshold]
                log.debug(
                    "chat-beats RAG: %d/%d codex chunks passed distance %.2f",
                    len(_cx_results), len(_cx_raw), _cx_threshold,
                )
            rag_results = sorted(
                _bn_results + _cx_results,
                key=lambda r: r.get("distance", 0.0),
            )
            _rag_max_dist = _cx_threshold
        else:
            rag_results = await self._rag_retrieve(
                project_id, resolved_chapter_id, rag_query, 14, rag_doc_types,
                include_project=_wants_project_memory,
            )
            _rag_max_dist = 1.0

        rag_context = "\n\n---\n\n".join(
            r["text"] for r in _cap_rag(
                rag_results,
                budget_tokens=limits["rag_budget"],
                max_distance=_rag_max_dist,
            )
        )

        # ── Image description context (project-level + chapter-level) ─────────
        # Priority: same as codex — injected alongside rag_context
        image_context = _fetch_image_context(self._pm._db, project_id, resolved_chapter_id)

        # ── Extra context for Writer role ──────────────────────────────────
        persona = ""
        style_chunks: List[str] = []
        if role == "writer":
            proj = self._pm.get_project(project_id)
            persona = (proj.get("persona") or "").strip() if proj else ""
            # Style query: for chapter rewrites the user's instruction text
            # ("rewrite this chapter…") has zero semantic similarity to uploaded
            # prose samples.  Use the chapter text itself as the retrieval anchor
            # so uploaded style references (e.g. a Dumas novel) are found by
            # content similarity.  For all other paths keep rag_query.
            _style_query = (
                (chapter_text or "")[:2000] if (_is_chapter_rewrite and chapter_text)
                else rag_query
            )
            style_results = await self._pm.retrieve_style_examples(
                project_id=project_id,
                query=_style_query,
                n_results=8,
            )
            style_chunks = [r["text"] for r in _cap_rag(style_results, budget_tokens=limits["style_budget"])]

        # ── Language instruction ───────────────────────────────────────────
        lang_line = (
            f"ALWAYS respond in the same language the user writes in. "
            f"The user appears to be writing in {lang_display} — respond in {lang_display}. "
            f"This applies to EVERYTHING in your response: content, section headers, "
            f"structural labels (e.g. PROBLEM / WHY / SUGGESTION / PASS / ISSUES / SUMMARY "
            f"/ 问题 / 原因 / 建议), and any meta-commentary. "
            f"Do NOT use English labels or headings if the response language is not English."
        )

        # ── Load role definition from TOML file ───────────────────────────
        from norvel_writer.core.role_loader import load_role

        def _bullets(items: list, prefix: str = "•") -> str:
            return "\n".join(f"{prefix} {item}" for item in items)

        # ── Build role-specific system prompt ──────────────────────────────
        # user_message is set per-role; for editor/QA it's always the plain
        # question; for writer it may be enhanced for rewrite requests.
        user_message = question

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
                "Feedback rules — STRICTLY follow these:\n"
                "- Report ONLY what needs to be improved — do NOT praise or say what is working well\n"
                "- For every issue, use this structure (translate the labels into the response language):\n"
                "    ▸ PROBLEM / 问题 / 問題: Quote the exact passage or describe the specific moment\n"
                "    ▸ WHY / 原因 / 理由: Explain clearly why this weakens the writing\n"
                "    ▸ SUGGESTION / 建议 / 提案: Give a concrete, specific revision\n"
                "- If the user asks about a specific aspect, address that first\n"
                "- Be direct and specific — vague feedback is not acceptable\n"
                "- Do NOT add a summary of positives at the end\n"
                "- Write ALL labels and content in the same language the user writes in"
            )

            system_prompt = (
                # Language instruction FIRST — small models must see this before the English background
                f"{lang_line}\n\n"
                f"{background}\n\n"
                "Your ONLY job right now is to identify what needs IMPROVEMENT in the chapter "
                "and tell the author exactly how to fix it.\n"
                "Do NOT praise what is working — the author wants actionable improvements only.\n"
                "Do NOT discuss the codex, world-building documents, or project metadata. "
                "Focus exclusively on the prose, structure, and craft of the chapter itself.\n\n"
                f"Your focus areas:\n{_bullets(focus_areas)}\n"
                f"(When referencing these areas in your response, translate their names into {lang_display}.)\n\n"
                f"{feedback_style}"
            )
            if not chapter_text:
                system_prompt += (
                    f"\n\n⚠️ No chapter content is loaded. "
                    f"Tell the user IN {lang_display}: "
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
            # ch_beats already loaded above (before RAG query).
            # Intent flags (_is_chapter_rewrite, _is_write_from_beats) also
            # computed above.  Three mutually exclusive paths:
            #   1. rewrite       — user wants to rewrite the existing draft
            #   2. write-beats   — user wants fresh prose written from beats
            #   3. chat          — general collaborative writing / discussion

            if _is_chapter_rewrite:
                # ── Path 1: Rewrite ───────────────────────────────────────
                # Chapter text goes into the USER message (most prominent
                # position for small models) rather than buried in system prompt.
                system_prompt = _build_writer_system_prompt(
                    lang_display=lang_display,
                    persona=persona,
                    editor_note=editor_note,
                    rag_context=rag_context,
                    image_context=image_context,
                    qa_note=qa_note,
                    style_chunks=style_chunks,
                    beats=ch_beats,
                    existing_text="",          # NOT in system prompt
                    mode="rewrite",
                    # If the user has uploaded style samples, write toward them.
                    # "preserve_tone_rhythm" would lock the model to the old draft's
                    # style, defeating the purpose of uploading a style reference.
                    style_mode="inspired_by" if style_chunks else "preserve_tone_rhythm",
                )
                _en_block = (
                    f"\n\nApply ALL editor suggestions above to the rewritten text."
                    if editor_note else ""
                )
                user_message = (
                    f"{question}{_en_block}\n\n"
                    f"════════════════════════════════════════\n"
                    f"CHAPTER TO REWRITE: {chapter_title or 'Current Chapter'}\n"
                    f"════════════════════════════════════════\n"
                    f"{chapter_text}\n"
                    f"════════════════════════════════════════\n"
                    f"Rewrite the entire chapter above. Produce completely new prose "
                    f"covering the same events and scenes. Do NOT summarise — "
                    f"write full, publication-quality prose from start to finish."
                )

            elif _is_write_from_beats:
                # ── Path 2: Write from Beats ──────────────────────────────
                # Identical to Draft AI's "Write from Beats" button:
                #   • mode="beats"  → beats FIRST in prompt + FINAL INSTRUCTION at end
                #   • No existing_text (writing fresh, not continuing)
                #   • Previous chapter tail injected as continuity anchor
                #   • Beats fence repeated in user message (recency bias fix)
                prev_chapter_tail = ""
                try:
                    from norvel_writer.utils.text_utils import strip_html as _strip_html
                    _all_ch = self._pm.list_chapters(project_id)
                    _ch_ids = [c["id"] for c in _all_ch]
                    if resolved_chapter_id in _ch_ids:
                        _idx = _ch_ids.index(resolved_chapter_id)
                        if _idx > 0:
                            _prev_id = _ch_ids[_idx - 1]
                            _prev_draft = self._pm.get_accepted_draft(_prev_id)
                            if _prev_draft:
                                _prev_raw = _strip_html(_prev_draft.get("content") or "")
                                prev_chapter_tail = _prev_raw[-2400:].strip()
                                log.debug(
                                    "chat-beats: injecting %d chars from prev chapter %r",
                                    len(prev_chapter_tail), _prev_id,
                                )
                except Exception as exc:
                    log.warning("chat-beats: could not fetch previous chapter: %s", exc)

                system_prompt = _build_writer_system_prompt(
                    lang_display=lang_display,
                    persona=persona,
                    editor_note=editor_note,
                    rag_context=rag_context,
                    image_context=image_context,
                    qa_note=qa_note,
                    style_chunks=style_chunks,
                    beats=ch_beats,
                    existing_text="",          # writing fresh from beats
                    mode="beats",
                    style_mode="inspired_by",
                    prev_chapter_tail=prev_chapter_tail,
                )
                beat_fence = (
                    "⚠ BEATS CONSTRAINT: Write ONLY the scenes and events described in the "
                    "Chapter Blueprint above. Do NOT invent new scenes, characters, or plot "
                    "points that are not in the beats. Start at Beat 1. Stop after the final beat."
                )
                user_message = f"{question}\n\n{beat_fence}"

            else:
                # ── Path 3: General collaborative chat ────────────────────
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
                user_message = question

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
                "Format your response as a structured report "
                "(translate section labels into the response language):\n"
                "- ✅ PASS / 通过 / 合格: brief list of items that are clearly correct\n"
                "- ⚠️ ISSUES / 问题 / 問題点: quote exact location + explain problem + suggest fix\n"
                "- 📋 SUMMARY / 总结 / まとめ: overall verdict and top 3 priorities to fix\n"
                "Write ALL section labels and content in the same language the user writes in."
            )

            system_prompt = (
                # Language instruction FIRST — small models must see this before the English background
                f"{lang_line}\n\n"
                f"{background}\n\n"
                f"Check all of the following:\n{_bullets(check_areas)}\n"
                f"(When referencing these check areas in your response, translate their names into {lang_display}.)\n\n"
                f"{report_format}"
            )
            if not chapter_text:
                system_prompt += (
                    f"\n\n⚠️ No chapter content is loaded. "
                    f"Tell the user IN {lang_display}: "
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

        # user_message defaults to `question` at the top of this block.
        # The writer role may replace it with a structured rewrite message
        # that includes the full chapter text as the explicit target.
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

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


def _style_mode_directive(style_mode: str, style_chunks: List[str], operation_mode: str) -> str:
    """Return a clear, behaviorally-specific directive for the given style_mode.

    style_mode      : one of inspired_by | imitate_closely | preserve_tone_rhythm | avoid_exact_phrasing
    style_chunks    : uploaded style-sample chunks (may be empty)
    operation_mode  : "continue" | "beats" | "rewrite" | "chat"
    """
    has_samples = bool(style_chunks)
    is_rewrite = operation_mode == "rewrite"

    if style_mode == "imitate_closely":
        if has_samples:
            return (
                "Closely imitate the voice in the Style Reference Samples (Priority 6). "
                "Replicate their sentence length variation, paragraph rhythm, dialogue formatting, "
                "punctuation density, and characteristic vocabulary precisely. "
                "The prose should sound as if it were written by the sample author."
            )
        else:
            return (
                "Closely imitate the author's established voice as already present in this chapter. "
                "Replicate the existing sentence length variation, paragraph rhythm, and characteristic "
                "vocabulary — keep the style distinctively consistent with what came before."
            )

    elif style_mode == "preserve_tone_rhythm":
        return (
            "Preserve the established tone, rhythm, and voice of the existing prose exactly. "
            "Keep the same narrative distance, sentence-length patterns, and vocabulary register. "
            "Refine word choice and structure where needed — but do not shift toward any external style, "
            "even if Style Reference Samples are provided."
        )

    elif style_mode == "avoid_exact_phrasing":
        if is_rewrite:
            suffix = (
                "Draw loosely on the Style Reference Samples (Priority 6) for vocabulary and imagery."
                if has_samples else
                "Preserve the narrative content and emotional beats while refreshing the language entirely."
            )
            return (
                "Produce substantially different prose — restructure sentences, vary word choices, and "
                "refresh phrasing throughout. Do not reuse any exact phrases or sentence patterns from "
                f"the original text. {suffix}"
            )
        else:
            return (
                "Vary sentence structures and word choices throughout — avoid predictable patterns and "
                "flat phrasing. Every sentence should feel fresh and precisely chosen, with no repeated "
                "constructions or filler language."
            )

    else:  # inspired_by (default)
        if has_samples:
            return (
                "Draw inspiration from the Style Reference Samples (Priority 6) — let their sentence "
                "rhythms, vocabulary, and prose characteristics inform your writing, while adapting "
                "naturally to fit this story's existing voice and context."
            )
        else:
            return (
                "Write in a style inspired by the author's established voice. "
                "Maintain the narrative tone, prose rhythm, and sentence structure that fits naturally "
                "within the project."
            )


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
    mode: str,          # "continue" | "beats" | "rewrite" | "chat"
    text_after_cursor: str = "",
    style_mode: str = "",
    constraints: Optional[List[str]] = None,
    prev_chapter_tail: str = "",
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

    # Beats-mode quality rules — layered on top of base rules.
    # Replace the mechanical checklist mindset with literary craft guidance.
    beats_quality_rules: List[str] = []
    if mode == "beats":
        beats_quality_rules = [
            "Treat each beat as a dramatic SCENE to write — not a sentence to paraphrase or summarise",
            "SHOW, don't tell — render action, emotion, and revelation through specific concrete detail",
            "Write with sensory richness: what characters see, hear, feel, smell, think, and want",
            "Give characters interiority — their inner reactions make beats feel alive, not mechanical",
            "Vary your pacing deliberately — build tension through action, then let it breathe in reflection or dialogue",
            "Let transitions between beats flow naturally; avoid abrupt 'next, this happened' jumps",
            "Strong verbs, precise nouns — avoid vague filler words and weak verb+adverb combinations",
            "Dialogue should reveal character and advance the scene, not just deliver information",
            "Scene-setting should be selective and purposeful — ground the reader without slowing momentum",
        ]

    # Mode-specific task description
    if mode == "beats":
        if prev_chapter_tail:
            task_line = (
                "Write a complete chapter that continues naturally from the previous chapter. "
                "Your Chapter Blueprint (beats) defines what happens — follow it exactly and completely. "
                "Open your chapter by flowing smoothly from where the previous chapter ended. "
                "Do NOT start a new scene, introduce a new location, or jump in time unless a beat explicitly requires it. "
                "Each beat is a dramatic milestone — bring it fully to life as a scene: vivid setting, "
                "character interiority, concrete sensory detail, rising tension, and dynamic prose. "
                "Move through ALL beats in order. Write like a skilled novelist, not like someone filling in a form."
            )
        else:
            task_line = (
                "Write a complete, compelling chapter from scratch, using the Chapter Blueprint below as your structural backbone. "
                "Each beat is a dramatic milestone — not a script to recite word-for-word. "
                "Bring every beat fully to life as a scene: vivid setting, character interiority, "
                "concrete sensory detail, rising tension, and dynamic prose. "
                "Move through all beats in order, giving each one the space and depth it deserves. "
                "Write like a skilled novelist, not like someone filling in a form."
            )
    elif mode == "continue":
        if text_after_cursor:
            task_line = (
                "Write new content to INSERT at the cursor position (marked ✍ in the draft below). "
                "Your text must flow naturally FROM the content above AND lead smoothly INTO the text that follows."
            )
        else:
            task_line = "Continue the story directly from where the current draft ends."
    elif mode == "rewrite":
        _style_directive = _style_mode_directive(style_mode, style_chunks, "rewrite")
        task_line = (
            f"Rewrite the passage provided by the author. "
            f"{_style_directive} "
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
            "Write only what is asked — new scenes, dialogue, descriptions, revisions, or other content as directed. "
            # Multilingual rewrite intent — critical for non-English users
            "If the user asks you to REWRITE or IMPROVE any content "
            "(重写 / 改写 / 重新写 / rewrite / improve / réécrire / umschreiben / riscrivere): "
            "produce COMPLETELY NEW, SUBSTANTIALLY DIFFERENT prose from scratch. "
            "Do NOT reproduce the original wording — treat the original as a plot summary only, then write fresh. "
            "Improve sentence structure, word choice, rhythm, and imagery. "
            "Apply every pinned Editor Suggestion, fix every pinned QA Issue, "
            "and keep all rewritten content strictly consistent with the memory documents "
            "(codex, beats, research, notes) in the rewritten output."
        )

    # Rewrite mode and chat mode: append critical differentiator rules.
    # Chat mode needs them because users can type "rewrite this chapter" in any language.
    rewrite_rules: List[str] = []
    if mode == "beats":
        # beats_quality_rules already set above — merged below
        pass
    if mode in ("rewrite", "chat"):
        rewrite_rules = [
            # Bilingual so small models catch it regardless of conversation language
            "⚠ REWRITE RULE / 改写规则: If rewriting (重写/改写/rewrite), your output MUST be completely and noticeably different from the original",
            "Do NOT copy sentences verbatim — 不得逐字复制原文",
            "Do NOT produce a superficial paraphrase — genuinely write new prose from scratch",
            "You may restructure paragraphs, change sentence order, alter imagery, or vary rhythm freely",
            "The STORY EVENTS and CHARACTER ACTIONS must remain the same — only the prose changes / 情节事件不变，只改变文字表达",
            "Apply ALL pinned Editor Suggestions and fix ALL pinned QA Issues in the rewritten text",
            "Keep all rewritten content consistent with memory documents — character names, traits, world rules, and plot facts from the Codex and Beats must not be altered or contradicted",
        ]

    if mode == "beats":
        all_rules = rules + beats_quality_rules
    elif rewrite_rules:
        all_rules = rules + rewrite_rules
    else:
        all_rules = rules

    prompt = (
        # Language instruction FIRST — before the English background so small models
        # don't default to English when the user writes in another language.
        f"ALWAYS write in {lang_display}. "
        f"Every word of your output — prose, labels, commentary — must be in {lang_display}. "
        f"Do NOT output English if {lang_display} is not English.\n\n"
        f"{background}\n\n"
        f"Task: {task_line}\n\n"
        "You MUST honour the following (in strict priority order):\n"
        f"1. {p1}\n"
        f"2. {p2}\n"
        f"3. {p3}\n"
        f"4. {p4}\n"
        f"5. {p5}\n"
        f"6. {p6}\n\n"
        f"When writing:\n{_bullets(all_rules)}"
    )

    # ── BEATS MODE: beats appear FIRST — they are the structural directive. ──
    # Memory/codex is demoted to "supporting detail" and must not introduce
    # new events.  Small models follow the most prominent recent block, so
    # placing beats before the codex prevents the codex from overriding them.
    if beats and mode == "beats":
        prompt += (
            f"\n\n╔══════════════════════════════════════════╗\n"
            f"║  YOUR WRITING DIRECTIVES — CHAPTER BEATS  ║\n"
            f"╚══════════════════════════════════════════╝\n"
            f"The beats below define the COMPLETE and ONLY structure of this chapter.\n"
            f"► Start writing at Beat 1. Stop writing after the final beat.\n"
            f"► Do NOT write scenes, events, or plot points that are not listed here.\n"
            f"► Do NOT continue past the last beat.\n"
            f"► Do NOT use codex/world-building content to invent new events — "
            f"beats are your sole story guide.\n"
            f"Each beat is a dramatic moment to bring fully to life — "
            f"then move directly to the next beat.\n\n"
            f"{beats}"
        )

        # If a previous chapter exists, anchor the opening here —
        # immediately after the beats so the model sees them together.
        if prev_chapter_tail:
            prompt += (
                f"\n\n╔══════════════════════════════════════════╗\n"
                f"║  WHERE THE STORY LEFT OFF (prev. chapter)  ║\n"
                f"╚══════════════════════════════════════════╝\n"
                f"Your chapter must open by continuing naturally from this passage.\n"
                f"Do NOT start a new scene, new location, or jump in time unless a beat explicitly says so.\n\n"
                f"...\n"
                f"{prev_chapter_tail}\n"
                f"[End of previous chapter]"
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
    # In beats mode this is supporting context only — label it accordingly so the
    # model does not treat it as a source of new plot events.
    combined_memory = rag_context
    if image_context:
        sep = "\n\n---\n\n" if combined_memory else ""
        combined_memory += f"{sep}### Visual Reference Descriptions\n{image_context}"
    if combined_memory:
        if mode == "beats":
            prompt += (
                f"\n\n## SUPPORTING CONTEXT — Character & World Details\n"
                f"Use the following ONLY to fill in consistent character names, physical descriptions, "
                f"world details, and established facts. "
                f"Do NOT derive new plot events, scenes, or sub-plots from this context — "
                f"the beats listed above are your sole structural guide.\n\n"
                f"{combined_memory}"
            )
        else:
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

    # Chapter beats — for non-beats modes, append here as before.
    # For beats mode the beats were already placed at the top of the context.
    if beats and mode != "beats":
        prompt += (
            f"\n\n## Chapter Beats — FOLLOW THESE EXACTLY\n"
            f"Cover each beat in order. Do NOT skip any. Do NOT add unlisted beats.\n\n"
            f"{beats}"
        )

    # Existing draft / chapter text
    if existing_text:
        if mode == "continue":
            label = "Current Draft (for context — do NOT repeat this; write NEW content only)"
            prompt += f"\n\n## {label}\n{existing_text}"
        elif mode == "chat":
            prompt += (
                f"\n\n## Current Chapter Draft\n"
                f"(EXISTING text — do NOT reproduce it verbatim. "
                f"If the user asks to rewrite, produce completely new prose covering the same events.)\n"
                f"\n{existing_text}"
            )
        else:
            label = "Current Chapter Content (existing draft)"
            prompt += f"\n\n## {label}\n{existing_text}"

    # Style guidance + constraints (shared by continue and beats modes)
    if mode in ("continue", "beats"):
        prompt += f"\n\nStyle guidance: {_style_mode_directive(style_mode, style_chunks, mode)}"
        if constraints:
            prompt += "\n\n## Additional Constraints\n" + _bullets(constraints)

    # ── BEATS MODE: repeat the directive at the very end ──────────────────
    # Small models have recency bias — the last few hundred tokens of the
    # system prompt strongly influence generation.  By restating the beats
    # constraint here (after all context blocks) we prevent the codex or
    # style content from overwriting the model's working directive.
    if mode == "beats" and beats:
        prompt += (
            f"\n\n╔══════════════════════════════════════════╗\n"
            f"║  ⚠  FINAL INSTRUCTION — BEATS ARE YOUR LAW  ⚠  ║\n"
            f"╚══════════════════════════════════════════╝\n"
            f"You have read the Chapter Blueprint and all supporting context.\n"
            f"Now write prose ONLY for what the beats describe — nothing more.\n"
            f"► Beat 1 is your starting point.\n"
            f"► The final beat is your stopping point.\n"
            f"► Every scene, event, and revelation must come from the beats list.\n"
            f"► Supporting context (codex, world details) informs HOW you write, not WHAT happens.\n"
            f"► Output pure prose only — no beat labels, no numbers, no headings."
        )

    return prompt


def _last_paragraphs(text: str, n_tokens: int = 512) -> str:
    """Return the last ~n_tokens worth of text for use as RAG query."""
    max_chars = n_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _cap_rag(results: list, budget_tokens: int, max_distance: float = 1.0) -> list:
    """
    Select as many RAG result chunks as fit within *budget_tokens* total,
    preserving relevance order (results are already sorted best-first by
    ChromaDB cosine distance — lower distance = more similar).

    Parameters
    ----------
    results       : list of dicts with keys 'text' and 'distance'
    budget_tokens : maximum total tokens to include (1 token ≈ 4 chars)
    max_distance  : cosine distance ceiling — chunks with distance ABOVE
                    this value are skipped entirely.  Use < 1.0 to exclude
                    low-relevance chunks.  Default 1.0 = no distance filter.
                    Because ChromaDB returns results sorted best-first,
                    once a chunk exceeds the threshold all subsequent ones
                    will too, so we break early for efficiency.

    Note on budget overflow: when a chunk is individually larger than the
    remaining budget we *skip* it (continue) rather than stopping — a later
    smaller chunk may still fit.  The max_distance check uses break because
    the list is sorted ascending; all subsequent chunks are equally or more
    distant.
    """
    selected: list = []
    used = 0
    for r in results:
        if r.get("distance", 0.0) > max_distance:
            # List is sorted ascending by distance — every remaining chunk is
            # equally or more distant.  No point continuing.
            break
        tokens = max(1, len(r["text"]) // 4)
        if used + tokens > budget_tokens:
            # This chunk alone is too large for the remaining space, but a
            # subsequent smaller chunk might still fit — skip, don't break.
            continue
        selected.append(r)
        used += tokens
    return selected


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


def _detect_language_override(question: str) -> Optional[str]:
    """
    Detect when the user explicitly requests a specific response language.

    Examples (all return the matching ISO code):
      "respond in Japanese"    → "ja"
      "用日语回答"               → "ja"
      "répondez en français"   → "fr"
      "write in Korean"        → "ko"
      "antworte auf Deutsch"   → "de"

    Returns None if no explicit override is detected.
    """
    import re

    # Map of keywords to ISO codes — keywords may appear in any language
    OVERRIDE_PATTERNS: list = [
        # Japanese
        (r"(?:respond|reply|write|answer|output|日语|日文|japanese|japanisch|japonais|japonés)\s*(?:in\s+)?(?:japanese|日语|日文|日本語|にほんご)", "ja"),
        # NOTE: bare 用/以 removed — they are common Chinese characters and cause false positives
        (r"(?:用日语|用日文|日本語で|일본어로)", "ja"),
        # Chinese Simplified
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:chinese simplified|simplified chinese|简体中文|简体)", "zh"),
        (r"(?:用|以)\s*(?:简体中文|中文简体|中文)", "zh"),
        # Chinese Traditional
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:chinese traditional|traditional chinese|繁体中文|繁體中文|繁體)", "zh-tw"),
        (r"(?:用|以)\s*(?:繁體中文|繁體|繁体)", "zh-tw"),
        # Korean
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:korean|한국어|한글|koreanisch|coréen|coreano)", "ko"),
        # NOTE: bare 用/以 removed — they are common Chinese characters and cause false positives
        (r"(?:用韩语|用韓語|한국어로|韓国語で)", "ko"),
        # French
        (r"(?:respond|reply|write|answer|output|répondez|réponds|écris)?\s*(?:in\s+|en\s+)?(?:french|français|franzöisch|francés|francese)", "fr"),
        # German
        (r"(?:respond|reply|write|answer|output|antworte|antwortet|schreibe)?\s*(?:in\s+|auf\s+)?(?:german|deutsch|allemand|alemán|tedesco)", "de"),
        # Spanish
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+|en\s+)?(?:spanish|español|espagnol|spagnolo|spanisch)", "es"),
        # Italian
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:italian|italiano|italien|italienisch|italiani)", "it"),
        # Portuguese
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:portuguese|português|portugais|portugiesisch|portoghese)", "pt"),
        # Russian
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:russian|русский|russe|russisch|ruso)", "ru"),
        # Arabic
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:arabic|عربي|عربية|arabe|arabisch)", "ar"),
        # Hindi
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:hindi|हिन्दी|hindou|hindi)", "hi"),
        # Dutch
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:dutch|nederlands|néerlandais|niederländisch|olandese)", "nl"),
        # Turkish
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:turkish|türkçe|turc|türkisch|turco)", "tr"),
        # Polish
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:polish|polski|polonais|polnisch|polacco)", "pl"),
        # Vietnamese
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:vietnamese|tiếng việt|vietnamien|vietnamesisch)", "vi"),
        # Thai
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:thai|ภาษาไทย|thaïlandais|thailändisch)", "th"),
        # English (explicit switch back)
        (r"(?:respond|reply|write|answer|output)?\s*(?:in\s+)?(?:english|anglais|inglés|inglese|englisch)", "en"),
    ]

    q = question.lower()
    for pattern, code in OVERRIDE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE | re.UNICODE):
            return code

    return None

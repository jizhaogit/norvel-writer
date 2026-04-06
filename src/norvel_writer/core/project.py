"""ProjectManager: facade for all project-level operations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)


def _build_where(
    doc_types: Optional[List[str]] = None,
    chapter_id: Optional[str] = None,
    scope: str = "all",
) -> Optional[Dict]:
    """Build a ChromaDB where filter combining doc_type and chapter scope.

    scope="chapter" → only chunks whose chapter_id matches the given chapter_id
    scope="project" → only chunks with no chapter_id (project-level docs, stored as "")
    scope="all"     → no chapter_id constraint (existing behaviour)
    """
    conditions: list = []
    if doc_types:
        conditions.append(
            {"doc_type": doc_types[0]} if len(doc_types) == 1
            else {"doc_type": {"$in": doc_types}}
        )
    if scope == "chapter" and chapter_id:
        conditions.append({"chapter_id": chapter_id})
    elif scope == "project":
        conditions.append({"chapter_id": ""})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class ProjectManager:
    """
    Central facade. The UI should talk exclusively through this class
    rather than touching storage repos directly.
    """

    def __init__(self, db=None, vector_store=None) -> None:
        from norvel_writer.storage.db import get_db
        from norvel_writer.storage.vector_store import get_vector_store
        self._db = db or get_db()
        self._vs = vector_store or get_vector_store()

        from norvel_writer.storage.repositories.project_repo import ProjectRepo
        from norvel_writer.storage.repositories.document_repo import DocumentRepo
        from norvel_writer.storage.repositories.draft_repo import DraftRepo
        from norvel_writer.storage.repositories.style_repo import StyleRepo

        self._projects = ProjectRepo(self._db)
        self._documents = DocumentRepo(self._db)
        self._drafts = DraftRepo(self._db)
        self._styles = StyleRepo(self._db)

    # ── Projects ──────────────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        description: str = "",
        language: str = "en",
    ) -> str:
        pid = self._projects.create_project(name, description, language)
        # Ensure per-project export directory
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        (cfg.projects_path / pid / "exports").mkdir(parents=True, exist_ok=True)
        (cfg.projects_path / pid / "external_edits").mkdir(parents=True, exist_ok=True)
        log.info("Created project %r (%s)", name, pid)
        return pid

    def get_project(self, project_id: str) -> Optional[Dict]:
        return self._projects.get_project(project_id)

    def list_projects(self) -> List[Dict]:
        return self._projects.list_projects()

    def update_project(self, project_id: str, **kwargs) -> None:
        self._projects.update_project(project_id, **kwargs)

    def delete_project(self, project_id: str) -> None:
        self._projects.delete_project(project_id)
        self._vs.delete_collection(f"project_{project_id}")
        self._vs.delete_collection(f"style_{project_id}")

    # ── Chapters ──────────────────────────────────────────────────────────

    def create_chapter(self, project_id: str, title: str) -> str:
        cid = self._projects.create_chapter(project_id, title)
        # Create a per-chapter files directory so chapter-specific documents
        # can be stored separately from project-level documents.
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        (cfg.projects_path / project_id / "chapters" / cid / "files").mkdir(
            parents=True, exist_ok=True
        )
        log.info("Created chapter %r (%s) with chapter files dir", title, cid)
        return cid

    def get_chapter(self, chapter_id: str) -> Optional[Dict]:
        return self._projects.get_chapter(chapter_id)

    def list_chapters(self, project_id: str) -> List[Dict]:
        return self._projects.list_chapters(project_id)

    def update_chapter(self, chapter_id: str, **kwargs) -> None:
        self._projects.update_chapter(chapter_id, **kwargs)

    def delete_chapter(self, chapter_id: str) -> None:
        # Look up chapter first so we have project_id for ChromaDB cleanup.
        chapter = self._projects.get_chapter(chapter_id)
        if chapter:
            project_id = chapter["project_id"]

            # 1. Collect document IDs before deletion (needed for ChromaDB cleanup).
            doc_ids = self._documents.get_document_ids_by_chapter(chapter_id)

            # 2. Remove each document's embeddings from the project ChromaDB collection.
            for doc_id in doc_ids:
                self._vs.delete_by_document(f"project_{project_id}", doc_id)
                self._vs.delete_by_document(f"style_{project_id}", doc_id)

            # 3. Bulk-delete document + chunk rows from SQLite (chunks cascade via FK).
            self._documents.delete_documents_by_chapter(chapter_id)

            # 4. Remove the chapter files directory from disk.
            try:
                from norvel_writer.config.settings import get_config
                import shutil
                ch_dir = get_config().projects_path / project_id / "chapters" / chapter_id
                if ch_dir.exists():
                    shutil.rmtree(ch_dir, ignore_errors=True)
            except Exception as exc:
                log.warning("delete_chapter: could not remove files dir for %s: %s", chapter_id, exc)

        # 5. Delete the chapter record (SQLite CASCADE removes drafts, versions, images).
        self._projects.delete_chapter(chapter_id)

    def list_chapter_documents(
        self, project_id: str, chapter_id: str, doc_type: Optional[str] = None
    ) -> List[Dict]:
        """List documents that belong to a specific chapter."""
        return self._documents.list_chapter_documents(project_id, chapter_id, doc_type)

    def ensure_all_chapter_folders(self) -> int:
        """Create chapter files directories for all existing chapters (idempotent).

        Returns the number of directories created/verified.
        """
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        count = 0
        for proj in self._projects.list_projects():
            pid = proj["id"]
            for ch in self._projects.list_chapters(pid):
                folder = cfg.projects_path / pid / "chapters" / ch["id"] / "files"
                folder.mkdir(parents=True, exist_ok=True)
                count += 1
        log.info("ensure_all_chapter_folders: verified %d chapter folder(s)", count)
        return count

    # ── Documents ─────────────────────────────────────────────────────────

    def list_documents(
        self, project_id: str, doc_type: Optional[str] = None
    ) -> List[Dict]:
        return self._documents.list_documents(project_id, doc_type)

    def delete_document(self, doc_id: str, project_id: str) -> None:
        doc = self._documents.get_document(doc_id)
        if doc:
            self._vs.delete_by_document(f"project_{project_id}", doc_id)
            self._vs.delete_by_document(f"style_{project_id}", doc_id)
        self._documents.delete_document(doc_id)

    # ── Drafts ────────────────────────────────────────────────────────────

    def save_draft(
        self,
        chapter_id: str,
        content: str,
        model_used: str,
        prompt_used: Optional[str] = None,
    ) -> str:
        return self._drafts.create_draft(chapter_id, content, model_used, prompt_used)

    def accept_draft(self, draft_id: str) -> None:
        self._drafts.accept_draft(draft_id)

    def get_accepted_draft(self, chapter_id: str) -> Optional[Dict]:
        return self._drafts.get_accepted_draft(chapter_id)

    def list_drafts(self, chapter_id: str) -> List[Dict]:
        return self._drafts.list_drafts(chapter_id)

    # ── Style Profiles ────────────────────────────────────────────────────

    def get_active_style_profile(self, project_id: str) -> Optional[Dict]:
        return self._styles.get_active_profile(project_id)

    def set_active_style_profile(self, project_id: str, profile_id: str) -> None:
        self._projects.update_project(project_id, style_profile_id=profile_id)

    def list_style_profiles(self, project_id: str) -> List[Dict]:
        return self._styles.list_style_profiles(project_id)

    # ── RAG Retrieval ─────────────────────────────────────────────────────

    def get_full_context_text(
        self,
        project_id: str,
        chapter_id: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        budget_tokens: Optional[int] = None,
    ) -> str:
        """Reconstruct full document text from SQLite chunks for full-doc context mode.

        Chunks are reassembled in ingestion / position order.  Each document is
        labelled with a header so the LLM understands the source type and title.
        The combined text is hard-truncated to *budget_tokens* (1 token ≈ 4 chars)
        when a budget is supplied — this is a safety net for smaller context
        windows; with a 128 K model the budget will rarely be hit.

        Returns an empty string when no chunks exist (e.g. first run, no uploads).
        """
        from collections import OrderedDict as _OD

        chunk_rows = self._documents.get_all_document_chunks(
            project_id, chapter_id, doc_types
        )
        if not chunk_rows:
            return ""

        # Reassemble chunks per document, preserving ingestion order
        docs: Dict[str, dict] = _OD()
        for row in chunk_rows:
            did = row["doc_id"]
            if did not in docs:
                docs[did] = {
                    "title": row["title"] or "Untitled",
                    "doc_type": row["doc_type"],
                    "texts": [],
                }
            docs[did]["texts"].append(row["text"])

        sections: List[str] = []
        used_chars = 0
        budget_chars = budget_tokens * 4 if budget_tokens else None

        for doc in docs.values():
            header = f"=== [{doc['doc_type']}] {doc['title']} ==="
            body = "\n".join(doc["texts"])
            section = f"{header}\n{body}"

            if budget_chars is not None:
                section_chars = len(section)
                if used_chars + section_chars > budget_chars:
                    remaining = budget_chars - used_chars
                    if remaining > 400:          # include truncated tail only if meaningful
                        sections.append(section[:remaining])
                    break
                used_chars += section_chars

            sections.append(section)

        return "\n\n---\n\n".join(sections)

    async def retrieve_context(
        self,
        project_id: str,
        query: str,
        n_results: int = 8,
        doc_types: Optional[List[str]] = None,
        chapter_id: Optional[str] = None,
        scope: str = "all",
    ) -> List[Dict]:
        """Retrieve relevant chunks for drafting context.

        scope="chapter" — only chunks belonging to chapter_id (chapter memory).
        scope="project" — only project-level chunks (no chapter_id).
        scope="all"     — all chunks regardless of chapter (legacy behaviour).
        """
        from norvel_writer.llm.embedder import EmbeddingService

        embedder = EmbeddingService()
        query_emb = await embedder.embed_single(query)
        if not query_emb:
            return []

        where = _build_where(doc_types, chapter_id, scope)

        results = self._vs.query(
            collection_name=f"project_{project_id}",
            query_embedding=query_emb,
            n_results=n_results,
            where=where,
        )
        return [
            {
                "id": r.id,
                "text": r.text,
                "distance": r.distance,
                "metadata": r.metadata,
            }
            for r in results
        ]

    async def retrieve_style_examples(
        self,
        project_id: str,
        query: str,
        n_results: int = 4,
    ) -> List[Dict]:
        """Retrieve style-similar passages from the style collection."""
        from norvel_writer.llm.embedder import EmbeddingService

        embedder = EmbeddingService()
        query_emb = await embedder.embed_single(query)
        if not query_emb:
            return []

        results = self._vs.query(
            collection_name=f"style_{project_id}",
            query_embedding=query_emb,
            n_results=n_results,
        )
        return [
            {"id": r.id, "text": r.text, "distance": r.distance}
            for r in results
        ]

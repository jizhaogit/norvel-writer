"""ProjectManager: facade for all project-level operations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)


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
        return self._projects.create_chapter(project_id, title)

    def get_chapter(self, chapter_id: str) -> Optional[Dict]:
        return self._projects.get_chapter(chapter_id)

    def list_chapters(self, project_id: str) -> List[Dict]:
        return self._projects.list_chapters(project_id)

    def update_chapter(self, chapter_id: str, **kwargs) -> None:
        self._projects.update_chapter(chapter_id, **kwargs)

    def delete_chapter(self, chapter_id: str) -> None:
        self._projects.delete_chapter(chapter_id)

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

    async def retrieve_context(
        self,
        project_id: str,
        query: str,
        n_results: int = 8,
        doc_types: Optional[List[str]] = None,
        chapter_id: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve relevant chunks for drafting context."""
        from norvel_writer.llm.embedder import EmbeddingService
        from norvel_writer.storage.vector_store import QueryResult

        embedder = EmbeddingService()
        query_emb = await embedder.embed_single(query)
        if not query_emb:
            return []

        where: Optional[Dict] = None
        if doc_types:
            if len(doc_types) == 1:
                where = {"doc_type": doc_types[0]}
            else:
                where = {"doc_type": {"$in": doc_types}}

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

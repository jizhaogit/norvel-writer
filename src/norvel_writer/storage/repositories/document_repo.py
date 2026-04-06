"""CRUD for documents and chunks."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from norvel_writer.storage.db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create_document(
        self,
        project_id: str,
        file_path: str,
        file_hash: str,
        doc_type: str,
        fmt: str,
        title: Optional[str] = None,
        language: Optional[str] = None,
        chapter_id: Optional[str] = None,
    ) -> str:
        did = str(uuid.uuid4())
        now = _now()
        with self._db.connect() as conn:
            conn.execute(
                """INSERT INTO documents
                   (id, project_id, chapter_id, file_path, file_hash, doc_type, format,
                    title, language, ingested_at, status)
                   VALUES(?,?,?,?,?,?,?,?,?,?,'pending')""",
                (did, project_id, chapter_id, file_path, file_hash,
                 doc_type, fmt, title, language, now),
            )
        return did

    def get_document(self, doc_id: str) -> Optional[dict]:
        row = self._db.execute_one("SELECT * FROM documents WHERE id=?", (doc_id,))
        return dict(row) if row else None

    def list_documents(
        self,
        project_id: str,
        doc_type: Optional[str] = None,
    ) -> List[dict]:
        """List project-level documents (chapter_id IS NULL or empty).

        Chapter-scoped documents are excluded — use list_chapter_documents()
        to retrieve those.
        """
        if doc_type:
            rows = self._db.execute(
                "SELECT * FROM documents WHERE project_id=? AND doc_type=? "
                "AND (chapter_id IS NULL OR chapter_id='') ORDER BY ingested_at",
                (project_id, doc_type),
            )
        else:
            rows = self._db.execute(
                "SELECT * FROM documents WHERE project_id=? "
                "AND (chapter_id IS NULL OR chapter_id='') ORDER BY ingested_at",
                (project_id,),
            )
        return [dict(r) for r in rows]

    def list_chapter_documents(
        self,
        project_id: str,
        chapter_id: str,
        doc_type: Optional[str] = None,
    ) -> List[dict]:
        """List documents scoped to a specific chapter."""
        if doc_type:
            rows = self._db.execute(
                "SELECT * FROM documents WHERE project_id=? AND chapter_id=? AND doc_type=? ORDER BY ingested_at",
                (project_id, chapter_id, doc_type),
            )
        else:
            rows = self._db.execute(
                "SELECT * FROM documents WHERE project_id=? AND chapter_id=? ORDER BY ingested_at",
                (project_id, chapter_id),
            )
        return [dict(r) for r in rows]

    def update_document_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: Optional[int] = None,
    ) -> None:
        if chunk_count is not None:
            with self._db.connect() as conn:
                conn.execute(
                    "UPDATE documents SET status=?, chunk_count=? WHERE id=?",
                    (status, chunk_count, doc_id),
                )
        else:
            with self._db.connect() as conn:
                conn.execute(
                    "UPDATE documents SET status=? WHERE id=?", (status, doc_id)
                )

    def update_document_title(self, doc_id: str, title: str) -> None:
        with self._db.connect() as conn:
            conn.execute("UPDATE documents SET title=? WHERE id=?", (title, doc_id))

    def delete_document(self, doc_id: str) -> None:
        with self._db.connect() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def get_document_ids_by_chapter(self, chapter_id: str) -> List[str]:
        """Return all document IDs scoped to a chapter (for pre-delete ChromaDB cleanup)."""
        rows = self._db.execute(
            "SELECT id FROM documents WHERE chapter_id=?", (chapter_id,)
        )
        return [row["id"] for row in rows]

    def delete_documents_by_chapter(self, chapter_id: str) -> None:
        """Bulk-delete all documents scoped to a chapter (chunks cascade via FK)."""
        with self._db.connect() as conn:
            conn.execute("DELETE FROM documents WHERE chapter_id=?", (chapter_id,))

    def find_by_hash(
        self,
        project_id: str,
        file_hash: str,
        chapter_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Find an existing document by hash, scoped to the same chapter_id.

        A project-level doc (chapter_id=None/'') and a chapter-level doc
        (chapter_id=uuid) with the same file content are stored separately,
        so deduplication must match on chapter_id as well.
        """
        if chapter_id:
            row = self._db.execute_one(
                "SELECT * FROM documents WHERE project_id=? AND file_hash=? AND chapter_id=?",
                (project_id, file_hash, chapter_id),
            )
        else:
            row = self._db.execute_one(
                "SELECT * FROM documents WHERE project_id=? AND file_hash=? "
                "AND (chapter_id IS NULL OR chapter_id='')",
                (project_id, file_hash),
            )
        return dict(row) if row else None

    # ── Chunks ────────────────────────────────────────────────────────────

    def insert_chunks(
        self,
        document_id: str,
        chunks: List[str],
        token_counts: Optional[List[int]] = None,
    ) -> List[str]:
        ids = []
        with self._db.connect() as conn:
            for i, text in enumerate(chunks):
                cid = str(uuid.uuid4())
                tc = token_counts[i] if token_counts else None
                conn.execute(
                    "INSERT INTO chunks(id, document_id, position, text, token_count) VALUES(?,?,?,?,?)",
                    (cid, document_id, i, text, tc),
                )
                ids.append(cid)
        return ids

    def list_chunks(self, document_id: str) -> List[dict]:
        rows = self._db.execute(
            "SELECT * FROM chunks WHERE document_id=? ORDER BY position",
            (document_id,),
        )
        return [dict(r) for r in rows]

    def delete_chunks(self, document_id: str) -> None:
        with self._db.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))

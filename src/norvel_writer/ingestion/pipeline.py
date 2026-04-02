"""Ingestion pipeline: file → chunks → embeddings → vector store + SQLite."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, List, Optional

from norvel_writer.config.defaults import IMAGE_FORMATS
from norvel_writer.ingestion.base import BaseIngestor
from norvel_writer.ingestion.docx_ingestor import DocxIngestor
from norvel_writer.ingestion.image_ingestor import ImageIngestor
from norvel_writer.ingestion.json_ingestor import JsonIngestor
from norvel_writer.ingestion.md_ingestor import MdIngestor
from norvel_writer.ingestion.pdf_ingestor import PdfIngestor
from norvel_writer.ingestion.txt_ingestor import TxtIngestor
from norvel_writer.utils.chunker import chunk_text
from norvel_writer.utils.text_utils import detect_language, hash_file

log = logging.getLogger(__name__)

_IMAGE_INGESTOR = ImageIngestor()

_INGESTORS: List[BaseIngestor] = [
    TxtIngestor(),
    MdIngestor(),
    DocxIngestor(),
    PdfIngestor(),
    JsonIngestor(),
    _IMAGE_INGESTOR,
]


def _get_ingestor(path: Path) -> Optional[BaseIngestor]:
    for ing in _INGESTORS:
        if ing.can_handle(path):
            return ing
    return None


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_FORMATS


class IngestPipeline:
    """
    Ingests a file into the project's knowledge base.

    For images: uses a vision model to generate a text description first,
    then embeds that description like any other document.

    Progress events: progress_cb(pct: int) where pct is 0-100.
    """

    def __init__(
        self,
        db=None,
        vector_store=None,
        embed_model: Optional[str] = None,
        vision_model: Optional[str] = None,
    ) -> None:
        from norvel_writer.storage.db import get_db
        from norvel_writer.storage.vector_store import get_vector_store
        from norvel_writer.config.settings import get_config
        self._db = db or get_db()
        self._vs = vector_store or get_vector_store()
        self._embed_model = embed_model
        cfg = get_config()
        self._vision_model = vision_model or getattr(cfg, "vision_model", None)

    async def run(
        self,
        file_path: Path,
        project_id: str,
        doc_type: str,
        chapter_id: Optional[str] = None,
        progress_cb: Optional[Callable[[int], None]] = None,
        reindex: bool = False,
        language: str = "English",
    ) -> str:
        """
        Ingest a file. Returns the document_id.

        Raises ValueError if the file format is unsupported.
        Skips re-ingestion if file hash matches existing record (unless reindex=True).
        """
        from norvel_writer.storage.repositories.document_repo import DocumentRepo
        from norvel_writer.llm.embedder import EmbeddingService

        doc_repo = DocumentRepo(self._db)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ingestor = _get_ingestor(file_path)
        if ingestor is None:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        file_hash = hash_file(file_path)

        # Skip if already indexed with same hash
        if not reindex:
            existing = doc_repo.find_by_hash(project_id, file_hash)
            if existing and existing["status"] == "ready":
                log.info("Skipping already-indexed file: %s", file_path.name)
                return existing["id"]

        def _progress(pct: int) -> None:
            if progress_cb:
                progress_cb(pct)

        _progress(5)

        # 1. Extract text — images need async vision processing
        try:
            if _is_image(file_path) and self._vision_model:
                _progress(10)
                doc = await _IMAGE_INGESTOR.ingest_async(
                    path=file_path,
                    vision_model=self._vision_model,
                    doc_type=doc_type,
                    language=language,
                )
                _progress(40)
            else:
                doc = ingestor.ingest(file_path)
                _progress(15)
        except Exception as exc:
            log.error("Ingestor failed for %s: %s", file_path, exc)
            raise

        # 2. Detect language
        lang = doc.language or detect_language(doc.text)

        # 3. Create DB record
        doc_id = doc_repo.create_document(
            project_id=project_id,
            file_path=str(file_path),
            file_hash=file_hash,
            doc_type=doc_type,
            fmt=file_path.suffix.lstrip(".").lower(),
            title=doc.title,
            language=lang,
            chapter_id=chapter_id,
        )
        doc_repo.update_document_status(doc_id, "processing")

        _progress(45)

        # 4. Chunk
        chunks = chunk_text(doc.text, language="english")
        if not chunks:
            log.warning("No chunks produced for %s", file_path.name)
            doc_repo.update_document_status(doc_id, "ready", chunk_count=0)
            return doc_id

        _progress(50)

        # 5. Store chunks in SQLite
        chunk_ids = doc_repo.insert_chunks(doc_id, chunks)

        _progress(55)

        # 6. Embed chunks
        def embed_progress(done: int, total: int) -> None:
            pct = 55 + int((done / total) * 30)
            _progress(pct)

        embedder = EmbeddingService(
            model=self._embed_model,
            progress_cb=embed_progress,
        )
        embeddings = await embedder.embed_texts(chunks)

        _progress(85)

        # 7. Upsert into vector store
        collection_name = f"project_{project_id}"
        metadatas = [
            {
                "document_id": doc_id,
                "doc_type": doc_type,
                "chapter_id": chapter_id or "",
                "position": str(i),
                "language": lang,
                "title": doc.title or "",
                "project_id": project_id,
                "is_image": str(_is_image(file_path)),
            }
            for i in range(len(chunk_ids))
        ]
        self._vs.upsert_chunks(
            collection_name=collection_name,
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        if doc_type == "style_sample":
            self._vs.upsert_chunks(
                collection_name=f"style_{project_id}",
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

        _progress(95)

        doc_repo.update_document_status(doc_id, "ready", chunk_count=len(chunks))

        _progress(100)
        log.info(
            "Ingested %s → %d chunks [project=%s, type=%s]",
            file_path.name, len(chunks), project_id, doc_type,
        )
        return doc_id

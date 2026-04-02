from __future__ import annotations
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument
from norvel_writer.utils.text_utils import normalize_whitespace


class DocxIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".docx"

    def ingest(self, path: Path) -> IngestedDocument:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = normalize_whitespace("\n\n".join(paragraphs))
        # Title: core property or first paragraph
        title = None
        if doc.core_properties.title:
            title = doc.core_properties.title
        elif paragraphs:
            title = paragraphs[0][:100]
        return IngestedDocument(
            text=text,
            title=title or path.stem,
            metadata={"author": doc.core_properties.author or ""},
        )

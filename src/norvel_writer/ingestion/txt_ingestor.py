from __future__ import annotations
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument
from norvel_writer.utils.text_utils import normalize_whitespace, extract_title_from_text


class TxtIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def ingest(self, path: Path) -> IngestedDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = normalize_whitespace(text)
        return IngestedDocument(
            text=text,
            title=extract_title_from_text(text) or path.stem,
        )

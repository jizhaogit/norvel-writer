from __future__ import annotations
import re
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument
from norvel_writer.utils.text_utils import normalize_whitespace


class MdIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".markdown"}

    def ingest(self, path: Path) -> IngestedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Extract title from first H1
        title = None
        m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        if m:
            title = m.group(1).strip()
        # Strip markdown syntax for embedding
        text = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        text = re.sub(r"`[^`]+`", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = normalize_whitespace(text)
        return IngestedDocument(
            text=text,
            title=title or path.stem,
        )

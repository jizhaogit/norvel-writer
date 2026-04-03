from __future__ import annotations
import re
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument


class MdIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".markdown"}

    def ingest(self, path: Path) -> IngestedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")

        # Extract title from first H1 — keep the raw Markdown intact for the AI.
        title: str | None = None
        m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        if m:
            title = m.group(1).strip()

        # The text IS already Markdown — preserve it as-is.
        # Modern embedding models and LLMs both understand Markdown structure
        # (headings, bold, lists, code blocks) and use it as semantic signal.
        return IngestedDocument(
            text=raw.strip(),
            title=title or path.stem,
        )

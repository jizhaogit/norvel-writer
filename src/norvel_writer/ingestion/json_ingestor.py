from __future__ import annotations
import json
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument
from norvel_writer.utils.text_utils import normalize_whitespace


def _flatten(obj, depth: int = 0) -> str:
    """Recursively convert JSON to readable text."""
    if depth > 8:
        return str(obj)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(f"{k}: {_flatten(v, depth + 1)}")
        return "\n".join(parts)
    if isinstance(obj, list):
        return "\n".join(_flatten(item, depth + 1) for item in obj)
    return str(obj)


class JsonIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".json"

    def ingest(self, path: Path) -> IngestedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw)
            text = normalize_whitespace(_flatten(data))
        except json.JSONDecodeError:
            text = normalize_whitespace(raw)
        return IngestedDocument(
            text=text,
            title=path.stem,
        )

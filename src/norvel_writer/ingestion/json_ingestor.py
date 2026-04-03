from __future__ import annotations
import json
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument


def _json_to_markdown(obj, depth: int = 0, max_depth: int = 8) -> str:
    """
    Recursively convert a JSON value to structured Markdown.

    • dict  → each key becomes a heading (level scales with nesting depth),
               its value is rendered below it.
    • list  → each item becomes a bullet point; nested objects are indented.
    • scalar → plain string value.

    Heading levels are capped at 6 (######) to stay valid Markdown.
    """
    if depth > max_depth:
        return str(obj)

    if isinstance(obj, dict):
        parts: list[str] = []
        for key, value in obj.items():
            heading_level = min(depth + 2, 6)   # start at ## so # is free for title
            prefix = "#" * heading_level
            key_str = str(key).replace("_", " ").replace("-", " ").strip()

            if isinstance(value, (dict, list)):
                parts.append(f"{prefix} {key_str}")
                inner = _json_to_markdown(value, depth + 1, max_depth)
                if inner.strip():
                    parts.append(inner)
            else:
                # Inline scalar: keep key as heading, value as paragraph
                val_str = str(value).strip()
                if val_str:
                    parts.append(f"{prefix} {key_str}\n\n{val_str}")
                else:
                    parts.append(f"{prefix} {key_str}")

        return "\n\n".join(parts)

    if isinstance(obj, list):
        items: list[str] = []
        for item in obj:
            if isinstance(item, dict):
                # Render nested dict as a block, indented under a bullet
                inner = _json_to_markdown(item, depth, max_depth)
                # Indent each line of the inner block for visual nesting
                indented = "\n".join(
                    ("  " + ln if ln.strip() else ln) for ln in inner.splitlines()
                )
                items.append(f"-\n{indented}")
            elif isinstance(item, list):
                inner = _json_to_markdown(item, depth, max_depth)
                items.append(f"- {inner}")
            else:
                items.append(f"- {item}")
        return "\n".join(items)

    return str(obj)


class JsonIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".json"

    def ingest(self, path: Path) -> IngestedDocument:
        import re
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Malformed JSON — store as a fenced code block so the AI can still
            # read the raw content without mistaking it for prose.
            md = f"```json\n{raw.strip()}\n```"
            return IngestedDocument(text=md, title=path.stem)

        # Build a top-level # heading from the filename
        title = path.stem.replace("_", " ").replace("-", " ").strip()
        md_body = _json_to_markdown(data)
        md = f"# {title}\n\n{md_body}".strip()

        # Derive a human-readable title: prefer a 'name', 'title', or 'id' key
        extracted_title: str | None = None
        if isinstance(data, dict):
            for key in ("title", "name", "id", "label"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    extracted_title = val.strip()[:100]
                    break

        return IngestedDocument(
            text=md,
            title=extracted_title or title or path.stem,
        )

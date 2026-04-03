from __future__ import annotations
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument


def _collect_text(elem) -> str:
    """Recursively collect all text content from an ODF element."""
    parts: list[str] = []
    for node in elem.childNodes:
        if hasattr(node, "data"):
            parts.append(node.data)
        elif hasattr(node, "childNodes"):
            tag = getattr(node, "qname", ("", ""))[1]
            if tag == "line-break":
                parts.append("\n")
            elif tag == "tab":
                parts.append("  ")   # two spaces — Markdown-safe tab substitute
            else:
                parts.append(_collect_text(node))
    return "".join(parts)


def _heading_level(elem) -> int:
    """
    Extract the outline level from an ODF heading element.
    ODF stores this as the 'outline-level' attribute (integer 1-6).
    Falls back to 1 if the attribute is absent or unparseable.
    """
    # odfpy exposes attributes through getAttribute; the local name varies
    for attr_name in ("outlinelevel", "outline-level"):
        try:
            val = elem.getAttribute(attr_name)
            if val is not None:
                return max(1, min(6, int(val)))
        except (TypeError, ValueError, AttributeError):
            pass

    # Last resort: inspect the style name (e.g. "Heading_20_1" → level 1)
    import re
    try:
        style = elem.getAttribute("stylename") or ""
        m = re.search(r"(\d+)$", style)
        if m:
            return max(1, min(6, int(m.group(1))))
    except Exception:
        pass

    return 1


def _elem_to_markdown(elem) -> str | None:
    """
    Convert a single top-level ODF text element to a Markdown block.

    • `h` (heading)   → `# / ## / ### …` depending on outline level
    • `p` (paragraph) → plain text (inline bold/italic not extracted by odfpy
                         without deeper per-run inspection, so kept as prose)
    Returns None for empty elements so callers can skip them.
    """
    tag = getattr(elem, "qname", ("", ""))[1]
    text = _collect_text(elem).strip()
    if not text:
        return None

    if tag == "h":
        level = _heading_level(elem)
        return "#" * level + " " + text

    # Regular paragraph
    return text


class OdtIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".odt"

    def ingest(self, path: Path) -> IngestedDocument:
        from odf.opendocument import load

        doc = load(str(path))

        blocks: list[str] = []
        first_heading: str | None = None

        for elem in doc.text.childNodes:
            tag = getattr(elem, "qname", ("", ""))[1]
            if tag not in ("p", "h"):
                continue

            md_block = _elem_to_markdown(elem)
            if md_block is None:
                continue

            # Track first heading for use as title
            if tag == "h" and first_heading is None:
                first_heading = _collect_text(elem).strip()[:100]

            blocks.append(md_block)

        md = "\n\n".join(blocks)

        # Title: document meta → first heading → filename
        meta_title: str | None = None
        if doc.meta:
            for node in doc.meta.childNodes:
                if getattr(node, "qname", ("", ""))[1] == "title":
                    raw = _collect_text(node).strip()
                    if raw:
                        meta_title = raw
                        break

        title = meta_title or first_heading or path.stem

        return IngestedDocument(
            text=md.strip(),
            title=title,
            metadata={},
        )

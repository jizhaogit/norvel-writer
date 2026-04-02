from __future__ import annotations
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument
from norvel_writer.utils.text_utils import normalize_whitespace


class OdtIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".odt"

    def ingest(self, path: Path) -> IngestedDocument:
        from odf.opendocument import load
        from odf.text import P, H
        from odf.element import Element

        doc = load(str(path))

        paragraphs: list[str] = []
        title_text: str | None = None

        # Walk all text paragraphs and headings in document order
        for elem in doc.text.childNodes:
            tag = getattr(elem, "qname", ("", ""))[1]
            if tag in ("p", "h"):
                text = _collect_text(elem).strip()
                if not text:
                    continue
                if tag == "h" and title_text is None:
                    title_text = text[:100]
                paragraphs.append(text)

        text = normalize_whitespace("\n\n".join(paragraphs))

        # Try document meta title first
        meta = doc.meta
        meta_title: str | None = None
        if meta:
            for node in meta.childNodes:
                if getattr(node, "qname", ("", ""))[1] == "title":
                    raw = _collect_text(node).strip()
                    if raw:
                        meta_title = raw
                        break

        title = meta_title or title_text or path.stem

        return IngestedDocument(
            text=text,
            title=title,
            metadata={},
        )


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
                parts.append("\t")
            else:
                parts.append(_collect_text(node))
    return "".join(parts)

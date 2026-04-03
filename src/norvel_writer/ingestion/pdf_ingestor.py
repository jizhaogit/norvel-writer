from __future__ import annotations
import re
import statistics
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument


def _looks_like_heading(line: str, median_size: float, line_sizes: dict[str, float]) -> bool:
    """
    Heuristic: a line is a heading candidate when it is short, does not end
    with sentence-terminal punctuation, and its average character font size is
    noticeably larger than the page median (or it is ALL CAPS / Title Case in
    a short line when no size information is available).
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    # Skip lines ending with sentence punctuation — those are prose, not headings
    if stripped[-1] in ".!?,;:":
        return False

    # Size-based detection (populated when char-level data is available)
    if stripped in line_sizes and median_size > 0:
        return line_sizes[stripped] >= median_size * 1.15

    # Fallback: short ALL-CAPS or Title-Case line with no lowercase words
    if len(stripped) <= 80:
        words = stripped.split()
        if all(w[0].isupper() for w in words if w) and len(words) <= 10:
            return True
        if stripped.isupper() and len(stripped) <= 60:
            return True

    return False


def _page_to_markdown(page) -> str:
    """
    Convert a single pdfplumber Page to a Markdown string.

    Strategy:
    1. Attempt character-level extraction to infer heading font sizes.
    2. Fall back to plain text extraction when char data is unavailable.
    3. Detect heading lines and prefix them with ##.
    4. Preserve paragraph breaks (double newlines).
    """
    # ── Try character-level size data ────────────────────────────────────────
    chars = page.chars or []
    line_sizes: dict[str, float] = {}
    median_size = 0.0

    if chars:
        # Group characters by their vertical position (rounded to nearest pt)
        from collections import defaultdict
        line_char_map: dict[int, list] = defaultdict(list)
        for ch in chars:
            y = round(float(ch.get("top", 0)))
            line_char_map[y].append(ch)

        all_sizes: list[float] = []
        for y_chars in line_char_map.values():
            sizes = [float(c.get("size", 0)) for c in y_chars if c.get("size")]
            if sizes:
                line_text = "".join(c.get("text", "") for c in y_chars).strip()
                avg = sum(sizes) / len(sizes)
                line_sizes[line_text] = avg
                all_sizes.extend(sizes)

        if all_sizes:
            median_size = statistics.median(all_sizes)

    # ── Extract text ──────────────────────────────────────────────────────────
    raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    if not raw.strip():
        return ""

    # ── Convert lines to Markdown ─────────────────────────────────────────────
    raw_lines = raw.splitlines()
    md_lines: list[str] = []
    prev_blank = False

    for line in raw_lines:
        stripped = line.strip()

        if not stripped:
            if not prev_blank:
                md_lines.append("")  # single blank line between paragraphs
            prev_blank = True
            continue
        prev_blank = False

        if _looks_like_heading(stripped, median_size, line_sizes):
            md_lines.append(f"## {stripped}")
        else:
            md_lines.append(stripped)

    return "\n".join(md_lines)


class PdfIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def ingest(self, path: Path) -> IngestedDocument:
        import pdfplumber

        page_blocks: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                block = _page_to_markdown(page)
                if block.strip():
                    page_blocks.append(block.strip())

        # Separate pages with a Markdown horizontal rule so the AI knows where
        # one page ends and the next begins (useful for multi-chapter PDFs).
        md = "\n\n---\n\n".join(page_blocks)

        # Title: first heading found, or filename
        title: str | None = None
        m = re.search(r"^##\s+(.+)$", md, re.MULTILINE)
        if m:
            title = m.group(1).strip()

        return IngestedDocument(
            text=md.strip(),
            title=title or path.stem,
            metadata={"page_count": str(len(page_blocks))},
        )

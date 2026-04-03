from __future__ import annotations
import re
from pathlib import Path  # noqa: F401 — used in type hint for `path` parameter
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument


def _txt_to_markdown(text: str) -> str:
    """
    Convert plain text to Markdown with lightweight structure detection.

    Detects and converts:
      • Setext headings  — line underlined with ===  →  # Heading
                           line underlined with ---  →  ## Heading
      • ATX headings     — lines already starting with #  (kept as-is)
      • ALL-CAPS titles  — short uppercase lines        →  ## Title Case
      • Bullet variants  — *, •, ·, ◦                  →  - item
      • Numbered lists   — "1." / "1)"                  →  kept as-is
    Everything else is preserved verbatim.
    """
    lines = text.splitlines()
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Setext-style headings ─────────────────────────────────────────
        if i + 1 < len(lines):
            ul = lines[i + 1].strip()
            if stripped and ul and len(ul) >= 3:
                if all(c == "=" for c in ul):
                    result.append(f"# {stripped}")
                    i += 2
                    continue
                # Guard: don't treat a list item's "---" separator as heading
                if all(c == "-" for c in ul) and not stripped.startswith("-"):
                    result.append(f"## {stripped}")
                    i += 2
                    continue

        # ── ATX headings already present ─────────────────────────────────
        if stripped.startswith("#"):
            result.append(stripped)
            i += 1
            continue

        # ── ALL-CAPS lines that look like section titles ──────────────────
        # Conditions: short, purely alphabetic/space/hyphen, no terminal punct
        if (
            stripped
            and len(stripped) <= 60
            and stripped.isupper()
            and not stripped[-1] in ".!?,;:"
            and re.fullmatch(r"[A-Z][A-Z0-9 '\-–—]+", stripped)
        ):
            result.append(f"## {stripped.title()}")
            i += 1
            continue

        # ── Normalize non-standard bullet characters ──────────────────────
        if re.match(r"^[*•·◦]\s+", stripped):
            result.append("- " + re.sub(r"^[*•·◦]\s+", "", stripped))
            i += 1
            continue

        # ── Numbered lists — keep as-is ───────────────────────────────────
        if re.match(r"^\d+[.)]\s", stripped):
            result.append(stripped)
            i += 1
            continue

        # ── Everything else verbatim ──────────────────────────────────────
        result.append(line)
        i += 1

    return "\n".join(result)


class TxtIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def ingest(self, path: Path) -> IngestedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        md = _txt_to_markdown(raw)

        # Derive title: first heading found, or first non-empty line
        title: str | None = None
        m = re.search(r"^#{1,6}\s+(.+)$", md, re.MULTILINE)
        if m:
            title = m.group(1).strip()
        else:
            for ln in md.splitlines():
                if ln.strip():
                    title = ln.strip()[:100]
                    break

        return IngestedDocument(
            text=md.strip(),
            title=title or path.stem,
        )

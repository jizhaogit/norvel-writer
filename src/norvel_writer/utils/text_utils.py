"""Text utilities: language detection, hashing, normalisation."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional


def hash_file(path: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    """
    Return an ISO 639-1 language code matching the LANGUAGES registry in defaults.py.
    Defaults to 'en' on failure.

    Strategy:
    1. Unicode script range detection — 100% accurate for CJK, Korean, Japanese,
       Arabic, Thai, Hebrew, Hindi. These scripts occupy non-overlapping Unicode
       blocks so no ML is needed. langdetect frequently misidentifies Chinese as
       Korean on short text, so we never let it override these decisions.
    2. langdetect with a deterministic seed — used only for Latin/Cyrillic scripts
       (English, French, German, Spanish, Russian, etc.) where it is reliable.
    """
    if not text or not text.strip():
        return "en"

    sample = text[:500]

    # ── Unicode script range counters ─────────────────────────────────────────
    # Korean Hangul MUST be checked before CJK — the blocks do not overlap
    # but langdetect confuses them on short text.
    hangul     = sum(1 for c in sample if '\uAC00' <= c <= '\uD7AF'
                                       or '\u1100' <= c <= '\u11FF'
                                       or '\u3130' <= c <= '\u318F')
    hiragana   = sum(1 for c in sample if '\u3040' <= c <= '\u309F')
    katakana   = sum(1 for c in sample if '\u30A0' <= c <= '\u30FF')
    cjk        = sum(1 for c in sample if '\u4E00' <= c <= '\u9FFF'
                                       or '\u3400' <= c <= '\u4DBF'
                                       or '\uF900' <= c <= '\uFAFF')
    arabic     = sum(1 for c in sample if '\u0600' <= c <= '\u06FF')
    thai       = sum(1 for c in sample if '\u0E00' <= c <= '\u0E7F')
    hebrew     = sum(1 for c in sample if '\u0590' <= c <= '\u05FF')
    devanagari = sum(1 for c in sample if '\u0900' <= c <= '\u097F')

    # ── Script decisions — order matters ──────────────────────────────────────
    if hangul > 0:
        return "ko"

    if hiragana > 0 or katakana > 0:
        return "ja"

    if cjk > 0:
        # May be Chinese Simplified or Traditional — try langdetect but never
        # accept "ko" if confused, and normalise zh-cn → zh.
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0  # deterministic result
            code = detect(sample)
            if code in ("zh-cn", "zh"):
                return "zh"
            if code == "zh-tw":
                return "zh-tw"
        except Exception:
            pass
        return "zh"  # safe default for any CJK-only text

    if arabic > 0:
        return "ar"
    if thai > 0:
        return "th"
    if devanagari > 0:
        return "hi"
    if hebrew > 0:
        return "he"

    # ── Latin / Cyrillic: use langdetect with deterministic seed ──────────────
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        code = detect(text[:2000])
        # Normalise langdetect variants to our registry keys
        _aliases: dict = {
            "zh-cn": "zh", "zh-tw": "zh-tw",
            "pt-br": "pt", "pt-pt": "pt",
        }
        return _aliases.get(code, code)
    except Exception:
        return "en"


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens tokens."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    # Truncate at word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + "…"


def strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace into plain text."""
    if not html:
        return ""
    if "<" not in html:
        return normalize_whitespace(html)
    # Remove tags
    text = re.sub(r"<(br|p|div|li|h[1-6]|tr|blockquote)[^>]*>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    import html as _html
    text = _html.unescape(text)
    return normalize_whitespace(text)


def extract_title_from_text(text: str) -> Optional[str]:
    """Try to extract a title from the first non-empty line."""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line and len(line) < 200:
            return line
    return None

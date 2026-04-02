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
    """Return ISO 639-1 language code, defaulting to 'en' on failure."""
    sample = text[:2000]
    try:
        from langdetect import detect, LangDetectException
        return detect(sample)
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


def extract_title_from_text(text: str) -> Optional[str]:
    """Try to extract a title from the first non-empty line."""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line and len(line) < 200:
            return line
    return None

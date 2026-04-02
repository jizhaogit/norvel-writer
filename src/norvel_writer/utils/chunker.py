"""Sentence-aware text chunker with overlapping windows."""
from __future__ import annotations

import re
from typing import List, Optional

_nltk_ready = False


def _ensure_nltk() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    _nltk_ready = True


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _sent_tokenize(text: str, language: str = "english") -> List[str]:
    _ensure_nltk()
    import nltk
    try:
        return nltk.sent_tokenize(text, language=language)
    except Exception:
        # Fallback: split on period/newline
        return re.split(r"(?<=[.!?])\s+|\n{2,}", text)


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    language: str = "english",
) -> List[str]:
    """
    Split text into overlapping chunks bounded by sentence boundaries.

    Returns a list of chunk strings.
    """
    if not text or not text.strip():
        return []

    sentences = _sent_tokenize(text.strip(), language=language)
    if not sentences:
        return [text.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    overlap_buffer: List[str] = []

    for sent in sentences:
        sent_tokens = _estimate_tokens(sent)

        # If a single sentence exceeds max_tokens, hard-split it
        if sent_tokens > max_tokens:
            # Flush current
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            # Hard split on word boundaries
            words = sent.split()
            sub: List[str] = []
            sub_tokens = 0
            for w in words:
                wt = _estimate_tokens(w)
                if sub_tokens + wt > max_tokens and sub:
                    chunks.append(" ".join(sub))
                    sub, sub_tokens = [], 0
                sub.append(w)
                sub_tokens += wt
            if sub:
                chunks.append(" ".join(sub))
            continue

        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap: List[str] = []
            ot = 0
            for s in reversed(current):
                st = _estimate_tokens(s)
                if ot + st > overlap_tokens:
                    break
                overlap.insert(0, s)
                ot += st
            current = overlap
            current_tokens = ot

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def chunk_by_paragraphs(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> List[str]:
    """Alternative chunker that preserves paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    merged: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        pt = _estimate_tokens(para)
        if current_tokens + pt > max_tokens and current_parts:
            merged.append("\n\n".join(current_parts))
            # Overlap: keep last paragraph if it fits
            if _estimate_tokens(current_parts[-1]) <= overlap_tokens:
                current_parts = [current_parts[-1]]
                current_tokens = _estimate_tokens(current_parts[0])
            else:
                current_parts, current_tokens = [], 0
        current_parts.append(para)
        current_tokens += pt

    if current_parts:
        merged.append("\n\n".join(current_parts))

    return [c for c in merged if c.strip()]

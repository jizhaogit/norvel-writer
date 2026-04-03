"""
Text chunker — backed by LangChain's battle-tested splitters.

All documents stored in this app are now Markdown, so the primary splitter is
MarkdownTextSplitter, which respects structural hierarchy:

  ## Heading  →  \n\n paragraph  →  \n line  →  word  →  (char as last resort)

The character-level last resort in LangChain only fires when ALL earlier split
points are exhausted — i.e. there is no heading, no blank line, no newline, and
no space in the entire chunk.  In practice this never happens for prose.
For CJK text with no spaces, character-level splitting is actually *correct*
(each character is a natural word boundary).

Token estimate: 1 token ≈ 4 chars (consistent with the rest of the codebase).
"""
from __future__ import annotations

from typing import List


def _char_size(tokens: int) -> int:
    """Convert a token budget to an approximate character budget."""
    return max(1, tokens * 4)


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 128,
    language: str = "english",   # kept for API compatibility; not needed for Markdown
) -> List[str]:
    """
    Split *text* into overlapping chunks.

    Uses LangChain's MarkdownTextSplitter so structural boundaries
    (headings, paragraphs, sentences) are always preferred over arbitrary
    word or character cuts.

    Parameters
    ----------
    text : str
        The document text, expected to be in Markdown format.
    max_tokens : int
        Approximate maximum size of each chunk in tokens (1 token ≈ 4 chars).
    overlap_tokens : int
        Approximate overlap between adjacent chunks in tokens.
    language : str
        Ignored — kept for backward compatibility with call sites that pass
        a language code.  Markdown structure is language-agnostic.

    Returns
    -------
    List[str]
        Non-empty chunk strings.
    """
    if not text or not text.strip():
        return []

    from langchain_text_splitters import MarkdownTextSplitter

    splitter = MarkdownTextSplitter(
        chunk_size=_char_size(max_tokens),
        chunk_overlap=_char_size(overlap_tokens),
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def chunk_by_paragraphs(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 128,
) -> List[str]:
    """
    Alternative chunker that splits at paragraph boundaries first.

    Uses RecursiveCharacterTextSplitter with paragraph-first separators.
    Suitable for plain text that is not structured as Markdown.
    """
    if not text or not text.strip():
        return []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_char_size(max_tokens),
        chunk_overlap=_char_size(overlap_tokens),
        separators=["\n\n", "\n", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]

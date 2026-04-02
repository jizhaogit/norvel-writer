"""Tests for the text chunker."""
import pytest
from norvel_writer.utils.chunker import chunk_text, chunk_by_paragraphs


def test_basic_chunking():
    text = " ".join(["This is a sentence."] * 100)
    chunks = chunk_text(text, max_tokens=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip()


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_single_chunk():
    text = "Hello world. This is a short text."
    chunks = chunk_text(text, max_tokens=512)
    assert len(chunks) == 1
    assert "Hello world" in chunks[0]


def test_overlap():
    sentences = ["Sentence number {}.".format(i) for i in range(50)]
    text = " ".join(sentences)
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
    # Verify chunks are non-empty
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_paragraph_chunker():
    paras = ["Paragraph " + str(i) + ". " * 20 for i in range(10)]
    text = "\n\n".join(paras)
    chunks = chunk_by_paragraphs(text, max_tokens=200)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.strip()

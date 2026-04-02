"""Tests for the diff engine."""
import pytest
from norvel_writer.core.diff_engine import compute_diff


def test_identical_texts():
    text = "The quick brown fox jumps over the lazy dog."
    chunks = compute_diff(text, text)
    tags = {c.tag for c in chunks}
    assert "insert" not in tags
    assert "delete" not in tags
    assert "replace" not in tags


def test_simple_addition():
    original = "Hello world."
    revised = "Hello beautiful world."
    chunks = compute_diff(original, revised)
    assert any(c.tag == "insert" for c in chunks)


def test_simple_deletion():
    original = "Hello beautiful world."
    revised = "Hello world."
    chunks = compute_diff(original, revised)
    assert any(c.tag == "delete" for c in chunks)


def test_empty_original():
    chunks = compute_diff("", "new text")
    assert any(c.tag == "insert" for c in chunks)


def test_empty_revised():
    chunks = compute_diff("old text", "")
    assert any(c.tag == "delete" for c in chunks)

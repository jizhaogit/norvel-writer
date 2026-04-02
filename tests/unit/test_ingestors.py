"""Tests for file ingestors."""
import json
import pytest
from pathlib import Path


def test_txt_ingestor(tmp_path):
    from norvel_writer.ingestion.txt_ingestor import TxtIngestor
    f = tmp_path / "sample.txt"
    f.write_text("Hello world.\n\nThis is a test document.", encoding="utf-8")
    result = TxtIngestor().ingest(f)
    assert "Hello world" in result.text
    assert result.title is not None


def test_md_ingestor(tmp_path):
    from norvel_writer.ingestion.md_ingestor import MdIngestor
    f = tmp_path / "sample.md"
    f.write_text("# My Title\n\nSome **bold** text here.\n\nAnother paragraph.", encoding="utf-8")
    result = MdIngestor().ingest(f)
    assert "My Title" in result.title
    assert "bold" in result.text  # markdown stripped, but word remains
    assert "**" not in result.text


def test_json_ingestor(tmp_path):
    from norvel_writer.ingestion.json_ingestor import JsonIngestor
    data = {"name": "Aragorn", "race": "Human", "traits": ["brave", "noble"]}
    f = tmp_path / "character.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    result = JsonIngestor().ingest(f)
    assert "Aragorn" in result.text
    assert "brave" in result.text


def test_unsupported_format_returns_none():
    from norvel_writer.ingestion.pipeline import _get_ingestor
    from pathlib import Path
    assert _get_ingestor(Path("file.xyz")) is None


def test_txt_can_handle():
    from norvel_writer.ingestion.txt_ingestor import TxtIngestor
    assert TxtIngestor().can_handle(Path("file.txt"))
    assert not TxtIngestor().can_handle(Path("file.pdf"))

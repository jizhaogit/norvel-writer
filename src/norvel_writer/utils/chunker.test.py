from pathlib import Path
from norvel_writer.utils.chunker import chunk_text

def test_chunk_text_preserves_first_line_from_markdown_source():
    source_path = Path("C:\\workspace\\norvel-writer\\src\\norvel_writer\\utils\\pal_zhaolinger.md")
    text = source_path.read_text(encoding="utf-8")
    assert text.strip()
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    assert first_line.startswith("# 赵灵儿")

    chunks = chunk_text(text, max_tokens=200, overlap_tokens=40)
    assert chunks, "Expected at least one chunk"

    for c in chunks:
        assert c.strip(), "chunk must not be empty"
        assert c.splitlines()[0].strip() == first_line.strip(), "The first line must be preserved in every chunk"

path = Path("C:\\workspace\\norvel-writer\\src\\norvel_writer\\utils\\pal_zhaolinger.md")
text = path.read_text("utf-8")
chunks = chunk_text(text, max_tokens=200, overlap_tokens=40)
print("chunks", len(chunks))
for i, chunk in enumerate(chunks, 1):
    print(i, chunk)
    print("\n---\n")
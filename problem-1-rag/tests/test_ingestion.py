import pytest
from pathlib import Path
from src.ingestion import (
    chunk_text,
    generate_deterministic_chunk_id,
    extract_text_from_md,
    extract_text_from_html,
    extract_document,
)


def test_chunk_text_basic():
    text = "This is a simple sentence to test chunking. It should split cleanly into segments."
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= 35  # word boundary boundary tolerance
        assert chunk.strip() != ""


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_generate_deterministic_chunk_id():
    doc_id = "test_doc.md"
    chunk_idx = 0
    chunk_text_content = "This is reproducible content."

    id1 = generate_deterministic_chunk_id(doc_id, chunk_idx, chunk_text_content)
    id2 = generate_deterministic_chunk_id(doc_id, chunk_idx, chunk_text_content)

    assert id1 == id2
    assert len(id1) == 36  # Valid UUID string format

    # Different content produces different UUID
    id3 = generate_deterministic_chunk_id(doc_id, chunk_idx, "Different text")
    assert id1 != id3


def test_extract_md(tmp_path: Path):
    md_file = tmp_path / "sample.md"
    content = "# Title\n\nThis is a markdown document testing extraction."
    md_file.write_text(content, encoding="utf-8")

    extracted = extract_text_from_md(md_file)
    assert "Title" in extracted
    assert "extraction" in extracted


def test_extract_html(tmp_path: Path):
    html_file = tmp_path / "sample.html"
    content = "<html><head><title>Test</title><style>body {color:red;}</style></head><body><h1>Heading</h1><p>Paragraph content</p></body></html>"
    html_file.write_text(content, encoding="utf-8")

    extracted = extract_text_from_html(html_file)
    assert "Heading" in extracted
    assert "Paragraph content" in extracted
    assert "color:red" not in extracted  # <style> removed


def test_extract_document_unsupported(tmp_path: Path):
    unknown_file = tmp_path / "test.xyz"
    unknown_file.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_document(unknown_file)

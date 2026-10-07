import pytest
from app.chunking import chunk_text, load_text


def test_load_markdown():
    assert load_text("a.md", b"# Title\nhello") == "# Title\nhello"


def test_unsupported_type():
    with pytest.raises(ValueError):
        load_text("a.exe", b"x")


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n  ") == []

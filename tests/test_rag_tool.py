import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_validate_pdf_rejects_non_pdf():
    from tools.rag_tool import validate_pdf

    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        is_valid, error = validate_pdf(f.name)
        assert not is_valid
        assert "pdf" in error.lower()

def test_validate_pdf_rejects_missing_file():
    from tools.rag_tool import validate_pdf

    is_valid, error = validate_pdf("/nonexistent/path/file.pdf")

    assert not is_valid
    assert "not found" in error.lower()

def test_retrieve_returns_empty_when_no_index(tmp_path):
    from tools import rag_tool

    original = rag_tool.INDEX_FILE
    rag_tool.INDEX_FILE = tmp_path / "nonexistent.faiss"

    result = rag_tool.retrieve("What is the revenue?")

    assert result == []

    rag_tool.INDEX_FILE = original

def test_get_index_status_reflects_reality(tmp_path):
    from tools import rag_tool

    original = rag_tool.INDEX_FILE
    rag_tool.INDEX_FILE = tmp_path / "nonexistent.faiss"

    status = rag_tool.get_index_status()
    assert status["index_exists"] is False

    rag_tool.INDEX_FILE = original

def test_ingest_pdf_rejects_non_pdf():
    from tools.rag_tool import ingest_pdf

    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        result = ingest_pdf(f.name)
        assert result["success"] is False
        assert "pdf" in result["error"].lower()

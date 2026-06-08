# tests/extractor/test_html_extractor.py
from __future__ import annotations

from pathlib import Path

import pytest

from extractor.html_extractor import HTMLExtractor
from utils.models import DocumentRecord


def make_document(
    *,
    path: Path,
    media_type: str = "text/html",
    doc_id: str = "doc-html",
) -> DocumentRecord:
    return DocumentRecord(
        id=doc_id,
        source_path=path,
        media_type=media_type,
    )


def test_supports_by_media_type(tmp_path):
    extractor = HTMLExtractor()
    doc = make_document(path=tmp_path / "sample.html", media_type="text/html")
    assert extractor.supports(doc)


def test_supports_by_extension(tmp_path):
    extractor = HTMLExtractor()
    doc = make_document(path=tmp_path / "sample.xhtml", media_type="application/octet-stream")
    assert extractor.supports(doc)


def test_does_not_support_other_types(tmp_path):
    extractor = HTMLExtractor()
    doc = make_document(path=tmp_path / "sample.txt", media_type="text/plain")
    assert extractor.supports(doc) is False


def test_extract_returns_clean_result(tmp_path):
    html_content = """
        <html>
            <head><title>Sample Title</title></head>
            <body>
                <h1>Heading</h1>
                <p>Paragraph with <strong>important</strong> text.</p>
            </body>
        </html>
    """
    path = tmp_path / "example.html"
    path.write_text(html_content, encoding="utf-8")

    extractor = HTMLExtractor()
    doc = make_document(path=path)

    result = extractor.extract(doc)

    assert result.document is doc
    assert "Heading" in result.raw_text
    assert "important" in result.raw_text
    assert "<h1>" not in result.raw_text
    assert result.clean_text.startswith("Sample Title Heading")
    assert result.errors == []


def test_extract_missing_file_raises(tmp_path):
    extractor = HTMLExtractor()
    missing_path = tmp_path / "missing.html"
    doc = make_document(path=missing_path)

    with pytest.raises(FileNotFoundError):
        extractor.extract(doc)

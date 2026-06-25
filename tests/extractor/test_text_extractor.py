# tests/extractor/test_text_extractor.py
from __future__ import annotations

from pathlib import Path

from extractor.text_extractor import TextExtractor
from utils.models import DocumentRecord


def make_document(
    *,
    path: Path,
    media_type: str = "text/plain",
    doc_id: str = "doc-text",
) -> DocumentRecord:
    return DocumentRecord(
        id=doc_id,
        path=path,
        media_type=media_type,
    )


def test_supports_plain_text(tmp_path: Path) -> None:
    document = make_document(path=tmp_path / "sample.txt")
    extractor = TextExtractor()
    assert extractor.supports(document)


def test_extract_text_file(tmp_path: Path) -> None:
    content = "This is a sample paragraph. It contains enough words to be split into sentences.\nAnother sentence with useful content."
    path = tmp_path / "sample.txt"
    path.write_text(content, encoding="utf-8")

    document = make_document(path=path)
    extractor = TextExtractor()
    result = extractor.extract(document)

    assert "sample paragraph" in result.raw_text
    assert result.clean_text.startswith("This is a sample paragraph")
    assert len(result.top_sentences) >= 1


def test_extract_missing_file_raises(tmp_path: Path) -> None:
    extractor = TextExtractor()
    document = make_document(path=tmp_path / "missing.txt")

    try:
        extractor.extract(document)
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass

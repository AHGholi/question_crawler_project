from __future__ import annotations

import types
from pathlib import Path

import pytest

from extractor import doc_extractor
from extractor.doc_extractor import DOCDocxExtractor
from utils.models import DocumentRecord

docx_module = pytest.importorskip("docx")
DocxDocument = docx_module.Document


def build_document_record(
    *,
    path: Path,
    media_type: str | None = None,
    doc_id: str = "doc-record",
) -> DocumentRecord:
    return DocumentRecord(
        id=doc_id,
        path=path,
        media_type=media_type,
    )


def create_docx_file(path: Path, paragraphs: list[str]) -> None:
    document = DocxDocument()
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(path)


def test_supports_docx_extension(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    create_docx_file(docx_path, ["Sample text"])

    record = build_document_record(path=docx_path)
    extractor = DOCDocxExtractor()

    assert extractor.supports(record) is True


def test_supports_doc_media_type(tmp_path: Path) -> None:
    doc_path = tmp_path / "legacy.doc"
    doc_path.write_bytes(b"dummy-content")

    record = build_document_record(
        path=doc_path,
        media_type="application/msword",
    )
    extractor = DOCDocxExtractor()

    assert extractor.supports(record) is True


def test_extract_docx_returns_text(tmp_path: Path) -> None:
    docx_path = tmp_path / "content.docx"
    paragraphs = ["First paragraph", "Second line"]
    create_docx_file(docx_path, paragraphs)

    record = build_document_record(
        path=docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    extractor = DOCDocxExtractor()

    result = extractor.extract(record)

    assert "First paragraph" in result.clean_text
    assert "Second line" in result.clean_text
    assert record.metadata["extraction_method"] == "python-docx"
    assert record.metadata["source_extension"] == "docx"


def test_extract_docx_missing_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    docx_path = tmp_path / "missing-dep.docx"
    create_docx_file(docx_path, ["Content"])

    record = build_document_record(path=docx_path)
    extractor = DOCDocxExtractor()

    monkeypatch.setattr(doc_extractor, "DocxDocument", None, raising=False)

    with pytest.raises(RuntimeError) as exc:
        extractor.extract(record)

    assert "python-docx" in str(exc.value)


def test_extract_doc_uses_textract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc_path = tmp_path / "legacy.doc"
    doc_path.write_bytes(b"legacy-doc-binary")

    record = build_document_record(
        path=doc_path,
        media_type="application/msword",
    )
    extractor = DOCDocxExtractor()

    def fake_process(input_path: str, *_, **__) -> bytes:
        assert input_path == str(doc_path)
        return b"Legacy DOC text"

    fake_textract = types.SimpleNamespace(process=fake_process)
    monkeypatch.setattr(doc_extractor, "textract", fake_textract, raising=False)

    result = extractor.extract(record)

    assert result.clean_text == "Legacy DOC text"
    assert record.metadata["extraction_method"] == "textract"
    assert record.metadata["source_extension"] == "doc"


def test_extract_doc_requires_textract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc_path = tmp_path / "legacy.doc"
    doc_path.write_bytes(b"legacy-doc-binary")

    record = build_document_record(path=doc_path)
    extractor = DOCDocxExtractor()

    monkeypatch.setattr(doc_extractor, "textract", None, raising=False)

    with pytest.raises(RuntimeError) as exc:
        extractor.extract(record)

    assert "textract" in str(exc.value)


def test_extract_unsupported_extension(tmp_path: Path) -> None:
    unsupported_path = tmp_path / "notes.txt"
    unsupported_path.write_text("plain text")

    record = build_document_record(path=unsupported_path)
    extractor = DOCDocxExtractor()

    assert extractor.supports(record) is False
    with pytest.raises(ValueError):
        extractor.extract(record)


def test_extract_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.docx"

    record = build_document_record(path=missing_path)
    extractor = DOCDocxExtractor()

    with pytest.raises(FileNotFoundError):
        extractor.extract(record)

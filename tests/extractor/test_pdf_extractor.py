# tests/extractor/test_pdf_extractor.py
from __future__ import annotations

import dataclasses
import pathlib

import pytest

from extractor.pdf_extractor import PDFExtractor
from utils.models import DocumentRecord


def build_document(path: pathlib.Path, *, media_type: str | None = "application/pdf") -> DocumentRecord:
    kwargs = {}
    for name, field in DocumentRecord.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if name == "path":
            kwargs[name] = path
        elif name == "media_type":
            kwargs[name] = media_type
        elif field.default is not dataclasses.MISSING:
            kwargs[name] = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
            kwargs[name] = field.default_factory()  # type: ignore[misc]
        else:
            kwargs[name] = f"{name}-value"
    return DocumentRecord(**kwargs)


def _write_pdf(path: pathlib.Path, data: bytes = b"%PDF-1.0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_pdf_extractor_supports_media_type(temp_output_dir):
    path = temp_output_dir / "sample.pdf"
    _write_pdf(path)
    doc = build_document(path, media_type="application/pdf")
    extractor = PDFExtractor()
    assert extractor.supports(doc)


def test_pdf_extractor_supports_extension_without_media_type(temp_output_dir):
    path = temp_output_dir / "document.pdf"
    _write_pdf(path)
    doc = build_document(path, media_type=None)
    extractor = PDFExtractor()
    assert extractor.supports(doc)


def test_pdf_extractor_extracts_text_with_fallback(temp_output_dir, monkeypatch):
    pdf_bytes = (
        b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n2 0 obj\n<<>>\nstream\nBT\n"
        b"(Hello PDF World) Tj\nET\nendstream\nendobj\n%%EOF"
    )
    pdf_path = temp_output_dir / "fallback.pdf"
    _write_pdf(pdf_path, pdf_bytes)

    monkeypatch.setattr("extractor.pdf_extractor.PdfReader", None, raising=False)

    doc = build_document(pdf_path)
    extractor = PDFExtractor()
    result = extractor.extract(doc)

    assert "Hello PDF World" in result.raw_text
    assert result.clean_text == "Hello PDF World"
    assert result.document is doc


def test_pdf_extractor_missing_file_raises(temp_output_dir):
    missing_path = temp_output_dir / "missing.pdf"
    doc = build_document(missing_path)
    extractor = PDFExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract(doc)

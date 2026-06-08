# extractor/pdf_extractor.py
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict


from utils.models import DocumentRecord, ExtractionResult

from .base import BaseExtractor, register_extractor
from .cleaner import clean_text

try:  # pragma: no cover
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]

_TEXT_OBJECT_RE = re.compile(r"\((?:\\.|[^\\()])*\)")


class PDFExtractor(BaseExtractor):
    name = "pdf"

    SUPPORTED_MEDIA_TYPES = {"application/pdf"}
    SUPPORTED_EXTENSIONS = {".pdf"}

    def supports(self, document: DocumentRecord) -> bool:
        media_type = (document.media_type or "").lower()
        if media_type in self.SUPPORTED_MEDIA_TYPES:
            return True

        path = document.path
        return path is not None and path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract(self, document: DocumentRecord) -> ExtractionResult:
        path = self._ensure_path(document)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        data = path.read_bytes()
        raw_text = ""
        method = "heuristic"

        if PdfReader is not None:
            try:
                raw_text = self._extract_with_reader(data)
                if raw_text.strip():
                    method = "pypdf"
            except Exception:  # pragma: no cover - fallback handles failures
                raw_text = ""

        if not raw_text.strip():
            raw_text = self._extract_with_fallback(data)

        cleaned = clean_text(raw_text, strip_html=False)
        return self._build_result(document, raw_text, cleaned, method)
    
    @staticmethod
    def _ensure_path(document: DocumentRecord) -> Path:
        path = document.path or document.source_path
        if path is None:
            raise ValueError("DocumentRecord.path must be set for PDF extraction.")
        return path


    def _extract_with_reader(self, data: bytes) -> str:
        if PdfReader is None:  # pragma: no cover
            return ""

        buffer = io.BytesIO(data)
        reader = PdfReader(buffer, strict=False)  # type: ignore[arg-type]
        texts = []
        for page in getattr(reader, "pages", []):
            page_text = page.extract_text() if hasattr(page, "extract_text") else ""
            if page_text:
                texts.append(page_text)
        return "\n".join(texts).strip()

    def _extract_with_fallback(self, data: bytes) -> str:
        decoded = data.decode("latin-1", errors="ignore")
        matches = _TEXT_OBJECT_RE.findall(decoded)
        if not matches:
            return ""

        chunks = []
        for match in matches:
            chunk = match[1:-1]  # strip parentheses
            chunk = (
                chunk.replace("\\r", "\r")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\(", "(")
                .replace("\\)", ")")
                .replace("\\\\", "\\")
            )
            if chunk:
                chunks.append(chunk)
        return "\n".join(chunks).strip()

    def _build_result(
        self,
        document: DocumentRecord,
        raw_text: str,
        cleaned: str,
        method: str,
    ) -> ExtractionResult:
        fields = getattr(ExtractionResult, "__dataclass_fields__", {})
        kwargs: Dict[str, object] = {
            "document": document,
            "raw_text": raw_text,
            "clean_text": cleaned,
        }
        metadata = {"extraction_method": method}
        if "metadata" in fields:
            kwargs["metadata"] = metadata
        elif "meta" in fields:
            kwargs["meta"] = metadata
        return ExtractionResult(**kwargs)  # type: ignore[arg-type]



register_extractor("pdf", PDFExtractor)

__all__ = ["PDFExtractor"]

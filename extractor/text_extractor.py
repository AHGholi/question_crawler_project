"""Text-file extraction for plain text documents.

This module reads .txt files and produces the same extraction structure used
for HTML and PDF documents.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from extractor.base import BaseExtractor, register_extractor
from extractor.cleaner import clean_text
from utils.models import DocumentRecord, ExtractionResult


class TextExtractor(BaseExtractor):
    name = "text"

    _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def supports(self, document: DocumentRecord) -> bool:
        """Return True for plain text documents and .txt files."""
        media_type = (document.media_type or "").lower()
        path = str(document.path or document.source_path or "").lower()
        return media_type == "text/plain" or path.endswith(".txt")

    def extract(self, document: DocumentRecord) -> ExtractionResult:
        """Read the text file and return a normalized extraction result."""
        text = document.content or ""
        if not text:
            path = document.path or document.source_path
            if path is None:
                raise ValueError("DocumentRecord.path must be set for text extraction.")
            if not path.exists():
                raise FileNotFoundError(f"Document path does not exist: {path}")
            text = path.read_text(encoding=document.encoding or "utf-8", errors="ignore")

        raw_text = text.strip()
        clean_text_value = clean_text(raw_text, strip_html=False)
        top_sentences = self._split_sentences(clean_text_value)[:80]
        summary = self._build_summary(top_sentences)

        return ExtractionResult(
            document=document,
            raw_text=raw_text,
            clean_text=clean_text_value,
            summary=summary,
            top_sentences=top_sentences,
            keywords=[],
            tokens=[],
            entities=[],
            keyword_scores={},
            errors=[],
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split the cleaned text into sentence-like chunks."""
        if not text:
            return []

        parts = self._SENT_SPLIT_RE.split(text)
        out: List[str] = []
        for part in parts:
            sentence = part.strip()
            if len(sentence) < 35:
                continue
            if sentence.lower().startswith(("sign in", "log in", "copyright")):
                continue
            out.append(sentence)
        return out

    @staticmethod
    def _build_summary(sentences: List[str]) -> Optional[str]:
        if not sentences:
            return None
        return sentences[0][:320]


register_extractor("txt", TextExtractor)

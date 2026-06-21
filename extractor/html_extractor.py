from __future__ import annotations

import re
from html import unescape
from typing import List

from bs4 import BeautifulSoup

from utils.models import DocumentRecord, ExtractionResult


class HTMLExtractor:
    _WS_RE = re.compile(r"\s+")

    def supports(self, document: DocumentRecord) -> bool:
        mt = (document.media_type or "").lower()
        path = str(document.path or document.source_path or "").lower()
        return "html" in mt or path.endswith(".html") or path.endswith(".htm")

    def extract(self, document: DocumentRecord) -> ExtractionResult:
        html = document.content or ""

        # If content was not preloaded, try reading from file path
        if not html:
            p = document.path or document.source_path
            if p is not None and p.exists():
                html = p.read_text(encoding=document.encoding or "utf-8", errors="ignore")

        soup = BeautifulSoup(html, "html.parser")

        # Minimal removal only (not aggressive)
        for t in ("script", "style", "noscript"):
            for node in soup.find_all(t):
                node.decompose()

        raw_text = self._clean_text(soup.get_text(" ", strip=True))
        clean_text = raw_text  # processor can do ranking/denoise later

        # Lightweight sentence split for downstream components
        top_sentences = self._split_sentences(clean_text)[:50]

        # Keep extraction permissive
        result = ExtractionResult(
            document=document,
            raw_text=raw_text,
            clean_text=clean_text,
            summary=(top_sentences[0] if top_sentences else None),
            top_sentences=top_sentences,
            keywords=[],
            tokens=[],
            entities=[],
            keyword_scores={},
            errors=[],
        )
        return result

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if len(p.strip()) >= 20]

    @classmethod
    def _clean_text(cls, text: str) -> str:
        t = unescape(text or "")
        t = t.replace("\xa0", " ")
        t = cls._WS_RE.sub(" ", t).strip()
        return t

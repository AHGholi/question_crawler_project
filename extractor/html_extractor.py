# extractor/html_extractor.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from utils.models import DocumentRecord, ExtractionResult

from .base import BaseExtractor, register_extractor
from .cleaner import clean_text, strip_html_tags
from .keyword_extractor import KeywordExtractor 


class HTMLExtractor(BaseExtractor):
    name = "html"

    SUPPORTED_MEDIA_TYPES = {"text/html", "application/xhtml+xml"}
    SUPPORTED_EXTENSIONS = {".html", ".htm", ".xhtml"}

    def __init__(
        self,
        *,
        encoding: str = "utf-8",
        errors: str = "ignore",
        keep_tags: Optional[Sequence[str]] = None,
        lowercase: bool = False,
        collapse_to: str = " ",
        # Keyword extraction tuning
        use_bigrams: bool = True,
        threshold_fraction: float = 0.6,
        min_top_k: int = 10,
        max_terms: int = 200,
        title_boost: float = 0.20,
        bigram_boost: float = 1.5,
    ) -> None:
        self.encoding = encoding
        self.errors = errors
        self.keep_tags = keep_tags
        self.lowercase = lowercase
        self.collapse_to = collapse_to

        # Instantiate keyword extractor with tunable params
        self.keyword_extractor = KeywordExtractor(
            use_bigrams=use_bigrams,
            threshold_fraction=threshold_fraction,
            min_top_k=min_top_k,
            max_terms=max_terms,
            title_boost=title_boost,
            bigram_boost=bigram_boost,
        )

    def supports(self, document: DocumentRecord) -> bool:
        media_type = (document.media_type or "").lower()
        if media_type in self.SUPPORTED_MEDIA_TYPES:
            return True

        path = document.path or document.source_path
        if path is None:
            return False
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract(self, document: DocumentRecord) -> ExtractionResult:
        path = self._resolve_path(document)
        if not path.is_file():
            raise FileNotFoundError(f"HTML source not found: {path}")

        raw_html = path.read_text(encoding=self.encoding, errors=self.errors)
        raw_text = strip_html_tags(raw_html, keep_tags=self.keep_tags)
        cleaned_text = clean_text(
            raw_html,
            strip_html=True,
            keep_tags=self.keep_tags,
            lowercase=self.lowercase,
            collapse_to=self.collapse_to,
        )

        kw_result = self.keyword_extractor.run(cleaned_text, title=document.title)

        return ExtractionResult(
            document=document,
            raw_text=raw_text,
            clean_text=cleaned_text,
            tokens=kw_result.tokens,
            keywords=kw_result.keywords,
            keyword_scores=kw_result.scores,  # NEW
        )

    @staticmethod
    def _resolve_path(document: DocumentRecord) -> Path:
        path = document.path or document.source_path
        if path is None:
            raise ValueError("DocumentRecord.path or source_path must be set for HTML extraction.")
        return path


register_extractor(HTMLExtractor.name, HTMLExtractor)

__all__ = ["HTMLExtractor"]

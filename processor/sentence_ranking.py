# processor/sentence_ranking.py
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from utils.models import ExtractionResult


@dataclass(slots=True)
class SentenceRankingConfig:
    top_k: int = 5
    minimum_score: float = 0.05
    keyword_weight: float = 3.0
    title_weight: float = 1.5
    position_decay: float = 0.88
    length_penalty_center: int = 20
    length_penalty_sigma: float = 12.0


class SentenceRankingService:
    """
    Produces ranked sentences (list + score map) for each ExtractionResult.

    Once `rank()` or `rank_batch()` is invoked, the passed ExtractionResult(s)
    have their `top_sentences` and `sentence_scores` fields populated.
    """

    def __init__(self, config: SentenceRankingConfig | None = None) -> None:
        self.config = config or SentenceRankingConfig()

    # --- public API ----------------------------------------------------- #

    def rank(self, extraction: ExtractionResult) -> ExtractionResult:
        """
        Populate `extraction.top_sentences` and `extraction.sentence_scores`.
        Returns the same object to allow call chaining.
        """
        if extraction.top_sentences and extraction.sentence_scores:
            # Already populated – nothing to do.
            return extraction

        sentences = self._split_sentences(
            extraction.clean_text or extraction.raw_text or ""
        )
        if not sentences:
            extraction.top_sentences = []
            extraction.sentence_scores = {}
            return extraction

        token_frequencies = self._document_frequency_model(
            extraction.clean_text or extraction.raw_text or ""
        )
        title_tokens = self._tokenize(extraction.document.title or "")
        keyword_tokens = [
            token for keyword in extraction.keywords for token in self._tokenize(keyword)
        ]
        keyword_set = set(keyword_tokens)

        raw_scores = {}
        for idx, sentence in enumerate(sentences):
            tokens = self._tokenize(sentence)
            if not tokens:
                continue

            base = self._base_frequency_score(tokens, token_frequencies)
            keyword_score = self._keyword_overlap_score(tokens, keyword_set)
            title_score = self._title_overlap_score(tokens, title_tokens)
            position_bonus = self._position_score(idx)
            length_penalty = self._length_penalty(len(tokens))

            composite = (
                base
                + self.config.keyword_weight * keyword_score
                + self.config.title_weight * title_score
            )
            composite *= position_bonus * length_penalty

            if composite <= 0:
                continue
            raw_scores[sentence] = composite

        if not raw_scores:
            # Fall back to the original ordering if everything zeroed out.
            raw_scores = {sentence: 1.0 for sentence in sentences[: self.config.top_k]}

        normalized_scores = self._normalize_scores(raw_scores)
        top_sentences = self._select_top(normalized_scores)

        extraction.top_sentences = top_sentences
        extraction.sentence_scores = {s: normalized_scores[s] for s in top_sentences}
        return extraction

    def rank_batch(self, extractions: Iterable[ExtractionResult]) -> List[ExtractionResult]:
        """
        Convenience helper for processing multiple ExtractionResult instances.
        """
        return [self.rank(extraction) for extraction in extractions]

    # --- scoring helpers ------------------------------------------------ #

    def _split_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # Try to use project tokeniser if available.
        try:
            from processor.helpers import text_processing  # type: ignore

            splitter = getattr(text_processing, "split_into_sentences", None) or getattr(
                text_processing, "sentence_tokenize", None
            )
            if splitter:
                sentences = splitter(text)
                if isinstance(sentences, Sequence):
                    return [sentence.strip() for sentence in sentences if sentence.strip()]
        except Exception:
            # Fallback handled below.
            pass

        # Regex-based fallback splitting.
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"[a-z0-9]+", text)

    def _document_frequency_model(self, text: str) -> Counter:
        tokens = self._tokenize(text)
        return Counter(tokens)

    def _base_frequency_score(self, sentence_tokens: List[str], doc_freq: Counter) -> float:
        if not doc_freq:
            return 0.0
        return sum(math.log1p(doc_freq[token]) for token in sentence_tokens)

    def _keyword_overlap_score(self, sentence_tokens: List[str], keyword_set: set[str]) -> float:
        if not keyword_set:
            return 0.0
        overlap = sum(1 for token in sentence_tokens if token in keyword_set)
        return overlap / len(keyword_set)

    def _title_overlap_score(self, sentence_tokens: List[str], title_tokens: List[str]) -> float:
        if not title_tokens:
            return 0.0
        title_set = set(title_tokens)
        if not title_set:
            return 0.0
        overlap = sum(1 for token in sentence_tokens if token in title_set)
        return overlap / len(title_set)

    def _position_score(self, sentence_idx: int) -> float:
        decay = self.config.position_decay
        return math.pow(decay, sentence_idx)

    def _length_penalty(self, sentence_length: int) -> float:
        # Gaussian around the preferred sentence length.
        center = self.config.length_penalty_center
        sigma = self.config.length_penalty_sigma
        exponent = -((sentence_length - center) ** 2) / (2 * sigma**2)
        return math.exp(exponent)

    def _normalize_scores(self, raw_scores: dict[str, float]) -> dict[str, float]:
        max_score = max(raw_scores.values(), default=0.0)
        if max_score <= 0.0:
            return {sentence: self.config.minimum_score for sentence in raw_scores}
        normalized = {
            sentence: max(score / max_score, self.config.minimum_score)
            for sentence, score in raw_scores.items()
        }
        return normalized

    def _select_top(self, normalized_scores: dict[str, float]) -> List[str]:
        sorted_sentences = sorted(
            normalized_scores.items(), key=lambda item: item[1], reverse=True
        )
        top = [sentence for sentence, _ in sorted_sentences[: self.config.top_k]]
        return top

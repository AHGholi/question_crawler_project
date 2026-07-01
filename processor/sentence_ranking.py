"""Sentence-level ranking for selecting the most important passages from extracted text.

This module scores and filters candidate sentences so the question-generation
stage receives the strongest, least noisy content.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from utils.models import ExtractionResult


@dataclass(slots=True)
class SentenceRankingConfig:
    """Configuration knobs that control sentence selection and noise filtering."""
    top_k: int = 10
    minimum_score: float = 0.05
    keyword_weight: float = 3.0
    title_weight: float = 1.5
    position_decay: float = 0.88
    length_penalty_center: int = 20
    length_penalty_sigma: float = 12.0

    # --- new noise control knobs ---
    min_tokens_per_sentence: int = 5
    max_tokens_per_sentence: int = 80
    max_punct_ratio: float = 0.35


class SentenceRankingService:
    """Score and select the most informative sentences from an extraction result.

    Once ``rank()`` or ``rank_batch()`` is invoked, the passed
    ``ExtractionResult`` objects have their ``top_sentences`` and
    ``sentence_scores`` fields populated.
    """

    # Common website chrome / navigation / UI patterns
    _NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bskip to (main )?content\b", re.I),
        re.compile(r"\bmain menu\b", re.I),
        re.compile(r"\bsign in\b", re.I),
        re.compile(r"\blog in\b", re.I),
        re.compile(r"\bcreate account\b", re.I),
        re.compile(r"\bdonate\b", re.I),
        re.compile(r"\bprivacy policy\b", re.I),
        re.compile(r"\bterms of (use|service)\b", re.I),
        re.compile(r"\bcookie(s)?\b", re.I),
        re.compile(r"\btable of contents\b", re.I),
        re.compile(r"\bjump to content\b", re.I),
        re.compile(r"\brelated changes\b", re.I),
        re.compile(r"\bpermanent link\b", re.I),
        re.compile(r"\bprintable version\b", re.I),
        re.compile(r"\bdownload as pdf\b", re.I),
        re.compile(r"\bedit (links|view history)\b", re.I),
        re.compile(r"\btoggle\b", re.I),
        re.compile(r"\blanguages?\b", re.I),
    )

    def __init__(self, config: SentenceRankingConfig | None = None) -> None:
        """Store the ranking configuration used for scoring and filtering."""
        self.config = config or SentenceRankingConfig()

    def rank(self, extraction: ExtractionResult) -> ExtractionResult:
        """Populate ranked sentences and scores for a single extraction result."""
        if extraction.top_sentences and extraction.sentence_scores:
            return extraction

        source_text = extraction.clean_text or extraction.raw_text or ""
        sentences = self._split_sentences(source_text)
        if not sentences:
            extraction.top_sentences = []
            extraction.sentence_scores = {}
            return extraction

        # 1) de-duplicate while preserving order
        sentences = self._dedupe_sentences(sentences)

        # 2) remove noisy lines (processor responsibility)
        filtered_sentences = [s for s in sentences if not self._is_noisy_sentence(s)]

        # If filtering is too aggressive, gracefully fall back to original deduped sentences
        working_sentences = filtered_sentences if filtered_sentences else sentences

        token_frequencies = self._document_frequency_model(source_text)
        title_tokens = self._tokenize(extraction.document.title or "")
        keyword_tokens = [
            token for keyword in extraction.keywords for token in self._tokenize(keyword)
        ]
        keyword_set = set(keyword_tokens)

        raw_scores: dict[str, float] = {}
        for idx, sentence in enumerate(working_sentences):
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

            if composite > 0:
                raw_scores[sentence] = composite

        if not raw_scores:
            fallback = working_sentences[: self.config.top_k]
            raw_scores = {s: 1.0 for s in fallback}

        normalized_scores = self._normalize_scores(raw_scores)
        top_sentences = self._select_top(normalized_scores)

        extraction.top_sentences = top_sentences
        extraction.sentence_scores = {s: normalized_scores[s] for s in top_sentences}
        return extraction

    def rank_batch(self, extractions: Iterable[ExtractionResult]) -> List[ExtractionResult]:
        """Apply sentence ranking to a collection of extraction results."""
        return [self.rank(extraction) for extraction in extractions]

    def _split_sentences(self, text: str) -> List[str]:
        """Split a block of text into candidate sentences using a helper or fallback regex."""
        text = text.strip()
        if not text:
            return []
        try:
            from processor.helpers import text_processing  # type: ignore

            splitter = getattr(text_processing, "split_into_sentences", None) or getattr(
                text_processing, "sentence_tokenize", None
            )
            if splitter:
                sentences = splitter(text)
                if isinstance(sentences, Sequence):
                    return [s.strip() for s in sentences if s and s.strip()]
        except Exception:
            pass

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """Convert text into a simple lowercase token list for scoring."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _document_frequency_model(self, text: str) -> Counter:
        """Build a word-frequency map for the full source text."""
        return Counter(self._tokenize(text))

    def _base_frequency_score(self, sentence_tokens: List[str], doc_freq: Counter) -> float:
        """Score a sentence by the importance of its constituent tokens."""
        if not doc_freq:
            return 0.0
        return sum(math.log1p(doc_freq[t]) for t in sentence_tokens)

    def _keyword_overlap_score(self, sentence_tokens: List[str], keyword_set: set[str]) -> float:
        """Reward sentences that contain many extracted keywords."""
        if not keyword_set:
            return 0.0
        overlap = sum(1 for t in sentence_tokens if t in keyword_set)
        return overlap / max(1, len(keyword_set))

    def _title_overlap_score(self, sentence_tokens: List[str], title_tokens: List[str]) -> float:
        """Boost sentences that overlap with the document title tokens."""
        title_set = set(title_tokens)
        if not title_set:
            return 0.0
        overlap = sum(1 for t in sentence_tokens if t in title_set)
        return overlap / len(title_set)

    def _position_score(self, sentence_idx: int) -> float:
        """Give earlier sentences a mild ranking advantage."""
        return math.pow(self.config.position_decay, sentence_idx)

    def _length_penalty(self, sentence_length: int) -> float:
        """Penalize sentences that are too short or too long relative to the target length."""
        center = self.config.length_penalty_center
        sigma = self.config.length_penalty_sigma
        exponent = -((sentence_length - center) ** 2) / (2 * sigma**2)
        return math.exp(exponent)

    def _normalize_scores(self, raw_scores: dict[str, float]) -> dict[str, float]:
        """Scale raw scores into a comparable range while preserving a minimum floor."""
        max_score = max(raw_scores.values(), default=0.0)
        if max_score <= 0:
            return {s: self.config.minimum_score for s in raw_scores}
        return {
            s: max(score / max_score, self.config.minimum_score)
            for s, score in raw_scores.items()
        }

    def _select_top(self, normalized_scores: dict[str, float]) -> List[str]:
        """Return the highest-scoring sentences up to the configured limit."""
        ranked = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
        return [s for s, _ in ranked[: self.config.top_k]]

    # -------------------- new helpers -------------------- #

    def _dedupe_sentences(self, sentences: List[str]) -> List[str]:
        """Remove repeated sentences while preserving their original order."""
        seen: set[str] = set()
        out: List[str] = []
        for s in sentences:
            key = re.sub(r"\s+", " ", s.strip().lower())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(s.strip())
        return out

    def _is_noisy_sentence(self, sentence: str) -> bool:
        """Reject boilerplate, navigation, or overly short sentences before ranking."""
        s = sentence.strip()
        if not s:
            return True

        # Pattern-based UI/nav filtering
        for pat in self._NOISE_PATTERNS:
            if pat.search(s):
                return True

        tokens = self._tokenize(s)
        n = len(tokens)
        if n < self.config.min_tokens_per_sentence:
            return True
        if n > self.config.max_tokens_per_sentence:
            return True

        # punctuation-heavy heuristic
        punct_count = len(re.findall(r"[^\w\s]", s))
        ratio = punct_count / max(1, len(s))
        if ratio > self.config.max_punct_ratio:
            return True

        # Looks like language menu / link farm (many short capitalized chunks)
        chunks = re.split(r"[|/•·,;]\s*", s)
        if len(chunks) >= 8:
            short_chunks = sum(1 for c in chunks if 0 < len(c.strip()) <= 12)
            if short_chunks / len(chunks) > 0.7:
                return True

        return False

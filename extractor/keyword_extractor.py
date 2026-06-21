# extractor/keyword_extractor.py
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_STOPWORDS: set[str] = {
    # Common English stopwords (subset, ~200)
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "that",
    "this",
    "these",
    "those",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "are",
    "do",
    "does",
    "did",
    "doing",
    "done",
    "have",
    "has",
    "had",
    "having",
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "on",
    "onto",
    "of",
    "off",
    "out",
    "over",
    "to",
    "up",
    "with",
    "without",
    "between",
    "within",
    "across",
    "under",
    "above",
    "below",
    "about",
    "not",
    "no",
    "nor",
    "only",
    "own",
    "same",
    "other",
    "another",
    "such",
    "very",
    "more",
    "most",
    "less",
    "least",
    "few",
    "many",
    "much",
    "any",
    "all",
    "each",
    "either",
    "neither",
    "both",
    "some",
    "sometimes",
    "often",
    "always",
    "never",
    "ever",
    "it",
    "its",
    "itself",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "who",
    "whom",
    "whose",
    "which",
    "what",
    "where",
    "when",
    "why",
    "how",
    "there",
    "here",
    "via",
    "etc",
    # Common verbs/aux
    "make",
    "made",
    "makes",
    "go",
    "goes",
    "went",
    "gone",
    "get",
    "gets",
    "got",
    "getting",
    "use",
    "uses",
    "used",
    "using",
    # Articles/determiners
    "per",
    "each",
    # Numbers (we’ll filter numeric tokens anyway, but safe to include)
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    # Additional function words / prepositions commonly surfaced as keywords
    "among",
    "amongst",
    "around",
    "during",
    "inside",
    "outside",
    "prior",
    "toward",
    "towards",
    "through",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")
NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")


@dataclass
class KeywordResult:
    tokens: List[str]
    keywords: List[str]
    scores: Dict[str, float]


class KeywordExtractor:
    """
    Standalone keyword extractor:
      - tokenizes text
      - builds unigrams + bigrams
      - scores candidates
      - returns high-scoring terms (no fixed top-K; dynamic threshold with safety minimum)
    """

    def __init__(
        self,
        *,
        stopwords: Optional[Iterable[str]] = None,
        min_token_len: int = 2,
        max_token_len: int = 40,
        use_bigrams: bool = True,
        threshold_fraction: float = 0.6,   # keep terms with score >= fraction * max_score
        min_top_k: int = 10,               # ensure we always keep at least this many if available
        max_terms: int = 200,              # cap to avoid massive lists
        title_boost: float = 0.20,         # boost for terms appearing in title
        bigram_boost: float = 1.5,         # multiplier for bigram scores
    ) -> None:
        self.stopwords = set(stopwords or DEFAULT_STOPWORDS)
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.use_bigrams = use_bigrams
        self.threshold_fraction = threshold_fraction
        self.min_top_k = min_top_k
        self.max_terms = max_terms
        self.title_boost = title_boost
        self.bigram_boost = bigram_boost

    def run(self, text: str, *, title: Optional[str] = None) -> KeywordResult:
        # Tokenize full text (lowercased for scoring; we keep casing only for entities elsewhere)
        tokens_all = self._tokenize(text.lower())
        if not tokens_all:
            return KeywordResult(tokens=[], keywords=[], scores={})

        # Filter tokens for keyword candidates (remove stopwords, numeric, short/long)
        tokens_filtered = [
            t
            for t in tokens_all
            if not NUMERIC_RE.match(t)
            and t not in self.stopwords
            and self.min_token_len <= len(t) <= self.max_token_len
        ]

        if not tokens_filtered:
            # If everything is stopwords/numeric, fallback to all tokens to avoid empty set
            tokens_filtered = [t for t in tokens_all if not NUMERIC_RE.match(t)]

        unigram_freq = Counter(tokens_filtered)
        total_tokens = max(1, sum(unigram_freq.values()))

        # Title boost terms (lowercased tokenization)
        title_tokens = set(self._tokenize(title.lower())) if title else set()

        # Score unigrams: normalized frequency + small boost if the term appears in the title
        unigram_scores: Dict[str, float] = {}
        for term, freq in unigram_freq.items():
            base = freq / total_tokens
            boost = self.title_boost if term in title_tokens else 0.0
            unigram_scores[term] = base * (1.0 + boost)

        # Build bigrams on the filtered token sequence (preserve order)
        bigram_scores: Dict[str, float] = {}
        if self.use_bigrams:
            bigram_freq: Counter[Tuple[str, str]] = Counter()
            for i in range(len(tokens_filtered) - 1):
                w1, w2 = tokens_filtered[i], tokens_filtered[i + 1]
                if (
                    len(w1) < self.min_token_len
                    or len(w2) < self.min_token_len
                    or w1 in self.stopwords
                    or w2 in self.stopwords
                    or NUMERIC_RE.match(w1)
                    or NUMERIC_RE.match(w2)
                ):
                    continue
                bigram_freq[(w1, w2)] += 1

            # Association-based score: freq_bigram / min(freq_w1, freq_w2) with safeguards
            for (w1, w2), f_bi in bigram_freq.items():
                f1, f2 = unigram_freq[w1], unigram_freq[w2]
                denom = max(1, min(f1, f2))
                assoc = f_bi / denom
                phrase = f"{w1} {w2}"
                boost = self.title_boost if (w1 in title_tokens or w2 in title_tokens) else 0.0
                bigram_scores[phrase] = (assoc * self.bigram_boost) * (1.0 + boost)

        # Combine scores
        candidate_scores: Dict[str, float] = dict(unigram_scores)
        candidate_scores.update(bigram_scores)

        if not candidate_scores:
            return KeywordResult(tokens=tokens_filtered, keywords=[], scores={})

        # Dynamic threshold: keep terms with score >= fraction * max_score
        max_score = max(candidate_scores.values())
        keep_threshold = self.threshold_fraction * max_score

        # Sort candidates by score desc, then by length desc (prefer informative longer n-grams if tie)
        ranked = sorted(candidate_scores.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)

        # Filter by threshold, but ensure a safety minimum
        filtered = [term for term, score in ranked if score >= keep_threshold]

        if len(filtered) < self.min_top_k:
            filtered = [term for term, _ in ranked[: self.min_top_k]]

        # Cap the list to avoid huge keyword arrays
        filtered = filtered[: self.max_terms]

        return KeywordResult(tokens=tokens_filtered, keywords=filtered, scores=candidate_scores)

    @staticmethod
    def _tokenize(text: Optional[str]) -> List[str]:
        if not text:
            return []
        return WORD_RE.findall(text)

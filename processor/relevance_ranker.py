# processor\relevance_ranker.py
"""Relevance ranking utilities for ordering documents by query similarity.

This module performs lightweight TF-IDF style ranking using the processor
subsystem's text processing, vectorization, and similarity helpers.
"""

from __future__ import annotations

from typing import List, Tuple

from .text_processing import TextProcessor
from .vectorizer import TextVectorizer
from .similarity import SimilarityEngine


class RelevanceRanker:
    """Rank documents by relevance to a query using TF-IDF and cosine similarity."""

    def __init__(
        self,
        text_processor: TextProcessor | None = None,
        vectorizer: TextVectorizer | None = None,
        similarity_engine: SimilarityEngine | None = None,
    ):
        """Store the preprocessing, vectorization, and similarity services used for ranking."""
        self.text_processor = text_processor or TextProcessor()
        self.vectorizer = vectorizer or TextVectorizer()
        self.similarity_engine = similarity_engine or SimilarityEngine()

    def rank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int | None = 5,
        min_score: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """Rank document texts by relevance to the supplied query."""
        if not documents:
            return []

        processed_docs = [self.text_processor.preprocess_text(doc) for doc in documents]
        processed_query = self.text_processor.preprocess_text(query)

        doc_tfidf = self.vectorizer.fit_transform(processed_docs)
        query_tfidf = self.vectorizer.transform([processed_query])

        scores = self.similarity_engine.query_similarity(query_tfidf, doc_tfidf)

        ranked = [(i, score) for i, score in enumerate(scores) if score >= min_score]
        ranked.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

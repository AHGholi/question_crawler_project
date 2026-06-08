# extractor/relevance_ranker.py

from typing import List, Tuple

from processor.text_processing import TextProcessor
from processor.vectorizer import TextVectorizer
from processor.similarity import SimilarityEngine


class RelevanceRanker:
    """
    Ranks documents by relevance to a given query using TF-IDF and cosine similarity.
    """

    def __init__(
        self,
        text_processor: TextProcessor | None = None,
        vectorizer: TextVectorizer | None = None,
        similarity_engine: SimilarityEngine | None = None,
    ):
        self.text_processor = text_processor or TextProcessor()
        self.vectorizer = vectorizer or TextVectorizer()
        self.similarity_engine = similarity_engine or SimilarityEngine()

    def rank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int | None = 5,
        min_score: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Rank documents by relevance to the query.

        Args:
            query: Topic or query string
            documents: List of raw document texts
            top_k: Number of top documents to return (None = all)
            min_score: Minimum cosine similarity threshold

        Returns:
            List of (document_index, similarity_score) sorted descending
        """
        if not documents:
            return []

        # 1. Preprocess documents
        processed_docs = [
            self.text_processor.preprocess_text(doc)
            for doc in documents
        ]

        # 2. Preprocess query
        processed_query = self.text_processor.preprocess_text(query)

        # 3. Vectorize documents
        doc_tfidf = self.vectorizer.fit_transform(processed_docs)

        # 4. Vectorize query
        query_tfidf = self.vectorizer.transform([processed_query])

        # 5. Compute similarity scores
        scores = self.similarity_engine.query_similarity(
            query_tfidf,
            doc_tfidf
        )

        # 6. Rank and filter
        ranked = [
            (i, score)
            for i, score in enumerate(scores)
            if score >= min_score
        ]

        ranked.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

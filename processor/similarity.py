# processor/similarity.py

from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    """
    Computes cosine similarity between TF-IDF vectors.
    """

    def compute_similarity_matrix(self, tfidf_matrix) -> np.ndarray:
        """
        Compute cosine similarity between all documents.

        Args:
            tfidf_matrix: Sparse TF-IDF matrix (n_docs x n_features)

        Returns:
            Dense cosine similarity matrix (n_docs x n_docs)
        """
        return cosine_similarity(tfidf_matrix)

    def get_most_similar(
        self,
        similarity_matrix: np.ndarray,
        doc_index: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get the top-k most similar documents to a given document.

        Args:
            similarity_matrix: Cosine similarity matrix
            doc_index: Index of the reference document
            top_k: Number of similar documents to return

        Returns:
            List of (document_index, similarity_score)
        """
        similarities = similarity_matrix[doc_index]

        indexed_scores = list(enumerate(similarities))
        indexed_scores = [
            (i, score) for i, score in indexed_scores if i != doc_index
        ]

        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    def query_similarity(
        self,
        query_vector,
        document_matrix
    ) -> List[float]:
        """
        Compute similarity between a query and all documents.

        Args:
            query_vector: TF-IDF vector for the query (1 x n_features)
            document_matrix: TF-IDF matrix for documents

        Returns:
            List of similarity scores
        """
        scores = cosine_similarity(query_vector, document_matrix)
        return scores.flatten().tolist()

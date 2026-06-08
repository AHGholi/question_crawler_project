import pytest
import numpy as np
from processor.similarity import SimilarityEngine


@pytest.fixture
def tfidf_example():
    """
    Simple synthetic TF-IDF-like matrix for testing similarity.
    """
    # 3 documents, 4 hypothetical features
    return np.array([
        [0.5, 0.2, 0.0, 0.1],
        [0.4, 0.3, 0.1, 0.0],
        [0.0, 0.0, 0.8, 0.2]
    ])


def test_compute_similarity_matrix(tfidf_example):
    engine = SimilarityEngine()
    sim_matrix = engine.compute_similarity_matrix(tfidf_example)

    # Should be square and symmetric
    assert sim_matrix.shape == (3, 3)
    assert np.allclose(sim_matrix, sim_matrix.T)

    # Diagonal values must be ~1 (self-similarity)
    assert np.allclose(np.diag(sim_matrix), np.ones(3))

    # Similarity between unrelated items should be low
    assert sim_matrix[0, 2] < 0.5


def test_get_most_similar(tfidf_example):
    engine = SimilarityEngine()
    sim_matrix = engine.compute_similarity_matrix(tfidf_example)

    top_similar = engine.get_most_similar(sim_matrix, doc_index=0, top_k=2)

    # Returns a list of tuples
    assert isinstance(top_similar, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in top_similar)

    # The document itself should not appear in results
    assert all(i != 0 for i, _ in top_similar)

    # Sorted in descending order by score
    scores = [s for _, s in top_similar]
    assert scores == sorted(scores, reverse=True)


def test_query_similarity(tfidf_example):
    engine = SimilarityEngine()

    # Query similar to doc0 and doc1, different from doc2
    query_vector = np.array([[0.45, 0.25, 0.05, 0.0]])

    scores = engine.query_similarity(query_vector, tfidf_example)

    assert len(scores) == 3
    assert scores[0] > scores[2]
    assert scores[1] > scores[2]
    assert all(0 <= s <= 1 for s in scores)

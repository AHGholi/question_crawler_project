# tests/extractor/test_relevance_ranker.py

import pytest

from extractor.relevance_ranker import RelevanceRanker


@pytest.fixture
def ranker():
    return RelevanceRanker()


@pytest.fixture
def documents():
    return [
        "Machine learning models can overfit training data.",
        "Support vector machines are supervised learning models.",
        "Overfitting happens when a model memorizes the training set.",
        "This document is about cooking recipes and ingredients."
    ]


def test_rank_documents_basic(ranker, documents):
    results = ranker.rank_documents(
        query="machine learning overfitting",
        documents=documents,
        top_k=3
    )

    assert len(results) == 3

    # Scores should be sorted descending
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)

    # Indices should be valid
    for idx, score in results:
        assert 0 <= idx < len(documents)
        assert 0.0 <= score <= 1.0


def test_irrelevant_document_ranked_last(ranker, documents):
    results = ranker.rank_documents(
        query="machine learning overfitting",
        documents=documents,
        top_k=None
    )

    # Cooking document should be last or near-zero
    last_idx, last_score = results[-1]

    assert last_idx == 3
    assert last_score < 0.05


def test_top_k_limits_results(ranker, documents):
    results = ranker.rank_documents(
        query="supervised learning models",
        documents=documents,
        top_k=1
    )

    assert len(results) == 1


def test_min_score_filters_results(ranker, documents):
    results = ranker.rank_documents(
        query="machine learning",
        documents=documents,
        min_score=0.1
    )

    assert all(score >= 0.1 for _, score in results)


def test_empty_documents_returns_empty_list(ranker):
    results = ranker.rank_documents(
        query="anything",
        documents=[]
    )

    assert results == []


def test_self_similarity_dominates(ranker):
    docs = [
        "neural networks and deep learning",
        "classical mechanics and physics"
    ]

    results = ranker.rank_documents(
        query="deep learning neural networks",
        documents=docs,
        top_k=1
    )

    best_idx, best_score = results[0]

    assert best_idx == 0
    assert best_score > 0.3

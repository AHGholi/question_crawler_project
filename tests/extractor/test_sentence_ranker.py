# tests/extractor/test_sentence_ranker.py

import pytest

from extractor.sentence_ranker import SentenceRanker


@pytest.fixture
def ranker():
    return SentenceRanker()


@pytest.fixture
def documents():
    return [
        (
            "Machine learning models can overfit the training data. "
            "Regularization techniques help reduce overfitting."
        ),
        (
            "Support vector machines are supervised learning models. "
            "They are effective in high-dimensional spaces."
        ),
    ]


def test_sentence_extraction(ranker, documents):
    results = ranker.rank_sentences(
        query="machine learning",
        documents=documents,
        top_k=None
    )

    sentences = [s for s, _ in results]

    assert any("overfit" in s.lower() for s in sentences)
    assert any("support vector machines" in s.lower() for s in sentences)


def test_ranking_order_descending(ranker, documents):
    results = ranker.rank_sentences(
        query="overfitting in machine learning",
        documents=documents,
        top_k=None
    )

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_irrelevant_sentence_ranked_last(ranker, documents):
    results = ranker.rank_sentences(
        query="overfitting",
        documents=documents,
        top_k=None
    )

    last_sentence, last_score = results[-1]

    assert "high-dimensional spaces" in last_sentence.lower()
    assert last_score < 0.05


def test_top_k_limits_results(ranker, documents):
    results = ranker.rank_sentences(
        query="machine learning",
        documents=documents,
        top_k=1
    )

    assert len(results) == 1


def test_min_score_filters_results(ranker, documents):
    results = ranker.rank_sentences(
        query="overfitting",
        documents=documents,
        min_score=0.1
    )

    assert all(score >= 0.1 for _, score in results)


def test_empty_documents_returns_empty_list(ranker):
    results = ranker.rank_sentences(
        query="anything",
        documents=[]
    )

    assert results == []

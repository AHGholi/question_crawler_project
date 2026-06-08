# tests/processor/test_vectorizer.py

import processor.vectorizer
print(f"Loading vectorizer from: {processor.vectorizer.__file__}")

import pytest

from processor.vectorizer import TextVectorizer


@pytest.fixture
def sample_documents():
    return [
        ["neural", "network", "learn"],
        ["deep", "learning", "network"]
    ]


def test_fit_transform_creates_matrix(sample_documents):
    vectorizer = TextVectorizer()

    tfidf_matrix = vectorizer.fit_transform(sample_documents)

    # 2 documents x N features
    assert tfidf_matrix.shape[0] == 2
    assert tfidf_matrix.shape[1] > 0


def test_feature_names_are_created(sample_documents):
    vectorizer = TextVectorizer()
    vectorizer.fit_transform(sample_documents)

    features = vectorizer.get_feature_names()

    assert isinstance(features, list)
    assert "network" in features
    assert "neural" in features
    assert "deep" in features


def test_transform_after_fit(sample_documents):
    vectorizer = TextVectorizer()
    vectorizer.fit_transform(sample_documents)

    new_docs = [["neural", "learning"]]
    tfidf_matrix = vectorizer.transform(new_docs)

    assert tfidf_matrix.shape[0] == 1
    assert tfidf_matrix.shape[1] == len(vectorizer.get_feature_names())


def test_extract_top_keywords(sample_documents):
    vectorizer = TextVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sample_documents)

    keywords = vectorizer.extract_top_keywords(tfidf_matrix, top_k=2)

    assert len(keywords) == 2

    for doc_keywords in keywords:
        assert len(doc_keywords) <= 2

        for keyword, score in doc_keywords:
            assert isinstance(keyword, str)
            assert isinstance(score, float)
            assert score > 0

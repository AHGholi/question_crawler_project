# tests/processor/test_text_processing.py

import pytest

from processor.text_processing import TextProcessor, preprocess_text


@pytest.fixture
def processor():
    """
    Provides a TextProcessor instance for tests.
    """
    return TextProcessor()


def test_normalize_text(processor):
    text = "THIS is a TEST! Visit https://example.com 123"
    normalized = processor.normalize_text(text)

    assert normalized == "this is a test! visit"


def test_clean_punctuation(processor):
    text = "Hello, world! This is a test."
    cleaned = processor.clean_punctuation(text)

    assert cleaned == "Hello world This is a test"


def test_tokenize_and_lemmatize(processor):
    text = "Neural networks are learning systems."

    tokens = processor.tokenize(text)
    lemmas = processor.lemmatize_tokens(tokens)

    # Order matters less than content
    assert "neural" in lemmas
    assert "network" in lemmas
    assert "learn" in lemmas
    assert "system" in lemmas

    # Stopwords should be removed
    assert "are" not in lemmas


def test_full_processing_pipeline(processor):
    text = "The neural networks were trained using large datasets."

    result = processor.process(text)

    expected_keywords = {"neural", "network", "train", "large", "dataset"}

    assert set(result).intersection(expected_keywords)
    assert isinstance(result, list)
    assert all(isinstance(token, str) for token in result)


def test_preprocess_text_function():
    text = "Machine learning models learn from data."
    result = preprocess_text(text)

    assert "machine" in result
    assert "learn" in result
    assert "model" in result
    assert "datum" in result


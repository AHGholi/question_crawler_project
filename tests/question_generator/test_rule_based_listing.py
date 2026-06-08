# tests/question_generator/test_rule_based_listing.py
from __future__ import annotations

import pytest

from question_generator.generator import RuleBasedQuestionGenerator

try:
    from tests.question_generator.test_rule_based_generator import make_extraction
except ImportError:
    from test_rule_based_generator import make_extraction



@pytest.fixture(autouse=True)
def _listing_config(monkeypatch):
    overrides = {
        ("question_generator", "strategy"): "rule_based",
        ("question_generator", "strategies"): ["listing"],
        ("question_generator", "max_questions"): 5,
        ("question_generator", "min_questions"): 1,
        ("question_generator", "min_sentence_score"): 0.0,
        ("question_generator", "max_keywords_per_question"): 3,
        ("question_generator", "answer_max_chars"): 200,
        ("question_generator", "confidence_decay"): 0.05,
    }

    def fake_get_setting(prefix, key, default=None):
        return overrides.get((prefix, key), default)

    monkeypatch.setattr("question_generator.generator.get_setting", fake_get_setting)


def test_listing_engine_generates_enumeration_question():
    extraction = make_extraction()
    enumeration_sentence = (
        "Python includes lists, tuples, dictionaries, and sets as built-in data structures."
    )

    extraction.clean_text = enumeration_sentence
    extraction.raw_text = enumeration_sentence
    extraction.keywords = ["Python"]
    extraction.top_sentences = [enumeration_sentence]
    extraction.sentence_scores = {enumeration_sentence: 0.95}

    generator = RuleBasedQuestionGenerator()
    question_set = generator(extraction)

    assert question_set.questions, "Listing engine should produce at least one question."

    listing_question = question_set.questions[0]
    assert listing_question.prompt == "List the key facts about Python mentioned."
    assert listing_question.answer is not None
    answer_lower = listing_question.answer.lower()
    assert "lists" in answer_lower
    assert "tuples" in answer_lower
    assert "dictionaries" in answer_lower
    assert "sets" in answer_lower
    assert listing_question.metadata["confidence"] == "0.900"

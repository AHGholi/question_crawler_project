# tests/question_generator/test_rule_based_generator.py
from __future__ import annotations

import pytest

from question_generator.generator import RuleBasedQuestionGenerator
from utils.models import DocumentRecord, ExtractionResult


@pytest.fixture(autouse=True)
def mock_question_generator_config(monkeypatch):
    """
    Ensures deterministic configuration values for every test in this module.
    """
    config_values = {
        "strategy": "rule_based",
        "strategies": ["factoid"],
        "max_questions": 5,
        "min_questions": 1,
        "min_sentence_score": 0.4,
        "max_keywords_per_question": 2,
        "answer_max_chars": 120,
        "confidence_decay": 0.1,
    }

    def fake_get_setting(section: str, key: str, default=None):
        if section == "question_generator":
            return config_values.get(key, default)
        return default

    monkeypatch.setattr("question_generator.generator.get_setting", fake_get_setting)


def build_document() -> DocumentRecord:
    return DocumentRecord(
        id="doc-001",
        title="Test Document",
        media_type="text/plain",
    )


def make_extraction(**overrides) -> ExtractionResult:
    base = {
        "document": build_document(),
        "raw_text": "Python is a high-level programming language created by Guido van Rossum.",
        "clean_text": "Python is a high-level programming language created by Guido van Rossum.",
        "keywords": ["Python", "Guido van Rossum"],
        "top_sentences": [
            "Python is a high-level programming language created by Guido van Rossum."
        ],
        "sentence_scores": {
            "Python is a high-level programming language created by Guido van Rossum.": 0.92
        },
        "questions": [],
        "errors": [],
    }
    base.update(overrides)
    return ExtractionResult(**base)


def test_generator_produces_factoid_question():
    extraction = make_extraction()
    generator = RuleBasedQuestionGenerator()

    question_set = generator(extraction, summary=None)

    assert len(question_set.questions) >= 1
    q = question_set.questions[0]
    assert q.prompt == "What is Python?"
    answer_text = q.answer or ""
    assert answer_text, "Generator should produce an answer string"
    assert answer_text.lower() in extraction.top_sentences[0].lower()
    assert q.metadata["source_sentence"] == extraction.top_sentences[0]
    assert extraction.questions, "ExtractionResult.questions should be populated"


def test_generator_handles_missing_keywords():
    extraction = make_extraction(keywords=[])
    generator = RuleBasedQuestionGenerator()

    question_set = generator(extraction, summary=None)

    assert question_set.questions == []
    assert extraction.questions == []
    assert any("question_generator_low_yield" in err for err in extraction.errors)


def test_generator_respects_sentence_score_threshold():
    extraction = make_extraction(
        sentence_scores={
            "Python is a high-level programming language created by Guido van Rossum.": 0.2
        }
    )
    generator = RuleBasedQuestionGenerator()

    question_set = generator(extraction, summary=None)

    assert question_set.questions == []
    assert extraction.questions == []


def test_confidence_reflects_sentence_scores_and_decay():
    """
    The higher-scored sentence should yield higher confidence, and later ranks should decay.
    """
    first_sentence = "Python is a high-level programming language created by Guido van Rossum."
    second_sentence = "Python is widely used in data science and automation."
    third_sentence = "Guido van Rossum started Python as a hobby project."

    extraction = make_extraction(
        top_sentences=[first_sentence, second_sentence, third_sentence],
        sentence_scores={
            first_sentence: 0.9,
            second_sentence: 0.78,
            third_sentence: 0.1,  # expands the normalization range without generating extra questions
        },
        keywords=["Python"],  # present in the first two sentences only
    )

    generator = RuleBasedQuestionGenerator()
    question_set = generator(extraction, summary=None)

    # Find questions by source sentence
    q1 = next(q for q in question_set.questions if q.metadata["source_sentence"] == first_sentence)
    q2 = next(q for q in question_set.questions if q.metadata["source_sentence"] == second_sentence)

    c1 = float(q1.metadata["confidence"])
    c2 = float(q2.metadata["confidence"])

    # Confidence must be bounded and reflect ordering
    assert 0.0 <= c1 <= 1.0
    assert 0.0 <= c2 <= 1.0
    assert c1 > c2, "Earlier, higher-scored sentence should have higher confidence"

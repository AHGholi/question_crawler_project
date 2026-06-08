# tests/question_generator/test_rule_based_definition.py
from __future__ import annotations

import pytest

from question_generator.generator import RuleBasedQuestionGenerator
from utils.models import DocumentRecord, ExtractionResult


@pytest.fixture(autouse=True)
def mock_definition_config(monkeypatch):
    """
    Configure the generator to use only the 'definition' strategy with deterministic values.
    """
    config_values = {
        "strategy": "rule_based",
        "strategies": ["definition"],
        "max_questions": 5,
        "min_questions": 1,
        "min_sentence_score": 0.4,
        "max_keywords_per_question": 2,
        "answer_max_chars": 160,
        "confidence_decay": 0.1,
    }

    def fake_get_setting(section: str, key: str, default=None):
        if section == "question_generator":
            return config_values.get(key, default)
        return default

    monkeypatch.setattr("question_generator.generator.get_setting", fake_get_setting)


def build_document() -> DocumentRecord:
    return DocumentRecord(
        id="doc-def-001",
        title="Definition Test Document",
        media_type="text/plain",
    )


def make_extraction(**overrides) -> ExtractionResult:
    base = {
        "document": build_document(),
        "raw_text": "Python is a high-level programming language created by Guido van Rossum.",
        "clean_text": "Python is a high-level programming language created by Guido van Rossum.",
        "keywords": ["Python"],
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


def test_definition_engine_produces_definition_question():
    extraction = make_extraction()
    generator = RuleBasedQuestionGenerator()

    question_set = generator(extraction, summary=None)

    assert len(question_set.questions) >= 1

    q = question_set.questions[0]
    assert q.prompt == "Define Python."
    assert q.answer, "Definition engine should produce an answer"
    assert "high-level" in q.answer.lower() or "programming language" in q.answer.lower()

    # Metadata should carry the source sentence and confidence
    assert q.metadata["source_sentence"] == extraction.top_sentences[0]
    conf = float(q.metadata["confidence"])
    assert 0.0 <= conf <= 1.0

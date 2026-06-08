#tests\integration\test_phase2_to_phase3_pipeline.py

import pytest

from question_generator.generator import RuleBasedQuestionGenerator


@pytest.fixture(scope="module")
def latest_phase2_payloads():
    from tests.integration.test_phase2_processing_pipeline import collect_latest_phase2_contexts

    contexts, metadata = collect_latest_phase2_contexts()
    assert contexts, "Phase 2 pipeline produced no contexts—ensure Phase 1 ran successfully"

    extractions = []
    for entry in contexts:
        context = entry["context"]
        document = getattr(context, "document", None)
        doc_id = (
            getattr(context, "document_id", None)
            or (getattr(document, "id", None) if document else None)
            or entry.get("document_id")
            or "unknown-document"
        )

        extraction = getattr(context, "extraction", None)
        assert extraction is not None, f"{doc_id} is missing extraction results"

        top_sentences = getattr(extraction, "top_sentences", None)
        assert top_sentences, f"{doc_id} has no top sentences"

        sentence_scores = getattr(extraction, "sentence_scores", None)
        if isinstance(sentence_scores, dict):
            normalized_scores = sentence_scores
        elif isinstance(sentence_scores, list):
            normalized_scores = {
                sentence: score for sentence, score in sentence_scores if isinstance(sentence, str)
            }
        else:
            normalized_scores = {}

        assert normalized_scores, f"{doc_id} has empty sentence_scores"

        extraction.sentence_scores = normalized_scores
        extractions.append(extraction)

    return extractions, metadata


@pytest.fixture
def question_generator(monkeypatch):
    def fake_get_setting(prefix: str, key: str, default=None):
        overrides = {
            "min_questions": 3,
            "max_questions": 12,
            "min_sentence_score": 0.10,
            "confidence_decay": 0.05,
        }
        return overrides.get(key, default)

    monkeypatch.setattr("question_generator.generator.get_setting", fake_get_setting)
    return RuleBasedQuestionGenerator()


@pytest.mark.integration
def test_phase2_to_phase3_end_to_end(latest_phase2_payloads, question_generator):
    extractions, metadata = latest_phase2_payloads
    assert metadata["total_documents"] >= len(extractions) >= 3, "Unexpected Phase 2 coverage"

    for extraction in extractions:
        question_set = question_generator(extraction)
        assert question_set is not None, f"No output generated for {extraction.document_id}"
        assert question_set.questions, f"Question set empty for {extraction.document_id}"

        for question in question_set.questions:
            assert question.prompt, "Question prompt missing"
            answer = getattr(question, "answer", None)
            assert isinstance(answer, str) and answer.strip(), "Question answer missing or empty"
            confidence = float(question.metadata["confidence"])
            assert 0 < confidence <= 1, "Confidence out of bounds"
            assert question.metadata.get("strategy"), "Strategy tag missing"


@pytest.mark.integration
def test_factoid_strategy_focus(latest_phase2_payloads, monkeypatch):
    monkeypatch.setattr(
        "question_generator.generator.get_setting",
        lambda prefix, key, default=None: {
            "strategies": ["factoid"],
            "min_questions": 1,
            "max_questions": 5,
            "min_sentence_score": 0.10,
            "confidence_decay": 0.05,
        }.get(key, default),
    )

    generator = RuleBasedQuestionGenerator()
    extractions, _ = latest_phase2_payloads
    questions = []

    for extraction in extractions:
        questions.extend(generator(extraction).questions)

    assert questions, "Factoid strategy produced no questions"
    for question in questions:
        answer = getattr(question, "answer", "")
        assert isinstance(answer, str) and answer.strip(), "Factoid answer missing or empty"
        answer_lower = answer.lower()
        assert any(term in answer_lower for term in ("machine", "learning")), "Factoid answer off-topic"


@pytest.mark.integration
def test_definition_strategy_depth(latest_phase2_payloads, monkeypatch):
    monkeypatch.setattr(
        "question_generator.generator.get_setting",
        lambda prefix, key, default=None: {
            "strategies": ["definition"],
            "min_questions": 1,
            "max_questions": 5,
            "min_sentence_score": 0.10,
            "confidence_decay": 0.05,
        }.get(key, default),
    )

    generator = RuleBasedQuestionGenerator()
    extractions, _ = latest_phase2_payloads

    for extraction in extractions:
        question_set = generator(extraction)
        assert question_set.questions, "Definition strategy returned no questions"
        for question in question_set.questions:
            answer = getattr(question, "answer", "") or ""
            assert isinstance(answer, str) and answer.strip(), "Definition answer missing or empty"
            words = answer.split()
            assert len(words) > 5, "Definition answer too short"
            lowered = answer.lower()
            assert " is " in lowered or " are " in lowered, "Definition missing copular verb"

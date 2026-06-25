# tests/question_generator/test_adapter_question_filter.py
from __future__ import annotations

from question_generator.adapter import PipelineQuestionGenerator
from question_generator.base import QGBackend, QGInput


class DummyBackend(QGBackend):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def generate(self, data: QGInput) -> list[str]:
        return self.outputs


def test_adapter_filters_statement_like_items() -> None:
    backend = DummyBackend([
        "This is a statement rather than a question.",
        "Q: What is machine learning? A: A field of study that uses algorithms to learn from data.",
    ])
    qg = PipelineQuestionGenerator(backend=backend, default_num_questions=5)
    qset = qg.generate(QGInput(topic="Test", title="Test", summary="", keywords=[], sentences=[], num_questions=5))

    assert len(qset.questions) == 1
    assert qset.questions[0].question.startswith("What is machine learning?")


def test_adapter_preserves_qa_pairs() -> None:
    backend = DummyBackend([
        "Q: Who invented the Light Bulb? A: Thomas Edison"
    ])
    qg = PipelineQuestionGenerator(backend=backend, default_num_questions=5)
    qset = qg.generate(QGInput(topic="Test", title="Test", summary="", keywords=[], sentences=[], num_questions=5))

    assert qset.questions[0].question == "Who invented the Light Bulb?"
    assert qset.questions[0].answer == "Thomas Edison"

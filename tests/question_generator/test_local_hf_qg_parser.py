# tests/question_generator/test_local_hf_qg_parser.py
from __future__ import annotations

from question_generator.local_hf_qg import LocalHFQuestionGenerator


def test_extract_qa_pairs_same_line_answer() -> None:
    raw = "Question: What is AI? Answer: Artificial intelligence"
    pairs = LocalHFQuestionGenerator._extract_qa_pairs(raw)
    assert pairs == [("What is AI?", "Artificial intelligence")]


def test_extract_qa_pairs_multi_line_answer() -> None:
    raw = "Q: What is machine learning?\nA: A method of teaching computers."
    pairs = LocalHFQuestionGenerator._extract_qa_pairs(raw)
    assert pairs == [("What is machine learning?", "A method of teaching computers.")]


def test_extract_qa_pairs_answer_with_line_breaks() -> None:
    raw = "Q: Define entropy.\nA: A measure of disorder\nthat appears in thermodynamics."
    pairs = LocalHFQuestionGenerator._extract_qa_pairs(raw)
    assert pairs == [("Define entropy.", "A measure of disorder that appears in thermodynamics.")]


def test_extract_qa_pairs_same_line_options_answer() -> None:
    raw = (
        "Question: If you're new to machine learning, we recommend completing modules in the order below."
        "Options:A An introduction to logistic regression, where ML models are designed to predict the probability of a given outcome."
    )
    pairs = LocalHFQuestionGenerator._extract_qa_pairs(raw)
    assert pairs == [(
        "If you're new to machine learning, we recommend completing modules in the order below.",
        "An introduction to logistic regression, where ML models are designed to predict the probability of a given outcome.",
    )]

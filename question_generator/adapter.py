# question_generator/adapter.py
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable, Optional, Sequence

from question_generator.base import QGBackend, QGInput
from utils.models import (
    DocumentRecord,
    ExtractionResult,
    QuestionItem,
    QuestionSet,
)


@dataclass(slots=True)
class _AdapterConfig:
    max_questions_default: int = 5
    min_question_len: int = 20
    min_answer_len: int = 5
    question_similarity_threshold: float = 0.85
    overlap_confidence_scale: float = 0.85
    mcq_option_count: int = 4
    mcq_noise_penalty: float = 0.1
    random_seed: int = 7


_WHITESPACE_RE = re.compile(r"\s+")
_MC_STEM_RE = re.compile(
    r"(which|all|none|each)\s+of\s+the\s+(following|statements)|"
    r"(is|are)\s+not\s+(true|correct)|"
    r"except\b|not\s+(true|correct)",
    re.IGNORECASE,
)
_OPTION_RE = re.compile(r"^[A-D]\s*[\).\:\-]\s+", re.IGNORECASE)
_PUNCT_TRIM = re.compile(r"^[\"'“”‘’`\(\)\[\]{}<>]+|[\"'“”‘’`\(\)\[\]{}<>]+$")
_TRAILING_JUNK_SPLIT_RE = re.compile(
    r"\b(?:options?|choices?|key\s*terms?|keywords?)\s*:\s*",
    re.IGNORECASE,
)
_STRIP_REFS_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _strip_inline_refs(text: str) -> str:
    text = _STRIP_REFS_RE.sub("", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text


def _truncate(text: str, max_len: int) -> str:
    t = text.strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def _normalize_question_text(text: str) -> str:
    text = text.strip()
    if text.lower().startswith("question:"):
        text = text.split(":", 1)[1]

    text = _PUNCT_TRIM.sub("", text)

    m = _TRAILING_JUNK_SPLIT_RE.search(text)
    if m:
        text = text[: m.start()]

    text = _normalize_whitespace(text)
    text = re.sub(r"[\[\]\(\)]+$", "", text).strip()
    return text


def _normalize_answer_text(text: str) -> str:
    text = _strip_inline_refs(text)
    text = _normalize_whitespace(text)
    text = re.sub(r"([.?!])\1+$", r"\1", text)
    return text


def _token_set(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"\b\w+\b", text)}


def _sentence_score(target: str, candidates: Sequence[str]) -> tuple[str, float]:
    target_tokens = _token_set(target)
    best_sentence = ""
    best_overlap = 0.0

    for sent in candidates:
        sent_tokens = _token_set(sent)
        if not sent_tokens:
            continue
        overlap = len(target_tokens & sent_tokens) / max(len(target_tokens), 1)
        if overlap > best_overlap:
            best_sentence = sent
            best_overlap = overlap

    return best_sentence, best_overlap


def _fallback_answer(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return "Answer not available."
    cutoff = sentence.find(". ")
    if 0 < cutoff < 240:
        return sentence[: cutoff + 1]
    return sentence[:240].rstrip() + ("..." if len(sentence) > 240 else "")


def _looks_like_incomplete_mcq(question: str) -> bool:
    return bool(_MC_STEM_RE.search(question) and not _OPTION_RE.search(question))


def _extract_existing_options(answer: str) -> list[str]:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    return [line for line in lines if _OPTION_RE.match(line)]


def _candidate_sentences(
    sentences: Sequence[str],
    keywords: Sequence[str],
    summary: Optional[str],
    source: str,
) -> list[str]:
    pool: list[str] = []
    pool.extend(sentence.strip() for sentence in sentences if sentence.strip())

    if summary:
        pool.extend(part.strip() for part in re.split(r"[.\n]", summary) if part.strip())

    pool.extend(keywords or [])
    pool.append(source)

    seen: set[str] = set()
    unique: list[str] = []
    for item in pool:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _pick_distractors(correct: str, pool: Sequence[str], count: int) -> list[str]:
    correct_norm = _normalize_answer_text(correct)
    correct_lower = correct_norm.lower()
    target_tokens = _token_set(correct_norm)

    scored: list[tuple[float, str]] = []
    for text in pool:
        norm = _normalize_answer_text(text)
        if not norm:
            continue
        if norm.lower() == correct_lower:
            continue

        tokens = _token_set(norm)
        if not tokens:
            continue

        overlap = len(target_tokens & tokens) / max(len(target_tokens | tokens), 1)
        scored.append((1 - overlap, norm))

    scored.sort()

    distractors: list[str] = []
    seen: set[str] = set()
    for _, norm in scored:
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        distractors.append(_truncate(norm, 140))
        if len(distractors) >= count:
            break

    while len(distractors) < count:
        distractors.append(f"Option {len(distractors) + 1}")

    return distractors[:count]


def _format_mcq_answer(correct: str, distractors: Sequence[str]) -> str:
    normalized_correct = _truncate(_normalize_answer_text(correct), 140)
    options = list(distractors) + [normalized_correct]

    # Defensive dedupe
    deduped: list[str] = []
    seen: set[str] = set()
    for opt in options:
        key = opt.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(opt.strip())

    if len(deduped) < 2:
        deduped.extend(["Option X", "Option Y"])

    random.shuffle(deduped)

    labels = ["A", "B", "C", "D", "E"]
    limited = deduped[: len(labels)]

    lines: list[str] = []
    correct_label: Optional[str] = None

    for idx, option in enumerate(limited):
        label = labels[idx]
        lines.append(f"{label}) {option}")
        if option.lower() == normalized_correct.lower():
            correct_label = label

    if correct_label is None:
        correct_label = labels[min(len(limited) - 1, len(labels) - 1)]

    lines.append(f"Correct: {correct_label}")
    return "\n".join(lines)


class PipelineQuestionGenerator:
    def __init__(
        self,
        backend: QGBackend,
        config: Optional[_AdapterConfig] = None,
        default_num_questions: Optional[int] = None,
    ) -> None:
        self.backend = backend
        if config is not None:
            self.config = config
        elif default_num_questions is not None:
            self.config = _AdapterConfig(max_questions_default=default_num_questions)
        else:
            self.config = _AdapterConfig()

        random.seed(self.config.random_seed)

    def __call__(
        self,
        extraction: ExtractionResult,
        summary: object | None = None,
    ) -> QuestionSet:
        clean_text = (extraction.clean_text or "").strip()

        topic = (
            (extraction.document.title or "").strip()
            or (clean_text[:200] if clean_text else "")
            or "document"
        )
        title = extraction.document.title

        keywords = list(extraction.keywords or [])
        sentences = list(extraction.top_sentences or [])

        summary_text: str | None = None
        if summary is not None:
            summary_text = (
                str(
                    getattr(summary, "summary_text", None)
                    or getattr(summary, "text", None)
                    or ""
                ).strip()
                or None
            )
        if summary_text is None:
            summary_text = (extraction.summary or "").strip() or None

        qg_input = QGInput(
            topic=topic,
            title=title,
            summary=summary_text,
            keywords=keywords,
            sentences=sentences,
            num_questions=self.config.max_questions_default,
        )

        qset = self.generate(qg_input)

        # Keep original document from extraction
        qset.document = extraction.document
        for qi in qset.questions:
            qi.source_document_id = extraction.document.id

        return qset

    def generate(self, qg_input: QGInput) -> QuestionSet:
        raw_questions = self.backend.generate(qg_input)
        normalized = self._normalize_and_filter(raw_questions)

        document = self._build_document_record(qg_input)
        questions = self._build_question_items(
            document=document,
            normalized_questions=normalized,
            sentences=qg_input.sentences,
            summary=qg_input.summary,
            keywords=qg_input.keywords,
        )

        return QuestionSet(
            document=document,
            questions=questions,
            generated_at=datetime.now(UTC),
            strategy="pipeline_v2_mcq",
        )

    def _build_document_record(self, qg_input: QGInput) -> DocumentRecord:
        return DocumentRecord(
            id=qg_input.topic or qg_input.title or "document",
            title=qg_input.title or qg_input.topic,
        )

    def _normalize_and_filter(self, items: Iterable[str]) -> list[str]:
        unique: list[str] = []
        for item in items:
            question = _normalize_question_text(item)
            if len(question) < self.config.min_question_len:
                continue
            if self._is_duplicate(question, unique):
                continue
            unique.append(question)
        return unique

    def _is_duplicate(self, question: str, existing: Sequence[str]) -> bool:
        q_tokens = _token_set(question)
        for other in existing:
            if not other:
                continue
            overlap = len(q_tokens & _token_set(other)) / max(len(q_tokens), 1)
            if overlap >= self.config.question_similarity_threshold:
                return True
        return False

    def _build_question_items(
        self,
        document: DocumentRecord,
        normalized_questions: Sequence[str],
        sentences: Sequence[str],
        summary: Optional[str],
        keywords: Sequence[str],
    ) -> list[QuestionItem]:
        items: list[QuestionItem] = []
        source_doc_id = document.id

        for question_text in normalized_questions:
            source_sentence, overlap = _sentence_score(question_text, sentences)
            fallback_sentence = source_sentence or (sentences[0] if sentences else "")
            fallback_sentence = _normalize_answer_text(fallback_sentence)

            answer = _normalize_answer_text(_fallback_answer(fallback_sentence))

            confidence = min(0.4 + overlap * self.config.overlap_confidence_scale, 0.95)

            tags: list[str] = []
            if _looks_like_incomplete_mcq(question_text):
                tags.append("mcq_detected")
                existing_options = _extract_existing_options(answer)
                if not existing_options:
                    candidates = _candidate_sentences(
                        sentences=sentences,
                        keywords=keywords,
                        summary=summary,
                        source=fallback_sentence,
                    )
                    distractors = _pick_distractors(
                        correct=answer,
                        pool=candidates,
                        count=max(self.config.mcq_option_count - 1, 1),
                    )
                    answer = _format_mcq_answer(answer, distractors)
                    tags.append("mcq_generated_options")
                    confidence = min(confidence, 0.72)
                else:
                    tags.append("mcq_existing_options")

            items.append(
                QuestionItem(
                    question=question_text,
                    answer=answer,
                    source_document_id=source_doc_id,
                    source_sentence=fallback_sentence,
                    confidence=confidence,
                    tags=tags,
                )
            )

        return items

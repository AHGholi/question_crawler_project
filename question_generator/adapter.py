from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from processor.text_processing import chunk_text
from question_generator.base import QGBackend, QGInput
from utils.models import ExtractionResult, QuestionItem, QuestionSet

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_TRAIL_PUNCT_RE = re.compile(r"[.?!]+$")
_QA_SPLIT_RE = re.compile(r"\s*(?:\|\||\|)\s*")
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:a|answer)\s*[:\-]\s*", re.IGNORECASE)


def _norm_ws(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", (text or "").strip())


def _token_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "")}


@dataclass(slots=True)
class _AdapterConfig:
    max_questions_default: int = 10
    min_question_len: int = 10
    max_question_len: int = 220
    question_similarity_threshold: float = 0.93
    overgenerate_floor: int = 8
    overgenerate_factor: int = 1
    overlap_confidence_scale: float = 0.85

    # NEW: chunk settings
    chunk_max_tokens: int = 220
    chunk_overlap_sentences: int = 1
    chunk_min_chars: int = 180
    max_chunks_used: int = 10


class PipelineQuestionGenerator:
    def __init__(
        self,
        backend: QGBackend,
        default_num_questions: int = 10,
        config: Optional[_AdapterConfig] = None,
    ) -> None:
        self.backend = backend
        self.config = config or _AdapterConfig(max_questions_default=default_num_questions)
        self.config.max_questions_default = int(default_num_questions)

    def __call__(self, extraction: ExtractionResult, num_questions: Optional[int] = None) -> QuestionSet:
        target = max(int(num_questions or self.config.max_questions_default), 1)

        qg_input = self._build_qg_input(extraction, target)
        ask_for = max(target * self.config.overgenerate_factor, self.config.overgenerate_floor)

        logger.info(
            "[QG][adapter] start doc_id=%s target=%d ask_for=%d chunks=%d",
            getattr(extraction.document, "id", None),
            target,
            ask_for,
            len(qg_input.chunks),
        )

        boosted = QGInput(
            topic=qg_input.topic,
            title=qg_input.title,
            summary=qg_input.summary,
            keywords=qg_input.keywords,
            sentences=qg_input.sentences,
            chunks=qg_input.chunks,  # NEW
            num_questions=ask_for,
        )

        raw = self.backend.generate(boosted) or []
        logger.info("[QG][adapter] backend_returned=%d requested=%d", len(raw), ask_for)
        logger.debug("[QG][adapter] backend_sample=%s", raw[:5])

        questions = self._normalize_and_filter(raw)[:target]
        logger.info("[QG][adapter] final_selected=%d target=%d", len(questions), target)
        logger.debug("[QG][adapter] final_sample=%s", questions[:5])

        items = self._build_question_items(questions, extraction)

        return QuestionSet(
            document=extraction.document,
            questions=items,
            strategy="pipeline_qg_adapter",
        )

    def _build_qg_input(self, extraction: ExtractionResult, target: int) -> QGInput:
        title = (extraction.document.title or "").strip() if extraction.document else ""
        topic = title or "Document"

        summary = (extraction.summary or "").strip()
        if not summary:
            summary = (extraction.clean_text or extraction.raw_text or "")[:700].strip()

        keywords = [k.strip() for k in (extraction.keywords or []) if (k or "").strip()]
        sentences = [s.strip() for s in (extraction.top_sentences or []) if (s or "").strip()]

        # NEW: build chunks from full clean text (fallback raw text)
        base_text = (extraction.clean_text or extraction.raw_text or "").strip()
        chunks: List[str] = []
        if base_text:
            try:
                chunks = chunk_text(
                    base_text,
                    max_tokens=self.config.chunk_max_tokens,
                    overlap_sentences=self.config.chunk_overlap_sentences,
                    min_chunk_chars=self.config.chunk_min_chars,
                )
            except Exception as exc:
                logger.warning("[QG][adapter] chunk_text failed: %s", exc)
                chunks = []

        # guardrail: avoid too many chunks for latency
        if len(chunks) > self.config.max_chunks_used:
            chunks = chunks[: self.config.max_chunks_used]

        logger.debug(
            "[QG][adapter] input topic=%r title=%r summary_len=%d keywords=%d sentences=%d chunks=%d target=%d",
            topic,
            title,
            len(summary),
            len(keywords),
            len(sentences),
            len(chunks),
            target,
        )

        return QGInput(
            topic=topic,
            title=title or None,
            summary=summary or None,
            keywords=keywords,
            sentences=sentences,
            chunks=chunks,  # NEW
            num_questions=target,
        )

    def _normalize_and_filter(self, raw_questions: Sequence[str]) -> List[str]:
        kept: List[str] = []

        drop_empty = 0
        drop_invalid = 0
        drop_duplicate = 0

        for raw in raw_questions:
            q, _ = self._split_question_answer(raw)
            q = self._normalize_question(q)

            if not q:
                drop_empty += 1
                continue

            if not self._is_valid_question(q):
                drop_invalid += 1
                continue

            if self._is_duplicate(q, kept):
                drop_duplicate += 1
                continue

            kept.append(q)

        logger.info(
            "[QG][adapter][filter_stats] input=%d kept=%d drop_empty=%d drop_invalid=%d drop_duplicate=%d",
            len(raw_questions),
            len(kept),
            drop_empty,
            drop_invalid,
            drop_duplicate,
        )
        logger.debug("[QG][adapter] kept_sample=%s", kept[:5])

        return kept

    def _split_question_answer(self, raw: str) -> tuple[str, str]:
        text = _norm_ws(raw)

        if "||" in text or "|" in text:
            parts = _QA_SPLIT_RE.split(text, maxsplit=1)
            if len(parts) == 2:
                q = _norm_ws(parts[0])
                a = _norm_ws(_ANSWER_PREFIX_RE.sub("", parts[1]))
                return q, a

        m = re.search(r"\s(?:A|Answer)\s*[:\-]\s*", text, flags=re.IGNORECASE)
        if m:
            q = _norm_ws(text[: m.start()])
            a = _norm_ws(text[m.end() :])
            return q, a

        return text, ""

    def _normalize_question(self, text: str) -> str:
        t = _norm_ws(text)
        t = _TRAIL_PUNCT_RE.sub("", t).strip()
        if not t:
            return ""
        if t[0].isalpha():
            t = t[0].upper() + t[1:]
        if not t.endswith("?"):
            t += "?"
        return t

    def _is_valid_question(self, q: str) -> bool:
        if not q:
            return False
        if len(q) < self.config.min_question_len or len(q) > self.config.max_question_len:
            return False
        if not _token_set(q):
            return False

        ql = q.lower()
        generic = (
            "what is this text about",
            "summarize the text",
            "what are the key points",
            "what is the main idea",
        )
        return not any(ql.startswith(g) for g in generic)

    def _is_duplicate(self, q: str, existing: Sequence[str]) -> bool:
        qt = _token_set(q)
        if not qt:
            return False

        for other in existing:
            ot = _token_set(other)
            if not ot:
                continue
            jaccard = len(qt & ot) / max(len(qt | ot), 1)
            if jaccard >= self.config.question_similarity_threshold:
                return True
        return False

    def _build_question_items(self, questions: Sequence[str], extraction: ExtractionResult) -> List[QuestionItem]:
        items: List[QuestionItem] = []
        sentence_pool = extraction.top_sentences or []

        for q in questions:
            source_sentence, conf = self._best_source_for_question(q, sentence_pool)
            answer = ""

            items.append(
                QuestionItem(
                    question=q,
                    answer=answer,
                    source_document_id=extraction.document.id,
                    source_sentence=source_sentence or "",
                    confidence=conf,
                    tags=["auto", "qg"],
                )
            )

        logger.info("[QG][adapter] built_question_items=%d", len(items))
        return items

    def _best_source_for_question(self, question: str, sentences: Sequence[str]) -> tuple[Optional[str], float]:
        if not sentences:
            return None, 0.35

        q_tokens = _token_set(question)
        if not q_tokens:
            return sentences[0], 0.4

        best_s: Optional[str] = None
        best_overlap = 0.0

        for s in sentences[:40]:
            st = _token_set(s)
            if not st:
                continue
            overlap = len(q_tokens & st) / max(len(q_tokens), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_s = s

        if best_s is None:
            return sentences[0], 0.4

        confidence = min(0.99, 0.45 + best_overlap * self.config.overlap_confidence_scale)
        return best_s, confidence

    def _fallback_answer(self, source_sentence: Optional[str], extraction: ExtractionResult) -> str:
        if source_sentence and source_sentence.strip():
            return source_sentence.strip()
        if extraction.summary and extraction.summary.strip():
            return extraction.summary.strip()
        txt = (extraction.clean_text or extraction.raw_text or "").strip()
        return txt[:220].strip() if txt else "Not available."

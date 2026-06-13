from __future__ import annotations

import logging
import re
from collections.abc import Iterable as IterableABC
from typing import Any, List, Optional, Tuple

from utils.config import get_setting
from utils.models import ExtractionResult, Question, QuestionItem, QuestionSet

from .configuration import GeneratorConfig
from .neural_qg import NeuralQGEngine


logger = logging.getLogger(__name__)


class NeuralQuestionGenerator:
    """
    Neural-only QuestionGeneratorStep:
      __call__(extraction: ExtractionResult, summary: Optional[str]) -> QuestionSet
    """

    DEFAULTS: dict[str, Any] = {
        "strategy": "neural",
        "strategies": ["neural"],
        "max_questions": 10,
        "min_questions": 0,
        "min_sentence_score": 0.45,
        "max_keywords_per_question": 2,
        "answer_max_chars": 280,
        "confidence_decay": 0.1,
        "model_name": "valhalla/t5-small-qg-prepend",
        "input_format": "prepend",  # "prepend" or "highlight"
        "device": "auto",           # "auto", "cuda", "cpu"
        "max_input_length": 256,
        "max_output_length": 64,
        "num_beams": 2,
        "num_return_sequences": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_answers_per_sentence": 2,
        "max_keyword_candidates": 60,
    }

    def __init__(self, *, config_prefix: str = "question_generator") -> None:
        self.config_prefix = config_prefix
        self._engine: NeuralQGEngine | None = None

    def __call__(self, extraction: ExtractionResult, summary: Optional[str] = None) -> Optional[QuestionSet]:
        config = self._load_config()

        ranked_sentences = self._assemble_ranked_sentences(
            extraction.top_sentences,
            extraction.sentence_scores,
            fallback_text=extraction.clean_text,
        )

        logger.info(
            "neural_qg_input: ranked_sentences=%d, top_sentences=%d, scores=%d, clean_text_len=%d, min_sentence_score=%.3f",
            len(ranked_sentences),
            len(extraction.top_sentences or []),
            len(extraction.sentence_scores or {}),
            len(extraction.clean_text or ""),
            config.min_sentence_score,
        )

        if not ranked_sentences:
            logger.warning("No ranked sentences available; cannot generate questions.")
            extraction.questions = []
            return None

        engine = self._get_engine(config)

        # Build keyword candidates
        keyword_candidates, max_kw_score = self._get_keyword_candidates(
            extraction=extraction, max_items=config.max_keyword_candidates
        )

        # Build (answer, sentence) pairs
        pairs: List[Tuple[str, str, float]] = []
        for sentence, score in ranked_sentences:
            if score < config.min_sentence_score:
                continue

            sentence_answers = self._answers_for_sentence(
                sentence=sentence,
                keyword_candidates=keyword_candidates,
                max_answers=config.max_answers_per_sentence,
                answer_max_chars=config.answer_max_chars,
            )

            for answer, kw_score in sentence_answers:
                pairs.append((answer, sentence, kw_score))

            if len(pairs) >= config.max_questions * config.max_answers_per_sentence:
                break

        if not pairs:
            logger.warning("No (answer, sentence) pairs found; cannot generate questions.")
            extraction.questions = []
            return None

        inputs = [engine.build_input(answer, sentence) for answer, sentence, _ in pairs]
        questions = engine.generate(inputs)

        items: List[QuestionItem] = []
        for (answer, sentence, kw_score), question_text in zip(pairs, questions):
            question_text = self._normalize_question(question_text)
            if not question_text:
                continue

            confidence = self._confidence_from_score(kw_score, max_kw_score)
            items.append(
                QuestionItem(
                    question=question_text,
                    answer=answer,
                    source_document_id=extraction.document_id,
                    source_sentence=sentence,
                    confidence=confidence,
                    tags=["neural_qg"],
                )
            )

            if len(items) >= config.max_questions:
                break

        deduped_items = self._dedupe_items(items)
        limited_items = deduped_items[: config.max_questions]

        if len(limited_items) < config.min_questions:
            msg = (
                f"question_generator_low_yield: produced {len(limited_items)} "
                f"< min_questions={config.min_questions}"
            )
            logger.info(msg)
            extraction.errors.append(msg)

        extraction.questions = limited_items

        qlist = [
            Question(
                prompt=item.question,
                answer=item.answer,
                metadata={
                    "source_sentence": item.source_sentence,
                    "confidence": f"{item.confidence:.3f}",
                    "tags": ",".join(item.tags),
                    "source_document_id": item.source_document_id,
                },
            )
            for item in limited_items
        ]

        if not qlist:
            return None

        return QuestionSet(
            document=extraction.document,
            questions=qlist,
            strategy=config.strategy,
        )

    def _load_config(self) -> GeneratorConfig:
        def fetch(key: str, fallback: Any) -> Any:
            return get_setting(self.config_prefix, key, default=fallback)

        raw_strategy = fetch("strategy", self.DEFAULTS["strategy"])
        strategy = str(raw_strategy) if raw_strategy is not None else str(self.DEFAULTS["strategy"])

        raw_strategies = fetch("strategies", self.DEFAULTS["strategies"])
        strategies = self._normalize_strategies(raw_strategies)

        return GeneratorConfig(
            strategy=strategy,
            strategies=strategies,
            max_questions=self._coerce_int(fetch("max_questions", self.DEFAULTS["max_questions"]), 10),
            min_questions=self._coerce_int(fetch("min_questions", self.DEFAULTS["min_questions"]), 0),
            min_sentence_score=self._coerce_float(
                fetch("min_sentence_score", self.DEFAULTS["min_sentence_score"]), 0.45
            ),
            max_keywords_per_question=self._coerce_int(
                fetch("max_keywords_per_question", self.DEFAULTS["max_keywords_per_question"]), 2
            ),
            answer_max_chars=self._coerce_int(fetch("answer_max_chars", self.DEFAULTS["answer_max_chars"]), 280),
            confidence_decay=self._coerce_float(
                fetch("confidence_decay", self.DEFAULTS["confidence_decay"]), 0.1
            ),
            model_name=str(fetch("model_name", self.DEFAULTS["model_name"])),
            input_format=str(fetch("input_format", self.DEFAULTS["input_format"])),
            device=str(fetch("device", self.DEFAULTS["device"])),
            max_input_length=self._coerce_int(fetch("max_input_length", self.DEFAULTS["max_input_length"]), 256),
            max_output_length=self._coerce_int(fetch("max_output_length", self.DEFAULTS["max_output_length"]), 64),
            num_beams=self._coerce_int(fetch("num_beams", self.DEFAULTS["num_beams"]), 2),
            num_return_sequences=self._coerce_int(
                fetch("num_return_sequences", self.DEFAULTS["num_return_sequences"]), 1
            ),
            temperature=self._coerce_float(fetch("temperature", self.DEFAULTS["temperature"]), 1.0),
            top_p=self._coerce_float(fetch("top_p", self.DEFAULTS["top_p"]), 1.0),
            max_answers_per_sentence=self._coerce_int(
                fetch("max_answers_per_sentence", self.DEFAULTS["max_answers_per_sentence"]), 2
            ),
        )

    def _get_engine(self, config: GeneratorConfig) -> NeuralQGEngine:
        if self._engine is None:
            self._engine = NeuralQGEngine(
                model_name=config.model_name,
                input_format=config.input_format,
                device=config.device,
                max_input_length=config.max_input_length,
                max_output_length=config.max_output_length,
                num_beams=config.num_beams,
                num_return_sequences=config.num_return_sequences,
                temperature=config.temperature,
                top_p=config.top_p,
            )
        return self._engine

    def _assemble_ranked_sentences(
        self,
        top_sentences: List[str],
        sentence_scores: dict[str, float],
        *,
        fallback_text: str,
    ) -> List[Tuple[str, float]]:
        pairs: List[Tuple[str, float]] = []

        if top_sentences:
            for sent in top_sentences:
                score = sentence_scores.get(sent)
                if score is None:
                    stripped = sent.strip()
                    score = sentence_scores.get(stripped, 0.5)
                pairs.append((sent, float(score)))
        elif sentence_scores:
            pairs = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            naive_sents = self._naive_sentence_split(fallback_text)
            pairs = [(s, 0.5) for s in naive_sents]

        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    @staticmethod
    def _naive_sentence_split(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text or "")
        return [p.strip() for p in parts if len(p.strip()) > 5]

    @staticmethod
    def _normalize_question(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        if not cleaned.endswith("?"):
            cleaned = cleaned.rstrip(".") + "?"
        return cleaned

    def _get_keyword_candidates(
        self,
        *,
        extraction: ExtractionResult,
        max_items: int,
    ) -> Tuple[List[Tuple[str, float]], float]:
        if extraction.keyword_scores:
            sorted_items = sorted(
                extraction.keyword_scores.items(), key=lambda x: x[1], reverse=True
            )
            limited = sorted_items[:max_items]
            max_score = limited[0][1] if limited else 0.0
            return [(k, float(v)) for k, v in limited], float(max_score)

        if extraction.keywords:
            limited = extraction.keywords[:max_items]
            return [(k, 0.5) for k in limited], 0.5

        if extraction.tokens:
            limited = extraction.tokens[:max_items]
            return [(k, 0.4) for k in limited], 0.4

        return [], 0.0

    @staticmethod
    def _answers_for_sentence(
        *,
        sentence: str,
        keyword_candidates: List[Tuple[str, float]],
        max_answers: int,
        answer_max_chars: int,
    ) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        if not sentence:
            return out

        lowered = sentence.lower()
        for keyword, score in keyword_candidates:
            kw = keyword.strip()
            if not kw or len(kw) > answer_max_chars:
                continue
            if f" {kw.lower()} " in f" {lowered} ":
                out.append((kw, score))
            elif re.search(rf"\b{re.escape(kw.lower())}\b", lowered):
                out.append((kw, score))

            if len(out) >= max_answers:
                break

        return out

    @staticmethod
    def _confidence_from_score(score: float, max_score: float) -> float:
        if max_score <= 0:
            return 0.5
        norm = max(0.0, min(1.0, score / max_score))
        return 0.4 + 0.6 * norm

    @staticmethod
    def _dedupe_items(items: List[QuestionItem]) -> List[QuestionItem]:
        seen: set[tuple[str, str]] = set()
        out: List[QuestionItem] = []
        for item in items:
            key = (item.question.strip().lower(), item.answer.strip().lower())
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    def _normalize_strategies(self, raw: Any) -> List[str]:
        if raw is None:
            return list(self.DEFAULTS["strategies"])

        if isinstance(raw, str):
            candidates = [raw]
        elif isinstance(raw, IterableABC) and not isinstance(raw, (str, bytes)):
            candidates = [str(item) for item in raw]
        else:
            candidates = []

        cleaned = [candidate.strip() for candidate in candidates if candidate and candidate.strip()]
        return cleaned or list(self.DEFAULTS["strategies"])

    @staticmethod
    def _coerce_int(value: Any, fallback: int) -> int:
        try:
            if isinstance(value, bool):
                raise ValueError
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _coerce_float(value: Any, fallback: float) -> float:
        try:
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        except (TypeError, ValueError):
            return fallback


class RuleBasedQuestionGenerator:
    """Placeholder rule-based generator.

    This is a minimal stub so the existing import path resolves.
    Actual rule-based generation is intentionally deprecated and should be
    replaced by an external question generation implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(
        self,
        extraction: ExtractionResult,
        summary: Optional[str] = None,
    ) -> Optional[QuestionSet]:
        if extraction is not None:
            extraction.questions = []
        return None

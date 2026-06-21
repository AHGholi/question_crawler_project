# question_generator\adapter.py

from __future__ import annotations

from typing import Optional, Tuple

from .base import QGBackend, QGInput
from utils.models import ExtractionResult, Question, QuestionSet


class PipelineQuestionGenerator:
    QA_SEP = " || "

    def __init__(self, backend: QGBackend, default_num_questions: int = 10) -> None:
        self.backend = backend
        self.default_num_questions = default_num_questions

    @classmethod
    def _split_qa(cls, item: str) -> Tuple[str, Optional[str]]:
        s = (item or "").strip()
        if not s:
            return "", None
        if cls.QA_SEP in s:
            q, a = s.split(cls.QA_SEP, 1)
            q, a = q.strip(), a.strip()
            return q, (a or None)
        return s, None

    def __call__(
        self,
        extraction: ExtractionResult,
        summary: Optional[str] = None,
    ) -> Optional[QuestionSet]:
        topic = extraction.document.title or extraction.document.id
        sentences = extraction.top_sentences or []
        keywords = extraction.keywords or []

        data = QGInput(
            topic=topic,
            title=extraction.document.title,
            summary=summary or extraction.summary,
            keywords=keywords,
            sentences=sentences,
            num_questions=self.default_num_questions,
        )

        raw_items = self.backend.generate(data)

        questions = []
        for item in raw_items:
            q, a = self._split_qa(item)
            if q:
                questions.append(Question(prompt=q, answer=a))

        if not questions:
            return None

        cfg = getattr(self.backend, "cfg", None)
        if cfg is not None:
            model_name = getattr(cfg, "model_name", "unknown")
        else:
            model_name = getattr(self.backend, "model", "unknown")

        return QuestionSet(
            document=extraction.document,
            questions=questions,
            strategy=f"hf:{model_name}",
        )

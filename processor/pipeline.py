# processor/pipeline.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Protocol

from utils.models import DocumentRecord, ExtractionResult, QuestionSet


class Validator(Protocol):
    def __call__(self, document: DocumentRecord) -> None: ...


class ExtractorStep(Protocol):
    def __call__(self, document: DocumentRecord) -> ExtractionResult: ...


class RankingStep(Protocol):
    def __call__(self, extraction: ExtractionResult) -> ExtractionResult: ...


class SummarizerStep(Protocol):
    def __call__(self, extraction: ExtractionResult) -> str: ...


class QuestionGeneratorStep(Protocol):
    def __call__(self, extraction: ExtractionResult, summary: Optional[str] = None) -> Optional[QuestionSet]: ...


@dataclass(slots=True)
class PipelineContext:
    document: DocumentRecord
    extraction: Optional[ExtractionResult] = None
    summary: Optional[str] = None
    questions: Optional[QuestionSet] = None
    errors: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)


class ProcessorPipeline:
    def __init__(
        self,
        *,
        validators: Iterable[Validator] | None = None,
        extractor: ExtractorStep,
        ranking: RankingStep | None = None,
        summarizer: SummarizerStep | None = None,
        question_generator: QuestionGeneratorStep | None = None,
    ) -> None:
        self.validators = list(validators or [])
        self.extractor = extractor
        self.ranking = ranking
        self.summarizer = summarizer
        self.question_generator = question_generator

    def run(self, document: DocumentRecord) -> PipelineContext:
        ctx = PipelineContext(document=document)

        try:
            for validator in self.validators:
                validator(document)
        except Exception as exc:
            ctx.add_error(f"validation_failed: {exc}")
            return ctx

        try:
            extraction = self.extractor(document)
            ctx.extraction = extraction
        except Exception as exc:
            ctx.add_error(f"extraction_failed: {exc}")
            return ctx

        if self.ranking and ctx.extraction:
            try:
                ctx.extraction = self.ranking(ctx.extraction)
            except Exception as exc:
                ctx.add_error(f"ranking_failed: {exc}")
                return ctx

        if self.summarizer and ctx.extraction:
            try:
                ctx.summary = self.summarizer(ctx.extraction)
            except Exception as exc:
                ctx.add_error(f"summarization_failed: {exc}")

        if self.question_generator and ctx.extraction:
            try:
                qset = self.question_generator(ctx.extraction, ctx.summary)
                if qset and qset.questions:
                    ctx.questions = qset
                else:
                    ctx.questions = None
            except Exception as exc:
                ctx.add_error(f"question_generation_failed: {exc}")


        return ctx


def build_default_pipeline(
    *,
    validators: Iterable[Validator] | None,
    extractor: ExtractorStep,
    ranking: RankingStep | None = None,
    summarizer: SummarizerStep | None,
    question_generator: QuestionGeneratorStep | None,
) -> ProcessorPipeline:
    return ProcessorPipeline(
        validators=validators,
        extractor=extractor,
        ranking=ranking,
        summarizer=summarizer,
        question_generator=question_generator,
    )


__all__ = ["PipelineContext", "ProcessorPipeline", "build_default_pipeline"]

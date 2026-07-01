# main_pipeline.py

"""Orchestrates the high-level crawl → process → report workflow.

The main pipeline coordinates document discovery with the processor pipeline and
collects basic statistics about successfully processed documents and failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Protocol

from processor.pipeline import PipelineContext, ProcessorPipeline
from utils.models import DocumentRecord

from dotenv import load_dotenv
load_dotenv()


class CrawlStep(Protocol):
    """Callable that yields document records ready for processing."""

    def __call__(self) -> Iterable[DocumentRecord]: ...


@dataclass(slots=True)
class ProcessingStats:
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_questions: int = 0

    def observe_context(self, context: PipelineContext) -> None:
        """Update counters based on the outcome of one processed document."""
        # Count every attempted document, then classify it as processed or failed.
        self.total_documents += 1
        if context.errors:
            self.failed_documents += 1
            return
        self.processed_documents += 1
        if context.questions:
            self.total_questions += len(context.questions.questions)


@dataclass(slots=True)
class MainPipelineResult:
    contexts: List[PipelineContext] = field(default_factory=list)
    stats: ProcessingStats = field(default_factory=ProcessingStats)

    @property
    def has_failures(self) -> bool:
        """Return whether any processed context recorded an error."""
        return any(ctx.errors for ctx in self.contexts)


class MainPipeline:
    """
    Full pipeline orchestrator: crawl → process → (optional) notify.
    """

    def __init__(
        self,
        *,
        crawler: CrawlStep,
        processor: ProcessorPipeline,
        post_process_hook: Callable[[PipelineContext], None] | None = None,
    ) -> None:
        # Store the two core collaborators that drive the pipeline execution.
        self.crawler = crawler
        self.processor = processor
        self.post_process_hook = post_process_hook

    def run(self, *, limit: Optional[int] = None) -> MainPipelineResult:
        """Execute the pipeline for each discovered document until the limit is reached."""
        result = MainPipelineResult()
        for idx, document in enumerate(self.crawler()):
            # Stop once the caller-defined processing cap has been reached.
            if limit is not None and idx >= limit:
                break

            context = self.processor.run(document)
            result.contexts.append(context)
            result.stats.observe_context(context)

            if self.post_process_hook:
                try:
                    self.post_process_hook(context)
                except Exception as exc:
                    context.add_error(f"post_processing_failed: {exc}")

        return result

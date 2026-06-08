from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator, Optional
from uuid import uuid4

from crawler.search import run_search
from extractor.html_extractor import HTMLExtractor
from main_pipeline import MainPipeline, MainPipelineResult, PipelineContext
from processor.pipeline import ProcessorPipeline
from processor.sentence_ranking import SentenceRankingService
from question_generator.generator import NeuralQuestionGenerator
from utils.models import DocumentRecord, ExtractionResult


def crawl_documents(query: str, max_results: int) -> Iterator[DocumentRecord]:
    """
    Adapter that turns `crawler.search.run_search` payloads into `DocumentRecord` instances.
    Yields only successfully downloaded HTML pages.
    """
    logging.info("Running search pipeline for query=%r (max_results=%s)", query, max_results)
    search_results = run_search(query, max_results=max_results)

    for idx, payload in enumerate(search_results, start=1):
        download = payload.get("download") or {}
        status = download.get("status", "unknown")

        if status != "ok":
            logging.warning(
                "Skipping search result %s; download status=%s reason=%s",
                idx,
                status,
                download.get("reason"),
            )
            continue

        path_value = download.get("path")
        if not path_value:
            logging.warning("Skipping search result %s; missing download path.", idx)
            continue

        path = Path(path_value)
        if not path.exists():
            logging.warning("Skipping search result %s; file not found at %s", idx, path)
            continue

        metadata: dict[str, str] = {}
        for key, value in (
            ("snippet", payload.get("snippet")),
            ("search_link", payload.get("search_link")),
            ("download_url", download.get("url")),
            ("rank", str(idx)),
        ):
            if value:
                metadata[key] = str(value)

        document = DocumentRecord(
            id=uuid4().hex,
            title=payload.get("title"),
            metadata=metadata,
            media_type="text/html",
            path=path,
            source_path=path,
            encoding="utf-8",
        )

        logging.debug(
            "Yielding DocumentRecord(id=%s, title=%r, path=%s)",
            document.id,
            document.title,
            document.path,
        )
        yield document


def build_crawler(query: str, max_results: int):
    """
    Returns a CrawlStep-compatible callable that produces DocumentRecord instances.
    """

    def _crawler() -> Iterator[DocumentRecord]:
        yield from crawl_documents(query, max_results=max_results)

    return _crawler


def build_processor_pipeline() -> ProcessorPipeline:
    """
    Assemble the ProcessorPipeline with extraction, ranking, and question generation.
    """
    html_extractor = HTMLExtractor()
    ranking_service = SentenceRankingService()
    question_generator = NeuralQuestionGenerator()

    def extractor_step(document: DocumentRecord) -> ExtractionResult:
        if not html_extractor.supports(document):
            raise ValueError(f"Unsupported document for HTML extraction: {document.id}")
        return html_extractor.extract(document)

    return ProcessorPipeline(
        validators=None,
        extractor=extractor_step,
        ranking=ranking_service.rank,
        summarizer=None,
        question_generator=question_generator,
    )

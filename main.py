# main.py

"""Entry-point module for running the exam-crawler pipeline.

This script wires together the search crawler, document extraction, text
processing, and question generation stages so the project can be run either
from the command line or from the UI.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterator, Optional
from uuid import uuid4

from dotenv import load_dotenv

from crawler.search import run_search
from extractor.html_extractor import HTMLExtractor
from extractor.keyword_extractor import KeywordExtractor
from extractor.text_extractor import TextExtractor

from main_pipeline import MainPipeline, MainPipelineResult, PipelineContext
from processor.pipeline import ProcessorPipeline
from processor.sentence_ranking import SentenceRankingService
from question_generator.adapter import PipelineQuestionGenerator
from question_generator.hf_qg import HFQuestionGenerator
from utils.config import get_setting
from utils.models import DocumentRecord, ExtractionResult

from question_generator.local_hf_qg import LocalHFQuestionGenerator, LocalHFQGConfig


# Configure the root logger so the pipeline emits consistent startup and runtime messages.
logging.basicConfig(level=logging.INFO)

# Load environment variables once at startup so configuration helpers can resolve them.
load_dotenv()


def crawl_documents(query: str, max_results: int) -> Iterator[DocumentRecord]:
    """Convert search results into document records for downstream processing.

    The crawler returns raw search payloads with download metadata. This adapter
    filters out failed downloads and turns the remaining items into the
    standardized DocumentRecord objects expected by the processing pipeline.
    """
    logging.info("Running search pipeline for query=%r (max_results=%s)", query, max_results)
    search_results = run_search(query, max_results=max_results)

    for idx, payload in enumerate(search_results, start=1):
        # Each search result may contain a download payload; only successful downloads become documents.
        download = payload.get("download") or {}
        status = download.get("status", "unknown")

        if status != "ok":
            # Skip search hits that failed to download or were blocked by the crawler.
            logging.warning(
                "Skipping search result %s; download status=%s reason=%s",
                idx,
                status,
                download.get("reason"),
            )
            continue

        path_value = download.get("path")
        if not path_value:
            # Some successful-looking payloads still do not contain a usable local file path.
            logging.warning("Skipping search result %s; missing download path.", idx)
            continue

        path = Path(path_value)
        if not path.exists():
            # The search result may have been recorded, but the file no longer exists locally.
            logging.warning("Skipping search result %s; file not found at %s", idx, path)
            continue

        metadata: dict[str, str] = {}
        # Carry over the most useful metadata so downstream stages have context about the source.
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
    """Create a crawler callable that yields documents from a web search query."""

    def _crawler() -> Iterator[DocumentRecord]:
        # Reuse the search adapter so callers can treat the crawler as a simple iterator.
        yield from crawl_documents(query, max_results=max_results)

    return _crawler

def build_question_generator() -> Optional[PipelineQuestionGenerator]:
    """Create the configured question generator backend for the pipeline.

    The selected provider can be a local Hugging Face model or the remote
    Hugging Face API, depending on configuration.
    """
    # Read the runtime options from configuration so the same code path works with
    # different backends and model settings without extra branching in the caller.
    qg_enabled = get_setting("question_generator", "enabled", default=True)
    if not qg_enabled:
        # The pipeline can be run without question generation when the feature is disabled.
        logging.info("Question generation disabled via config.")
        return None

    provider = str(get_setting("question_generator", "provider", default="local_hf")).lower().strip()
    num_questions = int(get_setting("question_generator", "num_questions", default=10))
    model = str(get_setting("question_generator", "model", default="google/flan-t5-small"))
    temperature = float(get_setting("question_generator", "temperature", default=0.7))
    max_new_tokens = int(get_setting("question_generator", "max_new_tokens", default=320))
    min_new_tokens = int(get_setting("question_generator", "min_new_tokens", default=12))
    do_sample = bool(get_setting("question_generator", "do_sample", default=True))
    top_p = float(get_setting("question_generator", "top_p", default=0.9))
    num_beams = int(get_setting("question_generator", "num_beams", default=4))
    device = int(get_setting("question_generator", "device", default=-1))

    if provider == "local_hf":
        # Use the locally installed transformer when the configuration selects the local backend.
        cfg = LocalHFQGConfig(
            model_name=model,
            device=device,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
        )
        backend = LocalHFQuestionGenerator(config=cfg)
        return PipelineQuestionGenerator(backend=backend, default_num_questions=num_questions)

    if provider == "huggingface":
        # The remote backend needs a token from the environment before the API call can be made.
        token_env = get_setting("question_generator", "hf_api_token_env", default="HF_API_TOKEN")
        hf_token = os.getenv(token_env)
        if not hf_token:
            # Fail early with a clear message if the required authentication is missing.
            raise RuntimeError(f"Missing Hugging Face token. Set env var {token_env} in your .env file.")

        timeout = int(get_setting("question_generator", "timeout", default=60))
        backend = HFQuestionGenerator(
            api_token=hf_token,
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return PipelineQuestionGenerator(backend=backend, default_num_questions=num_questions)

    raise ValueError(f"Unsupported question generator provider: {provider}")


def build_processor_pipeline() -> ProcessorPipeline:
    """Assemble the processing pipeline with extraction, ranking, and optional question generation."""
    # Compose the concrete processing stages that will be executed for each document.
    html_extractor = HTMLExtractor()
    text_extractor = TextExtractor()
    ranking_service = SentenceRankingService()
    keyword_extractor = KeywordExtractor()
    qg_adapter = build_question_generator()

    def extractor_step(document: DocumentRecord) -> ExtractionResult:
        # Use the first extractor that reports support for the current document type.
        for extractor in (html_extractor, text_extractor):
            if extractor.supports(document):
                # Once a suitable extractor is found, stop and return its extraction result.
                return extractor.extract(document)
        raise ValueError(
            f"Unsupported document for extraction: {document.id} media_type={document.media_type} path={document.path}"
        )

    def ranking_step(extraction: ExtractionResult) -> ExtractionResult:
        # Extract keywords and sentence-level evidence before ranking the passage.
        text = extraction.clean_text or extraction.raw_text or ""
        # The keyword extractor produces the tokens and scores used by the ranking stage.
        kr = keyword_extractor.run(text, title=extraction.document.title or "")
        extraction.tokens = kr.tokens
        extraction.keywords = kr.keywords
        extraction.keyword_scores = kr.scores
        return ranking_service.rank(extraction)

    # Wrapper to satisfy ProcessorPipeline QuestionGeneratorStep protocol:
    # expected signature: (extraction, summary=None) -> QuestionSet | None
    def question_generator_step(extraction: ExtractionResult, summary: str | None = None):
        # Question generation is optional and should be skipped cleanly when disabled.
        if qg_adapter is None:
            # Returning None keeps the processor pipeline contract intact while disabling the feature.
            return None
        if summary and not extraction.summary:
            # Reuse a supplied summary when the extraction already has no summary of its own.
            extraction.summary = summary
        return qg_adapter(extraction)

    return ProcessorPipeline(
        validators=None,
        extractor=extractor_step,
        ranking=ranking_step,
        summarizer=None,
        question_generator=question_generator_step if qg_adapter is not None else None,
    )


def build_file_crawler(file_paths: list[str]):
    """Create a crawler that processes locally provided files instead of web search results."""

    def _crawler() -> Iterator[DocumentRecord]:
        # Treat each supplied path as a document candidate and skip missing or unsupported files.
        for path_str in file_paths:
            # Each provided path is treated as a separate input document candidate.
            path = Path(path_str)
            if not path.exists():
                # Missing files are skipped so the run can continue with the remaining inputs.
                logging.warning("Skipping missing input file: %s", path)
                continue

            title = path.stem
            suffix = path.suffix.lower()
            if suffix in {".html", ".htm"}:
                # HTML inputs are treated as web-like content that can be parsed by the extractor.
                media_type = "text/html"
            elif suffix == ".txt":
                # Plain text files are passed through the textual extractor path.
                media_type = "text/plain"
            elif suffix == ".docx":
                # Word documents use the Office Open XML media type.
                media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif suffix == ".doc":
                # Legacy Word documents use the older binary document media type.
                media_type = "application/msword"
            else:
                # Unsupported extensions are still accepted as opaque documents but will be handled conservatively.
                media_type = "application/octet-stream"
            yield DocumentRecord(
                id=uuid4().hex,
                title=title,
                metadata={"source_file": str(path)},
                media_type=media_type,
                path=path,
                source_path=path,
                encoding="utf-8",
            )

    return _crawler


def run_main_pipeline(
    query: str | None = None,
    max_results: int = 5,
    limit: Optional[int] = None,
    input_files: list[str] | None = None,
) -> MainPipelineResult:
    """Run the full pipeline using either a search query or a list of local files."""
    # Select the appropriate source of documents before constructing the orchestration layer.
    if input_files:
        # Local files are processed directly when the caller supplies them.
        crawler = build_file_crawler(input_files)
    elif query is not None:
        # A web search query is used when no explicit files are provided.
        crawler = build_crawler(query, max_results=max_results)
    else:
        # This branch protects the public entry point from being called without any source input.
        raise ValueError("Either a search query or input_files must be provided.")

    pipeline = MainPipeline(crawler=crawler, processor=build_processor_pipeline())
    return pipeline.run(limit=limit)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exam crawler pipeline")
    parser.add_argument("--query", help="Search query", default=None)
    parser.add_argument("--input-files", nargs="+", help="Local text or HTML files to process")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum search results")
    parser.add_argument("--limit", type=int, default=0, help="Document processing limit (0 = no limit)")
    return parser.parse_args()


if __name__ == "__main__":
    # Reconfigure logging for direct script execution so CLI output is easier to read.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_cli_args()

    if not args.query and not args.input_files:
        # The CLI needs one of the two supported input modes to run.
        raise SystemExit("Either --query or --input-files must be provided.")

    if args.query and args.input_files:
        # The command line parser should not accept both modes at once.
        raise SystemExit("Provide only one of --query or --input-files.")

    limit_arg = None if args.limit == 0 else args.limit
    result = run_main_pipeline(
        query=args.query,
        max_results=args.max_results,
        limit=limit_arg,
        input_files=args.input_files,
    )

    print("Processed documents:", result.stats.processed_documents)
    print("Failed documents:", result.stats.failed_documents)
    print("Total questions:", result.stats.total_questions)
    if result.stats.failed_documents:
        # Report the per-document failures after the summary output so the user can inspect them.
        for ctx in result.contexts:
            if ctx.errors:
                print(f"- {ctx.document.id}: {ctx.errors}")

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
from extractor.pdf_extractor import PDFExtractor
from extractor.doc_extractor import DOCDocxExtractor

from main_pipeline import MainPipeline, MainPipelineResult
from processor.pipeline import ProcessorPipeline
from processor.sentence_ranking import SentenceRankingService
from question_generator.adapter import PipelineQuestionGenerator
from question_generator.hf_qg import HFQuestionGenerator, HFQGConfig
from utils.config import get_setting
from utils.models import DocumentRecord, ExtractionResult

from question_generator.local_hf_qg import LocalHFQuestionGenerator, LocalHFQGConfig


# Configure the root logger so the pipeline emits consistent startup and runtime messages.
logging.basicConfig(level=logging.INFO)

# Load environment variables once at startup so configuration helpers can resolve them.
load_dotenv()


def crawl_documents(query: str, max_results: int) -> Iterator[DocumentRecord]:
    """Convert search results into document records for downstream processing."""
    logging.info("Running search pipeline for query=%r (target_successes=%s)", query, max_results)

    search_results = run_search(query, max_results=max_results * 3)
    yielded = 0

    for idx, payload in enumerate(search_results, start=1):
        if yielded >= max_results:
            break

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

        content_type = download.get("content_type")
        if not content_type:
            logging.warning("Skipping search result %s; missing content_type.", idx)
            continue

        metadata: dict[str, str] = {}
        for key, value in (
            ("snippet", payload.get("snippet")),
            ("search_link", payload.get("search_link")),
            ("download_url", download.get("url")),
            ("final_url", download.get("final_url")),
            ("rank", str(idx)),
        ):
            if value:
                metadata[key] = str(value)

        document = DocumentRecord(
            id=uuid4().hex,
            title=payload.get("title"),
            metadata=metadata,
            media_type=str(content_type),
            path=path,
            source_path=path,
            encoding="utf-8",
        )

        yielded += 1
        logging.debug(
            "Yielding DocumentRecord(id=%s, title=%r, path=%s, media_type=%s)",
            document.id,
            document.title,
            document.path,
            document.media_type,
        )
        yield document


def build_crawler(query: str, max_results: int):
    """Create a crawler callable that yields documents from a web search query."""

    def _crawler() -> Iterator[DocumentRecord]:
        # Reuse the search adapter so callers can treat the crawler as a simple iterator.
        yield from crawl_documents(query, max_results=max_results)

    return _crawler

def build_question_generator(
    provider: str | None = None,
    model: str | None = None,
    hf_api_token: str | None = None,
    num_questions: int | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    min_new_tokens: int | None = None,
    do_sample: bool | None = None,
    top_p: float | None = None,
    num_beams: int | None = None,
    device: int | None = None,
    timeout: int | None = None,
) -> Optional[PipelineQuestionGenerator]:
    """Create the configured question generator backend for the pipeline.

    Runtime arguments take precedence over config values, so the UI/CLI can
    choose the backend dynamically.
    """
    qg_enabled = get_setting("question_generator", "enabled", default=True)
    if not qg_enabled:
        logging.info("Question generation disabled via config.")
        return None

    provider = (provider or get_setting("question_generator", "provider", default="local_hf")).lower().strip()
    num_questions = int(num_questions if num_questions is not None else get_setting("question_generator", "num_questions", default=10))
    model = str(model or get_setting("question_generator", "model", default="google/flan-t5-large"))
    temperature = float(temperature if temperature is not None else get_setting("question_generator", "temperature", default=0.7))
    max_new_tokens = int(max_new_tokens if max_new_tokens is not None else get_setting("question_generator", "max_new_tokens", default=320))
    min_new_tokens = int(min_new_tokens if min_new_tokens is not None else get_setting("question_generator", "min_new_tokens", default=12))
    do_sample = bool(do_sample if do_sample is not None else get_setting("question_generator", "do_sample", default=True))
    top_p = float(top_p if top_p is not None else get_setting("question_generator", "top_p", default=0.9))
    num_beams = int(num_beams if num_beams is not None else get_setting("question_generator", "num_beams", default=4))
    device = int(device if device is not None else get_setting("question_generator", "device", default=-1))
    timeout = int(timeout if timeout is not None else get_setting("question_generator", "timeout", default=60))

    if provider == "local_hf":
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

    if provider in {"huggingface", "hf_api", "api"}:
        token_env = get_setting("question_generator", "hf_api_token_env", default="HF_API_TOKEN")
        resolved_token = (hf_api_token or "").strip() or os.getenv(token_env)

        if not resolved_token:
            raise RuntimeError(
                f"Missing Hugging Face token. Provide it in the UI/CLI or set env var {token_env} in your .env file."
            )

        backend = HFQuestionGenerator(
            config=HFQGConfig(
                api_token=resolved_token,
                model_name=model,
                timeout=timeout or 60,
                max_new_tokens=max_new_tokens or 96,
            )
        )

        return PipelineQuestionGenerator(backend=backend, default_num_questions=num_questions)

    raise ValueError(f"Unsupported question generator provider: {provider}")


def build_processor_pipeline(
    qg_provider: str | None = None,
    qg_model: str | None = None,
    qg_api_token: str | None = None,
    qg_num_questions: int | None = None,
    qg_temperature: float | None = None,
    qg_max_new_tokens: int | None = None,
    qg_min_new_tokens: int | None = None,
    qg_do_sample: bool | None = None,
    qg_top_p: float | None = None,
    qg_num_beams: int | None = None,
    qg_device: int | None = None,
    qg_timeout: int | None = None,
) -> ProcessorPipeline:
    """Assemble the processing pipeline with extraction, ranking, and optional question generation."""
    html_extractor = HTMLExtractor()
    text_extractor = TextExtractor()
    pdf_extractor = PDFExtractor()
    doc_extractor = DOCDocxExtractor()

    ranking_service = SentenceRankingService()
    keyword_extractor = KeywordExtractor()
    qg_adapter = build_question_generator(
        provider=qg_provider,
        model=qg_model,
        hf_api_token=qg_api_token,
        num_questions=qg_num_questions,
        temperature=qg_temperature,
        max_new_tokens=qg_max_new_tokens,
        min_new_tokens=qg_min_new_tokens,
        do_sample=qg_do_sample,
        top_p=qg_top_p,
        num_beams=qg_num_beams,
        device=qg_device,
        timeout=qg_timeout,
    )

    def extractor_step(document: DocumentRecord) -> ExtractionResult:
        media_type = (document.media_type or "").strip().lower()

        if media_type in {"text/html", "application/xhtml+xml"}:
            return html_extractor.extract(document)

        if media_type == "text/plain":
            return text_extractor.extract(document)

        if media_type == "application/pdf":
            return pdf_extractor.extract(document)

        if media_type in {
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/doc",
            "application/x-msword",
            "application/vnd.ms-word",
            "application/x-docx",
        }:
            return doc_extractor.extract(document)

        suffix = document.path.suffix.lower() if document.path else ""

        if suffix in {".html", ".htm"}:
            return html_extractor.extract(document)

        if suffix == ".txt":
            return text_extractor.extract(document)

        if suffix == ".pdf":
            return pdf_extractor.extract(document)

        if suffix in {".doc", ".docx"}:
            return doc_extractor.extract(document)

        raise ValueError(
            f"Unsupported document for extraction: {document.id} "
            f"media_type={document.media_type} path={document.path}"
        )

    def ranking_step(extraction: ExtractionResult) -> ExtractionResult:
        text = extraction.clean_text or extraction.raw_text or ""
        kr = keyword_extractor.run(text, title=extraction.document.title or "")
        extraction.tokens = kr.tokens
        extraction.keywords = kr.keywords
        extraction.keyword_scores = kr.scores
        return ranking_service.rank(extraction)

    def question_generator_step(extraction: ExtractionResult, summary: str | None = None):
        if qg_adapter is None:
            return None
        if summary and not extraction.summary:
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
                media_type = "text/html"
            elif suffix == ".txt":
                media_type = "text/plain"
            elif suffix == ".pdf":
                media_type = "application/pdf"
            elif suffix == ".docx":
                media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif suffix == ".doc":
                media_type = "application/msword"
            else:
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
    qg_provider: str | None = None,
    qg_model: str | None = None,
    qg_api_token: str | None = None,
    qg_num_questions: int | None = None,
    qg_temperature: float | None = None,
    qg_max_new_tokens: int | None = None,
    qg_min_new_tokens: int | None = None,
    qg_do_sample: bool | None = None,
    qg_top_p: float | None = None,
    qg_num_beams: int | None = None,
    qg_device: int | None = None,
    qg_timeout: int | None = None,
) -> MainPipelineResult:
    """Run the full pipeline using either a search query or a list of local files."""
    if input_files:
        crawler = build_file_crawler(input_files)
    elif query is not None:
        crawler = build_crawler(query, max_results=max_results)
    else:
        raise ValueError("Either a search query or input_files must be provided.")

    pipeline = MainPipeline(
        crawler=crawler,
        processor=build_processor_pipeline(
            qg_provider=qg_provider,
            qg_model=qg_model,
            qg_api_token=qg_api_token,
            qg_num_questions=qg_num_questions,
            qg_temperature=qg_temperature,
            qg_max_new_tokens=qg_max_new_tokens,
            qg_min_new_tokens=qg_min_new_tokens,
            qg_do_sample=qg_do_sample,
            qg_top_p=qg_top_p,
            qg_num_beams=qg_num_beams,
            qg_device=qg_device,
            qg_timeout=qg_timeout,
        ),
    )
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

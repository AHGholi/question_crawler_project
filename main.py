# main.py

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


logging.basicConfig(level=logging.INFO)

# Load environment variables once at startup
load_dotenv()


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

def build_question_generator() -> Optional[PipelineQuestionGenerator]:
    qg_enabled = get_setting("question_generator", "enabled", default=True)
    if not qg_enabled:
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
        token_env = get_setting("question_generator", "hf_api_token_env", default="HF_API_TOKEN")
        hf_token = os.getenv(token_env)
        if not hf_token:
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
    """
    Assemble the ProcessorPipeline with extraction, keyword extraction, ranking, and question generation.
    """
    html_extractor = HTMLExtractor()
    text_extractor = TextExtractor()
    ranking_service = SentenceRankingService()
    keyword_extractor = KeywordExtractor()
    qg_adapter = build_question_generator()

    def extractor_step(document: DocumentRecord) -> ExtractionResult:
        for extractor in (html_extractor, text_extractor):
            if extractor.supports(document):
                return extractor.extract(document)
        raise ValueError(
            f"Unsupported document for extraction: {document.id} media_type={document.media_type} path={document.path}"
        )

    def ranking_step(extraction: ExtractionResult) -> ExtractionResult:
        text = extraction.clean_text or extraction.raw_text or ""
        kr = keyword_extractor.run(text, title=extraction.document.title or "")
        extraction.tokens = kr.tokens
        extraction.keywords = kr.keywords
        extraction.keyword_scores = kr.scores
        return ranking_service.rank(extraction)

    # Wrapper to satisfy ProcessorPipeline QuestionGeneratorStep protocol:
    # expected signature: (extraction, summary=None) -> QuestionSet | None
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
    def _crawler() -> Iterator[DocumentRecord]:
        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists():
                logging.warning("Skipping missing input file: %s", path)
                continue

            title = path.stem
            suffix = path.suffix.lower()
            if suffix in {".html", ".htm"}:
                media_type = "text/html"
            elif suffix == ".txt":
                media_type = "text/plain"
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
) -> MainPipelineResult:
    if input_files:
        crawler = build_file_crawler(input_files)
    elif query is not None:
        crawler = build_crawler(query, max_results=max_results)
    else:
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_cli_args()

    if not args.query and not args.input_files:
        raise SystemExit("Either --query or --input-files must be provided.")

    if args.query and args.input_files:
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
        for ctx in result.contexts:
            if ctx.errors:
                print(f"- {ctx.document.id}: {ctx.errors}")

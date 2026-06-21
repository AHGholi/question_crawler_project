# tests/integration/test_downloaded_files_local_qg.py
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from extractor.html_extractor import HTMLExtractor
from extractor.keyword_extractor import KeywordExtractor
from processor.pipeline import ProcessorPipeline
from processor.sentence_ranking import SentenceRankingService
from question_generator.adapter import PipelineQuestionGenerator
from question_generator.local_hf_qg import LocalHFQuestionGenerator
from utils.models import DocumentRecord, ExtractionResult


def _build_local_processor() -> ProcessorPipeline:
    html_extractor = HTMLExtractor()
    ranking_service = SentenceRankingService()
    keyword_extractor = KeywordExtractor(
        min_top_k=20,
        max_terms=60,            # was 200
        threshold_fraction=0.75, # was 0.4/0.6 style
        bigram_boost=1.8,
        title_boost=0.35,
    )

    backend = LocalHFQuestionGenerator(
        model_name="google/flan-t5-large",
        device=-1,
        max_new_tokens=160,
        do_sample=False,
        temperature=0.0,
    )
    qg = PipelineQuestionGenerator(backend=backend, default_num_questions=5)

    def extractor_step(document: DocumentRecord) -> ExtractionResult:
        if not html_extractor.supports(document):
            raise ValueError(f"Unsupported document for HTML extraction: {document.id}")
        return html_extractor.extract(document)

    def ranking_step(extraction: ExtractionResult) -> ExtractionResult:
        # 1) keyword stage
        kr = keyword_extractor.run(
            extraction.clean_text or extraction.raw_text or "",
            title=extraction.document.title or "",
        )
        extraction.tokens = kr.tokens
        extraction.keywords = kr.keywords
        extraction.keyword_scores = kr.scores

        # 2) sentence ranking stage (now can use keywords)
        return ranking_service.rank(extraction)

    return ProcessorPipeline(
        validators=None,
        extractor=extractor_step,
        ranking=ranking_step,
        summarizer=None,
        question_generator=qg,
    )


@pytest.mark.integration
def test_downloaded_files_local_qg() -> None:
    folder = Path("downloaded_files")
    assert folder.exists(), "downloaded_files folder not found"

    html_files = sorted(folder.glob("*.html"))
    assert html_files, "No HTML files found in downloaded_files"

    processor = _build_local_processor()

    total_questions = 0
    for path in html_files[:3]:  # keep test short
        doc = DocumentRecord(
            id=uuid4().hex,
            title=path.stem,
            metadata={"source": str(path)},
            media_type="text/html",
            path=path,
            source_path=path,
            encoding="utf-8",
        )

        ctx = processor.run(doc)

        

        assert ctx is not None
        assert not ctx.errors, f"{path.name} failed with errors: {ctx.errors}"
        assert ctx.extraction is not None, f"{path.name}: extraction missing"
        assert len((ctx.extraction.clean_text or "").strip()) > 0, (
            f"{path.name}: extractor produced empty clean_text"
        )

        qset = ctx.questions
        questions = qset.questions if qset else []

        ext = ctx.extraction
        print("clean_text_len:", len((ext.clean_text if ext else "") or ""))
        print("summary:", ext.summary if ext else None)
        print("top_sentences:", len((ext.top_sentences if ext else []) or []))
        print("keywords:", len((ext.keywords if ext else []) or []))
        print("errors:", ctx.errors)

        print(f"\n=== {path.name} ===")
        for i, q in enumerate(questions, 1):
            # q might be an object with .prompt OR a plain string depending on your pipeline
            prompt = getattr(q, "prompt", str(q))
            print(f"{i}. {prompt}")

        total_questions += len(questions)

    assert total_questions > 0, "No questions were generated"

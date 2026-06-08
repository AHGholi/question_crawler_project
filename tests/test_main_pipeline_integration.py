from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Iterable, List

import pytest

from main import build_processor_pipeline
from main_pipeline import MainPipeline
from utils.models import DocumentRecord

PHASE1_FIXTURE_PATH = Path("tests/fixtures/phase1/sample_documents.json")


def _instantiate_document(record: dict) -> DocumentRecord:
    """
    Instantiate a DocumentRecord from raw dict data, supporting:
    - dataclass-based DocumentRecord
    - pydantic-style model_validate
    - custom from_dict helpers
    """
    if hasattr(DocumentRecord, "model_validate"):
        return DocumentRecord.model_validate(record)  # type: ignore[attr-defined]
    if hasattr(DocumentRecord, "from_dict"):
        return DocumentRecord.from_dict(record)  # type: ignore[attr-defined]

    annotations = getattr(DocumentRecord, "__annotations__", {})
    constructor_fields = list(annotations.keys())

    if constructor_fields:
        filtered = {field: record[field] for field in constructor_fields if field in record}
        return DocumentRecord(**filtered)  # type: ignore[arg-type]

    signature = inspect.signature(DocumentRecord)
    kwargs = {}
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if name in record:
            kwargs[name] = record[name]
        elif param.default is inspect._empty:
            raise KeyError(f"Fixture missing required field '{name}' for DocumentRecord.")
    return DocumentRecord(**kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def phase1_documents() -> List[DocumentRecord]:
    if not PHASE1_FIXTURE_PATH.exists():
        pytest.skip(
            f"Phase-1 fixture missing at {PHASE1_FIXTURE_PATH}. "
            "Run the crawler and export DocumentRecord JSON to enable this integration test."
        )

    raw_payload = json.loads(PHASE1_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list) or not raw_payload:
        pytest.skip(f"No documents available in {PHASE1_FIXTURE_PATH}.")

    return [_instantiate_document(entry) for entry in raw_payload]


@pytest.fixture(scope="module")
def main_pipeline(phase1_documents: List[DocumentRecord]) -> MainPipeline:
    processor = build_processor_pipeline()

    def crawler_stub() -> Iterable[DocumentRecord]:
        return iter(phase1_documents)

    return MainPipeline(
        crawler=crawler_stub,
        processor=processor,
        post_process_hook=None,
    )


def test_main_pipeline_end_to_end(main_pipeline: MainPipeline, phase1_documents: List[DocumentRecord]) -> None:
    result = main_pipeline.run(limit=len(phase1_documents))

    assert result.stats.total_documents == len(phase1_documents), "Not all documents were processed."
    assert result.stats.processed_documents > 0, "No documents processed successfully."
    assert not result.has_failures, f"Pipeline reported failures: {[ctx.errors for ctx in result.contexts if ctx.errors]}"

    for ctx in result.contexts:
        doc_id = getattr(ctx.document, "id", "<unknown>")
        assert ctx.extraction is not None, f"{doc_id} missing extraction result."
        assert ctx.extraction.top_sentences, f"{doc_id} missing top sentences."
        assert isinstance(ctx.extraction.sentence_scores, dict), f"{doc_id} sentence_scores is not a dict."
        assert ctx.extraction.sentence_scores, f"{doc_id} missing sentence scores."

        if ctx.questions:
            assert ctx.questions.questions, f"{doc_id} produced an empty QuestionSet."
            for question in ctx.questions.questions:
                confidence = question.metadata.get("confidence")
                assert isinstance(confidence, str), f"{doc_id} question confidence must be a string."
                whole, dot, fractional = confidence.partition(".")
                assert dot, f"{doc_id} confidence must contain a decimal point (got '{confidence}')."
                assert len(fractional) == 3, f"{doc_id} confidence must have three decimal places (got '{confidence}')."
                assert whole.isdigit() or (whole.startswith("-") and whole[1:].isdigit()), (
                    f"{doc_id} confidence must represent a float (got '{confidence}')."
                )

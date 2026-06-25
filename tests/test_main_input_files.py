# tests/test_main_input_files.py
from __future__ import annotations

from pathlib import Path

import pytest

import main
from main_pipeline import PipelineContext
from utils.models import DocumentRecord


class DummyProcessor:
    def __init__(self) -> None:
        self.ran = 0

    def run(self, document: DocumentRecord) -> PipelineContext:
        self.ran += 1
        return PipelineContext(document=document)


def test_build_file_crawler_yields_text_record(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Hello world. This is test text.", encoding="utf-8")

    crawler = main.build_file_crawler([str(path)])
    records = list(crawler())

    assert len(records) == 1
    record = records[0]
    assert record.media_type == "text/plain"
    assert record.path == path
    assert record.source_path == path
    assert record.title == "sample"


def test_run_main_pipeline_with_input_files_uses_file_crawler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Hello world. This is test text.", encoding="utf-8")

    processor = DummyProcessor()
    monkeypatch.setattr(main, "build_processor_pipeline", lambda: processor)

    result = main.run_main_pipeline(query=None, input_files=[str(path)])

    assert result.stats.processed_documents == 1
    assert result.stats.failed_documents == 0
    assert processor.ran == 1
    assert result.contexts[0].document.path == path

from __future__ import annotations

import pytest

from extractor import base
from extractor.base import BaseExtractor, ExtractorRegistry, create_extractor, register_extractor
from utils.models import DocumentRecord, ExtractionResult


class DummyExtractor(BaseExtractor):
    name = "dummy"

    def supports(self, document: DocumentRecord) -> bool:
        return True

    def extract(self, document: DocumentRecord) -> ExtractionResult:
        return ExtractionResult(document=document, raw_text="", clean_text="")


class AlternativeDummy(DummyExtractor):
    name = "dummy"


def test_register_inserts_mapping():
    reg = ExtractorRegistry()
    reg.register(DummyExtractor.name, DummyExtractor)
    assert reg._registry["dummy"] is DummyExtractor  # type: ignore[attr-defined]


def test_register_duplicate_without_replace_raises():
    reg = ExtractorRegistry()
    reg.register(DummyExtractor.name, DummyExtractor)
    with pytest.raises(ValueError):
        reg.register(DummyExtractor.name, DummyExtractor)


def test_register_with_replace_overwrites():
    reg = ExtractorRegistry()
    reg.register(DummyExtractor.name, DummyExtractor)
    reg.register(DummyExtractor.name, AlternativeDummy, replace=True)
    assert reg._registry["dummy"] is AlternativeDummy  # type: ignore[attr-defined]


def test_create_extractor_uses_global_registry(monkeypatch):
    reg = ExtractorRegistry()
    monkeypatch.setattr(base, "registry", reg)
    register_extractor(DummyExtractor.name, DummyExtractor)
    instance = create_extractor("dummy")
    assert isinstance(instance, DummyExtractor)


def test_create_extractor_unknown_alias(monkeypatch):
    reg = ExtractorRegistry()
    monkeypatch.setattr(base, "registry", reg)
    with pytest.raises(KeyError, match="Extractor 'missing' is not registered"):
        create_extractor("missing")

# utils\models.py
"""Core data models used throughout the pipeline.

These dataclasses describe documents, extraction results, questions, and the
question-set objects passed between stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass(slots=True)
class DocumentRecord:
    """Metadata and file information for a crawled or ingested document."""

    id: str
    title: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    media_type: str | None = None
    path: Path | None = None
    source_path: Path | None = None 
    content: str | None = None
    encoding: str | None = None

    def __post_init__(self) -> None:
        """Normalize path-like values and keep source and target paths consistent."""
        if self.path is not None and not isinstance(self.path, Path):
            self.path = Path(self.path)

        if self.source_path is not None and not isinstance(self.source_path, Path):
            self.source_path = Path(self.source_path)

        # Keep path/source_path in sync
        if self.path is None and self.source_path is not None:
            self.path = self.source_path
        elif self.source_path is None and self.path is not None:
            self.source_path = self.path
            
    @property
    def url(self) -> str | None:
        """Return the document path as a URL-like string when available."""
        if self.source_path:
            return str(self.source_path)
        if self.path:
            return str(self.path)
        return None

    @property
    def timestamp(self) -> Optional[str]:
        """Return the document timestamp stored in metadata, if present."""
        return self.metadata.get("timestamp")
    


@dataclass(slots=True)
class ExtractionResult:
    """Container for the text and metadata produced by an extractor stage."""

    document: DocumentRecord
    raw_text: str
    clean_text: str
    tokens: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    keyword_scores: Dict[str, float] = field(default_factory=dict)
    summary: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    # NEW optional fields for ranking / question generation
    top_sentences: List[str] = field(default_factory=list)
    sentence_scores: Dict[str, float] = field(default_factory=dict)

    questions: List["QuestionItem"] = field(default_factory=list)


    @property
    def document_id(self) -> str:
        """Return the owning document identifier."""
        return self.document.id


@dataclass(slots=True)
class Question:
    """Simple question representation with optional answer and metadata."""

    prompt: str
    answer: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class QuestionSet:
    """A collection of generated questions for a single document."""

    document: DocumentRecord
    questions: List["QuestionItem"]
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    strategy: Optional[str] = None

@dataclass
class QuestionItem:
    """Canonical shape for pipeline-generated question and answer pairs."""
    question: str
    answer: str
    source_document_id: str
    source_sentence: str
    confidence: float
    tags: List[str] = field(default_factory=list)


__all__ = ["DocumentRecord", "ExtractionResult", "Question", "QuestionSet", "QuestionItem"]
